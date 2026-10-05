"""Functional masked-categorical PPO transaction and q-step lookahead.

This is the differentiable counterpart of
:meth:`lira.envs.harvest.learner.CategoricalLearner.transaction`.  Starting
from a snapshot of the live learner (actors, critics, Adam moments, shared
dual), it collects ``q`` fresh training episodes, each followed by one
functional PPO/critic/Adam/dual update under responsibility logits that are
a differentiable function of the tangent coordinate ``eta``; then it collects
terminal episodes and forms

* the direct objective (terminal joint-welfare policy score through the final
  functional learner state), and
* the score-function correction for the training batches' dependence on the
  allocation (welfare times the summed behavior log-probabilities).

Unlike the live learner's ``_actor_loss``, which detaches the combined
advantage, the functional loss keeps ``advantage = A_R - N*lambda*rho_i*A_C``
attached so that the gradient flows from the PPO update back to the
responsibility logits.  The live learner is never mutated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Protocol, Sequence

import torch
from torch.func import functional_call

from lira.responsibility import simplex_from_logits

from .functional_adam import FunctionalParameterCollection, _safe_sqrt_collection_update
from .interfaces import AdapterStep, StepObservation
from .learner import MaskedCategoricalPolicy, ValueCritic
from .meta_unroll import (
    ProbabilityTangentChart,
    SamplingCorrectionSpec,
    TerminalScore,
    compose_score_terms,
    terminal_joint_welfare_score,
)


N_CONSTRAINTS = 1


class CategoricalEnvLike(Protocol):
    """Structural protocol satisfied by the Harvest learning adapter."""

    def reset(self, seed: int) -> StepObservation: ...

    def step(self, actions: Sequence[int]) -> AdapterStep: ...


@dataclass(frozen=True)
class FunctionalTransactionState:
    """Immutable functional actor/critic/Adam/dual state for the categorical transaction.

    ``actor_parameters`` / ``reward_critics`` / ``cost_critics`` have one
    ``FunctionalParameterCollection`` per agent (length N).
    ``lambda_values`` is the K=1 shared dual (never forked per agent, per the
    ``SharedLambda`` pattern in ``lira.responsibility``).  ``duplicated_lambdas``
    is carried through for state fidelity with ``CategoricalLearner`` (which
    also tracks the per-agent duals of the PAL arm) but is not consumed by
    the lookahead math in this module.
    """

    actor_parameters: tuple[FunctionalParameterCollection, ...]
    reward_critics: tuple[FunctionalParameterCollection, ...]
    cost_critics: tuple[FunctionalParameterCollection, ...]
    lambda_values: torch.Tensor
    duplicated_lambdas: torch.Tensor
    step_count: int
    rollout_counter: int

    def __post_init__(self) -> None:
        n = len(self.actor_parameters)
        if n < 1 or len(self.reward_critics) != n or len(self.cost_critics) != n:
            raise ValueError("functional transaction state requires matched N-agent collections")
        if tuple(self.lambda_values.shape) != (N_CONSTRAINTS,):
            raise ValueError("shared lambda must have shape (1,)")
        if tuple(self.duplicated_lambdas.shape) != (n,):
            raise ValueError("duplicated lambdas must have shape (N,)")
        if not torch.isfinite(self.lambda_values).all() or torch.any(self.lambda_values < 0):
            raise ValueError("shared lambda must be finite and nonnegative")
        if isinstance(self.step_count, bool) or self.step_count < 0:
            raise ValueError("functional transaction step_count must be a nonnegative int")
        if isinstance(self.rollout_counter, bool) or self.rollout_counter < 0:
            raise ValueError("functional transaction rollout_counter must be a nonnegative int")

    @property
    def n_agents(self) -> int:
        return len(self.actor_parameters)

    @classmethod
    def from_modules(
        cls,
        actors: Sequence[MaskedCategoricalPolicy],
        actor_optimizers: Sequence[torch.optim.Optimizer],
        reward_critics: Sequence[ValueCritic],
        reward_optimizers: Sequence[torch.optim.Optimizer],
        cost_critics: Sequence[ValueCritic],
        cost_optimizers: Sequence[torch.optim.Optimizer],
        lambda_values: torch.Tensor,
        duplicated_lambdas: torch.Tensor,
        *,
        step_count: int = 0,
        rollout_counter: int = 0,
    ) -> "FunctionalTransactionState":
        return cls(
            tuple(FunctionalParameterCollection.from_module(m, o) for m, o in zip(actors, actor_optimizers, strict=True)),
            tuple(FunctionalParameterCollection.from_module(m, o) for m, o in zip(reward_critics, reward_optimizers, strict=True)),
            tuple(FunctionalParameterCollection.from_module(m, o) for m, o in zip(cost_critics, cost_optimizers, strict=True)),
            lambda_values.detach().clone(), duplicated_lambdas.detach().clone(),
            step_count, rollout_counter,
        )

    @classmethod
    def from_learner(cls, learner: Any) -> "FunctionalTransactionState":
        """Snapshot a ``CategoricalLearner``-shaped object (duck-typed)."""
        return cls.from_modules(
            learner.actor, learner.actor_optimizers,
            learner.reward_critic, learner.reward_optimizers,
            learner.cost_critic, learner.cost_optimizers,
            learner.shared_lambda.values, learner.duplicated_lambdas,
            step_count=learner.transaction_counter, rollout_counter=learner.rollout_counter,
        )


@dataclass(frozen=True)
class CategoricalTape:
    """Tagged uniforms for masked-categorical sampling; mirrors ``CategoricalRolloutTape``."""

    reset_seed: int
    uniforms: torch.Tensor  # T x N

    def validate(self, *, horizon: int, n_agents: int) -> None:
        if self.uniforms.shape != (horizon, n_agents):
            raise ValueError("categorical tape has the wrong shape")
        if not self.uniforms.is_floating_point() or not torch.isfinite(self.uniforms).all():
            raise ValueError("categorical tape must be finite floating point")
        if torch.any((self.uniforms < 0) | (self.uniforms >= 1)):
            raise ValueError("categorical tape uniforms must lie in [0,1)")

    def tape_hash(self) -> str:
        return hashlib.sha256(
            str(self.reset_seed).encode("utf-8") + self.uniforms.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()


@dataclass(frozen=True)
class FunctionalRawRollout:
    """Frozen, detached environment samples for a typed functional replay."""

    observations: torch.Tensor   # T x N x D
    actions: torch.Tensor        # T x N int64
    action_masks: torch.Tensor   # T x N x C bool
    rewards: torch.Tensor        # T x N
    shared_damage: torch.Tensor  # T
    dones: torch.Tensor          # T
    reset_seed: int
    tape_hash: str

    def validate(self, *, horizon: int, n_agents: int, obs_dim: int, categories: int) -> None:
        expected = {
            "observations": (horizon, n_agents, obs_dim), "actions": (horizon, n_agents),
            "action_masks": (horizon, n_agents, categories), "rewards": (horizon, n_agents),
            "shared_damage": (horizon,), "dones": (horizon,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or value.requires_grad:
                raise ValueError(f"functional raw rollout has invalid {name}")
        if self.actions.dtype != torch.int64 or self.action_masks.dtype != torch.bool:
            raise ValueError("functional raw rollout uses int64 actions and bool masks")
        for name in ("observations", "rewards", "shared_damage", "dones"):
            if not torch.isfinite(getattr(self, name).to(torch.float64)).all():
                raise FloatingPointError(f"functional raw rollout has non-finite {name}")
        if not torch.all(self.action_masks.any(dim=-1)):
            raise ValueError("functional raw rollout has an all-invalid action mask")
        if not torch.all(self.action_masks.gather(-1, self.actions[..., None]).squeeze(-1)):
            raise ValueError("functional raw rollout action is masked")
        if not isinstance(self.reset_seed, int) or len(self.tape_hash) != 64:
            raise ValueError("functional raw rollout provenance is invalid")


@dataclass(frozen=True)
class FunctionalBatch:
    """Differentiable-through-state per-step quantities for one collected episode."""

    observations: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    rewards: torch.Tensor
    shared_damage: torch.Tensor
    dones: torch.Tensor
    old_log_probs: torch.Tensor       # T x N, differentiable through actor state
    reward_values: torch.Tensor       # T x N, differentiable through critic state
    cost_values: torch.Tensor         # T x N, differentiable through critic state
    reward_advantages: torch.Tensor   # T x N
    cost_advantages: torch.Tensor     # T x N
    reward_returns: torch.Tensor      # T x N
    cost_returns: torch.Tensor        # T


@dataclass(frozen=True)
class FunctionalFreshRollout:
    raw: FunctionalRawRollout
    batch: FunctionalBatch
    state: FunctionalTransactionState
    behavior_joint_log_prob: torch.Tensor


@dataclass(frozen=True)
class _TerminalBatchView:
    """Duck-typed view handed to ``meta_unroll.terminal_joint_welfare_score``."""

    behavior: str
    training_eligible: bool
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


# ---------------------------------------------------------------------------
# Functional module evaluation helpers
# ---------------------------------------------------------------------------


def _net_mapping(params: FunctionalParameterCollection) -> dict[str, torch.Tensor]:
    """``FunctionalParameterCollection`` names are ``net.<...>``; strip for ``.net``."""
    return {name[len("net."):]: value for name, value in zip(params.names, params.values, strict=True)}


def _functional_masked_logits(
    policy: MaskedCategoricalPolicy, params: FunctionalParameterCollection,
    observations: torch.Tensor, masks: torch.Tensor,
) -> torch.Tensor:
    logits = functional_call(policy.net, _net_mapping(params), (observations,))
    return logits.masked_fill(~masks, float("-inf"))


def _functional_critic_value(
    critic: ValueCritic, params: FunctionalParameterCollection, observations: torch.Tensor,
) -> torch.Tensor:
    return functional_call(critic.net, _net_mapping(params), (observations,)).squeeze(-1)


def _discounted_cost_returns(shared_damage: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(shared_damage)
    running = torch.zeros((), dtype=shared_damage.dtype, device=shared_damage.device)
    for t in range(shared_damage.shape[0] - 1, -1, -1):
        running = shared_damage[t] + gamma * (1.0 - dones[t]) * running
        returns[t] = running
    return returns


def _build_batch(
    state: FunctionalTransactionState, learner: Any,
    raw: FunctionalRawRollout, old_log_probs: torch.Tensor,
) -> FunctionalBatch:
    cfg = learner.config
    n_agents = state.n_agents
    reward_values = torch.stack(
        [_functional_critic_value(learner.reward_critic[i], state.reward_critics[i], raw.observations[:, i]) for i in range(n_agents)],
        dim=1,
    )
    cost_values = torch.stack(
        [_functional_critic_value(learner.cost_critic[i], state.cost_critics[i], raw.observations[:, i]) for i in range(n_agents)],
        dim=1,
    )
    reward_adv = torch.zeros_like(raw.rewards)
    reward_returns = torch.zeros_like(raw.rewards)
    cost_adv = torch.zeros_like(raw.rewards)
    dtype, device = raw.rewards.dtype, raw.rewards.device
    next_r = torch.zeros(n_agents, dtype=dtype, device=device)
    next_c = torch.zeros(n_agents, dtype=dtype, device=device)
    gae_r = torch.zeros(n_agents, dtype=dtype, device=device)
    gae_c = torch.zeros(n_agents, dtype=dtype, device=device)
    for t in range(cfg.horizon - 1, -1, -1):
        not_done = 1.0 - raw.dones[t]
        delta_r = raw.rewards[t] + cfg.gamma * not_done * next_r - reward_values[t]
        delta_c = raw.shared_damage[t] + cfg.gamma * not_done * next_c - cost_values[t]
        gae_r = delta_r + cfg.gamma * cfg.gae_lambda * not_done * gae_r
        gae_c = delta_c + cfg.gamma * cfg.gae_lambda * not_done * gae_c
        reward_adv[t] = gae_r
        cost_adv[t] = gae_c
        reward_returns[t] = gae_r + reward_values[t]
        next_r, next_c = reward_values[t], cost_values[t]
    cost_returns = _discounted_cost_returns(raw.shared_damage, raw.dones, cfg.gamma)
    return FunctionalBatch(
        raw.observations, raw.actions, raw.action_masks, raw.rewards, raw.shared_damage, raw.dones,
        old_log_probs, reward_values, cost_values, reward_adv, cost_adv, reward_returns, cost_returns,
    )


def _extract_observation(
    value: StepObservation, *, n_agents: int, categories: int, obs_dim: int, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(value.observation_matrix, dtype=dtype)
    raw_masks = value.action_mask_matrix
    if raw_masks.ndim != 2 or raw_masks.shape[0] != n_agents or raw_masks.shape[1] > categories:
        raise ValueError("action mask shape exceeds the frozen categorical contract")
    masks = torch.zeros((n_agents, categories), dtype=torch.bool)
    masks[:, : raw_masks.shape[1]] = torch.as_tensor(raw_masks, dtype=torch.bool)
    if observations.shape != (n_agents, obs_dim) or not torch.isfinite(observations).all():
        raise ValueError("observation shape/dtype is outside the frozen contract")
    if not torch.all(masks.any(dim=-1)):
        raise ValueError("emitted an all-invalid action mask")
    return observations, masks


def _valid_prefix_mask(dones: torch.Tensor, horizon: int) -> torch.Tensor:
    """First-done-inclusive validity mask; mirrors ``terminal_joint_welfare_score``."""
    valid = torch.zeros(horizon, dtype=dones.dtype, device=dones.device)
    alive = True
    for t in range(horizon):
        if not alive:
            break
        valid[t] = 1.0
        alive = not bool(dones[t].detach().item())
    return valid


# ---------------------------------------------------------------------------
# Fresh / replay collection
# ---------------------------------------------------------------------------


def functional_fresh_rollout(
    learner: Any,
    state: FunctionalTransactionState,
    tape: CategoricalTape,
    *,
    env: CategoricalEnvLike | None = None,
) -> FunctionalFreshRollout:
    """Collect one fresh functional masked-categorical episode.

    ``learner`` is duck-typed: it must expose ``.actor`` / ``.reward_critic`` /
    ``.cost_critic`` (sequences of ``MaskedCategoricalPolicy`` / ``ValueCritic``,
    one per agent) and ``.config`` (with ``obs_dim``, ``categories``, ``horizon``,
    ``gamma``, ``gae_lambda``).  ``env`` defaults to ``learner.env``; the
    training script passes a separate adapter instance so the lookahead never
    touches the live environment.

    Sampled actions are detached before being sent to the environment and
    before being used as the fixed index into the behavior log-probability.
    Reward/cost critic values and the resulting old_log_probs /
    GAE quantities remain differentiable through ``state``.
    """
    environment = learner.env if env is None else env
    cfg = learner.config
    n_agents = state.n_agents
    tape.validate(horizon=cfg.horizon, n_agents=n_agents)
    dtype = state.actor_parameters[0].values[0].dtype
    obs_value = environment.reset(seed=tape.reset_seed)
    observations, masks = _extract_observation(
        obs_value, n_agents=n_agents, categories=cfg.categories, obs_dim=cfg.obs_dim, dtype=dtype,
    )
    obs_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    logprob_rows: list[torch.Tensor] = []
    reward_rows: list[torch.Tensor] = []
    shared_damage_rows: list[torch.Tensor] = []
    done_rows: list[torch.Tensor] = []
    terminated = False
    for step in range(cfg.horizon):
        obs_rows.append(observations.detach())
        mask_rows.append(masks)
        actions_list: list[torch.Tensor] = []
        logprob_list: list[torch.Tensor] = []
        for agent in range(n_agents):
            agent_obs = observations[agent : agent + 1]
            agent_mask = masks[agent : agent + 1]
            logits = _functional_masked_logits(learner.actor[agent], state.actor_parameters[agent], agent_obs, agent_mask)
            probs = torch.softmax(logits, dim=-1)
            uniforms = tape.uniforms[step, agent : agent + 1].to(dtype)
            action = (uniforms[:, None] >= probs.cumsum(dim=-1)).sum(dim=-1).clamp(max=cfg.categories - 1).to(torch.int64)
            action = action.detach()
            log_prob = torch.distributions.Categorical(logits=logits).log_prob(action)[0]
            actions_list.append(action[0])
            logprob_list.append(log_prob)
        actions = torch.stack(actions_list).to(torch.int64)
        step_out = environment.step(actions.detach().tolist())
        if not isinstance(step_out, AdapterStep):
            raise TypeError("env must return AdapterStep")
        action_rows.append(actions.detach())
        logprob_rows.append(torch.stack(logprob_list))
        rewards = torch.as_tensor(step_out.rewards, dtype=dtype)
        if rewards.shape != (n_agents,) or not torch.isfinite(rewards).all():
            raise ValueError("categorical env emitted an invalid per-agent reward vector")
        reward_rows.append(rewards)
        shared_damage_rows.append(torch.tensor(float(step_out.transition.team_damage), dtype=dtype))
        done_rows.append(torch.tensor(float(step_out.terminated), dtype=dtype))
        terminated = bool(step_out.terminated)
        if terminated:
            break
        observations, masks = _extract_observation(
            step_out.observation, n_agents=n_agents, categories=cfg.categories, obs_dim=cfg.obs_dim, dtype=dtype,
        )
    steps = len(action_rows)

    def pad_zero(rows: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
        if len(rows) == cfg.horizon:
            return torch.stack(rows)
        filler = torch.zeros(shape, dtype=dtype)
        return torch.stack(rows + [filler] * (cfg.horizon - len(rows)))

    observations_tensor = pad_zero(obs_rows, (n_agents, cfg.obs_dim))
    actions_tensor = (
        torch.stack(action_rows) if steps == cfg.horizon
        else torch.stack(action_rows + [torch.zeros(n_agents, dtype=torch.int64)] * (cfg.horizon - steps))
    )
    masks_tensor = (
        torch.stack(mask_rows) if steps == cfg.horizon
        else torch.stack(mask_rows + [mask_rows[-1]] * (cfg.horizon - steps))
    )
    rewards_tensor = pad_zero(reward_rows, (n_agents,))
    shared_damage_tensor = pad_zero(shared_damage_rows, ())
    dones_tensor = torch.stack(done_rows + [torch.ones((), dtype=dtype)] * (cfg.horizon - steps))
    old_log_probs = (
        torch.stack(logprob_rows) if steps == cfg.horizon
        else torch.stack(logprob_rows + [torch.zeros(n_agents, dtype=dtype)] * (cfg.horizon - steps))
    )

    raw = FunctionalRawRollout(
        observations_tensor, actions_tensor, masks_tensor, rewards_tensor, shared_damage_tensor,
        dones_tensor, tape.reset_seed, tape.tape_hash(),
    )
    raw.validate(horizon=cfg.horizon, n_agents=n_agents, obs_dim=cfg.obs_dim, categories=cfg.categories)
    batch = _build_batch(state, learner, raw, old_log_probs)
    advanced_state = replace(state, rollout_counter=state.rollout_counter + 1)
    return FunctionalFreshRollout(raw, batch, advanced_state, batch.old_log_probs.sum())


def functional_replay_raw_rollout(
    raw: FunctionalRawRollout, state: FunctionalTransactionState, learner: Any,
) -> FunctionalFreshRollout:
    """Replay fixed raw data while recomputing all learner-derived axes.

    This never calls the environment: it is the same-data functional
    finite-difference validation boundary.
    """
    cfg = learner.config
    n_agents = state.n_agents
    raw.validate(horizon=cfg.horizon, n_agents=n_agents, obs_dim=cfg.obs_dim, categories=cfg.categories)
    logprob_cols: list[torch.Tensor] = []
    for agent in range(n_agents):
        logits = _functional_masked_logits(
            learner.actor[agent], state.actor_parameters[agent], raw.observations[:, agent], raw.action_masks[:, agent],
        )
        logprob_cols.append(torch.distributions.Categorical(logits=logits).log_prob(raw.actions[:, agent]))
    old_log_probs = torch.stack(logprob_cols, dim=1)
    valid = _valid_prefix_mask(raw.dones, cfg.horizon)
    old_log_probs = old_log_probs * valid[:, None]
    batch = _build_batch(state, learner, raw, old_log_probs)
    advanced_state = replace(state, rollout_counter=state.rollout_counter + 1)
    return FunctionalFreshRollout(raw, batch, advanced_state, batch.old_log_probs.sum())


# ---------------------------------------------------------------------------
# Functional PPO-with-cost-penalty + critic + dual update
# ---------------------------------------------------------------------------


def _maybe_separate_axis(collection: FunctionalParameterCollection, partial_gradient: bool) -> FunctionalParameterCollection:
    """The zero-offset trick that isolates the ``learner`` axis from ``data``.

    When ``partial_gradient=True`` (used for a freshly
    collected differentiable batch whose ``old_log_probs`` already depend on
    the SAME leaf tensors), this builds a numerically-identical but
    graph-disconnected node so ``torch.autograd.grad(loss, learner_axis)``
    picks up only the "new policy" path, not the PPO ratio's behavior branch.
    """
    if not partial_gradient:
        return collection
    return FunctionalParameterCollection(
        collection.names, tuple(value + torch.zeros_like(value) for value in collection.values), collection.optimizers,
    )


def functional_frozen_transaction(
    batch: FunctionalBatch,
    state: FunctionalTransactionState,
    responsibility_logits: torch.Tensor,
    learner: Any,
    *,
    q: int = 1,
    partial_gradient: bool = False,
    fixed_lambda: bool = False,
) -> FunctionalTransactionState:
    """Functional reimplementation of ``CategoricalLearner.transaction``'s PPO math.

    ``responsibility_logits`` is the differentiable meta-parameter; the
    per-agent coefficient ``N * lambda * rho[agent]`` (``rho`` from
    ``responsibility.simplex_from_logits``, never forked per agent for
    ``lambda``) is combined with the reward/cost GAE advantage WITHOUT the
    ``.detach()`` that ``CategoricalLearner._actor_loss`` applies for live
    training -- see the module docstring for why that detach must be omitted
    here for Direct-Unroll to have a nonzero meta-gradient at all.
    """
    cfg = learner.config
    n_agents = state.n_agents
    if responsibility_logits.shape != (N_CONSTRAINTS, n_agents) or not torch.isfinite(responsibility_logits).all():
        raise ValueError("invalid functional responsibility logits")
    if q < 1:
        raise ValueError("functional frozen transaction q must be positive")
    active = (batch.dones == 0).to(batch.rewards.dtype)
    active_count = active.sum().clamp_min(1.0)
    current = state
    for _ in range(q):
        rho = simplex_from_logits(responsibility_logits, cfg.rho_floor)[0]
        actor_states = list(current.actor_parameters)
        reward_states = list(current.reward_critics)
        cost_states = list(current.cost_critics)
        for agent in range(n_agents):
            obs = batch.observations[:, agent]
            reward_target = batch.reward_returns[:, agent]
            cost_target = batch.cost_returns

            reward_params = _maybe_separate_axis(reward_states[agent], partial_gradient)
            reward_pred = _functional_critic_value(learner.reward_critic[agent], reward_params, obs)
            reward_loss = (((reward_pred - reward_target) ** 2) * active).sum() / active_count
            reward_grad = torch.autograd.grad(reward_loss, reward_params.values, create_graph=True)
            reward_states[agent] = _safe_sqrt_collection_update(reward_params, reward_grad)

            cost_params = _maybe_separate_axis(cost_states[agent], partial_gradient)
            cost_pred = _functional_critic_value(learner.cost_critic[agent], cost_params, obs)
            cost_loss = (((cost_pred - cost_target) ** 2) * active).sum() / active_count
            cost_grad = torch.autograd.grad(cost_loss, cost_params.values, create_graph=True)
            cost_states[agent] = _safe_sqrt_collection_update(cost_params, cost_grad)

            coefficient = n_agents * current.lambda_values[0] * rho[agent]
            actor_params = _maybe_separate_axis(actor_states[agent], partial_gradient)
            logits = _functional_masked_logits(learner.actor[agent], actor_params, obs, batch.action_masks[:, agent])
            distribution = torch.distributions.Categorical(logits=logits)
            logp = distribution.log_prob(batch.actions[:, agent])
            entropy = distribution.entropy()
            ratio = torch.exp(logp - batch.old_log_probs[:, agent])
            # Intentionally NOT detached; see the module and function docstrings.
            advantage = batch.reward_advantages[:, agent] - coefficient * batch.cost_advantages[:, agent]
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
            surrogate = torch.minimum(ratio * advantage, clipped * advantage)
            actor_loss = (
                -(surrogate * active).sum() / active_count
                - cfg.entropy_coefficient * (entropy * active).sum() / active_count
            )
            actor_grad = torch.autograd.grad(actor_loss, actor_params.values, create_graph=True)
            actor_states[agent] = _safe_sqrt_collection_update(actor_params, actor_grad)

        lambda_values = current.lambda_values
        if not fixed_lambda:
            episode_damage = batch.shared_damage.sum()
            lambda_values = torch.clamp(lambda_values + cfg.lambda_lr * (episode_damage - cfg.damage_limit), min=0.0)
        current = FunctionalTransactionState(
            tuple(actor_states), tuple(reward_states), tuple(cost_states),
            lambda_values, current.duplicated_lambdas, current.step_count + 1, current.rollout_counter,
        )
    return current


# ---------------------------------------------------------------------------
# q-step Direct-Unroll / DU+SC orchestration
# ---------------------------------------------------------------------------


def _terminal_score(batch: FunctionalBatch) -> TerminalScore:
    view = _TerminalBatchView("on_policy", True, batch.old_log_probs, batch.rewards, batch.dones)
    return terminal_joint_welfare_score(view)


def _validate_disjoint_tapes(
    update_tapes: Sequence[CategoricalTape], terminal_tapes: Sequence[CategoricalTape],
    *, horizon: int, n_agents: int,
) -> None:
    """Require training (update) and terminal sample streams to be disjoint.

    A training sample stream is never reused as a terminal (evaluation)
    stream.  The sample-stream identity is the tape's ``reset_seed`` (what actually determines the
    environment trajectory); this checks BOTH full-tape-content disjointness
    (``tape_hash``, catching an accidentally duplicated uniforms draw) and the
    stricter, seed-level invariant the task requires: a ``reset_seed`` used
    for a training update must never also be used for a held-out evaluation,
    even with different uniforms.
    """
    if not update_tapes or not terminal_tapes:
        raise ValueError("direct unroll requires nonempty update and terminal tapes")
    update_hashes: list[str] = []
    update_seeds: list[int] = []
    for tape in update_tapes:
        tape.validate(horizon=horizon, n_agents=n_agents)
        digest = tape.tape_hash()
        if digest in update_hashes:
            raise ValueError("update tapes must be distinct within one transaction")
        update_hashes.append(digest)
        update_seeds.append(tape.reset_seed)
    terminal_hashes: list[str] = []
    terminal_seeds: list[int] = []
    for tape in terminal_tapes:
        tape.validate(horizon=horizon, n_agents=n_agents)
        digest = tape.tape_hash()
        if digest in terminal_hashes:
            raise ValueError("terminal tapes must be distinct within one transaction")
        terminal_hashes.append(digest)
        terminal_seeds.append(tape.reset_seed)
    if set(update_hashes) & set(terminal_hashes):
        raise ValueError("update and terminal tapes must be disjoint")
    if set(update_seeds) & set(terminal_seeds):
        raise ValueError("update and terminal reset_seed streams must be disjoint")


@dataclass(frozen=True)
class DirectUnrollResult:
    """A q-step functional learner trace; no outer allocation update."""

    initial_state: FunctionalTransactionState
    final_state: FunctionalTransactionState
    update_raw: tuple[FunctionalRawRollout, ...]
    terminal_raw: tuple[FunctionalRawRollout, ...]
    behavior_joint_log_probs: tuple[torch.Tensor, ...]
    terminal_scores: tuple[TerminalScore, ...]
    direct_objective: torch.Tensor
    correction_objective: torch.Tensor
    corrected_objective: torch.Tensor

    def gradient(self, meta_parameter: torch.Tensor, *, corrected: bool = False) -> torch.Tensor:
        if not meta_parameter.requires_grad:
            raise ValueError("gradient target must require gradients")
        target = self.corrected_objective if corrected else self.direct_objective
        derivative = torch.autograd.grad(target, meta_parameter, retain_graph=True, allow_unused=True)[0]
        return torch.zeros_like(meta_parameter) if derivative is None else derivative


def run_direct_unroll(
    learner: Any,
    responsibility_logits: torch.Tensor,
    *,
    update_tapes: Sequence[CategoricalTape],
    terminal_tapes: Sequence[CategoricalTape],
    fixed_raw_updates: Sequence[FunctionalRawRollout] | None = None,
    fixed_raw_terminals: Sequence[FunctionalRawRollout] | None = None,
    state: FunctionalTransactionState | None = None,
    fixed_lambda: bool = False,
    sampling_correction: SamplingCorrectionSpec | None = None,
    env: CategoricalEnvLike | None = None,
) -> DirectUnrollResult:
    """Execute q fresh (or replayed) functional updates and build the Direct/SC objectives.

    For every update tape, a fresh functional rollout is collected (or, if
    ``fixed_raw_updates`` is supplied, replayed with no environment call),
    followed by ONE functional ``functional_frozen_transaction`` (``q=1``,
    ``partial_gradient=True``).  Every rollout (update or terminal) uses the
    same ``functional_fresh_rollout``/``functional_replay_raw_rollout``, which
    handles the first-``terminated``/padding boundary uniformly.  Supplying
    ``fixed_raw_updates``/``fixed_raw_terminals`` replays byte-identical raw
    environment samples across perturbed ``eta`` evaluations (useful for
    finite-difference checks).
    """
    correction_spec = SamplingCorrectionSpec() if sampling_correction is None else sampling_correction
    if (fixed_raw_updates is None) != (fixed_raw_terminals is None):
        raise ValueError("fixed raw updates and terminals must be supplied together")
    current = FunctionalTransactionState.from_learner(learner) if state is None else state
    n_agents = current.n_agents
    if responsibility_logits.shape != (N_CONSTRAINTS, n_agents) or not torch.isfinite(responsibility_logits).all():
        raise ValueError("invalid responsibility logits")
    _validate_disjoint_tapes(update_tapes, terminal_tapes, horizon=learner.config.horizon, n_agents=n_agents)
    if fixed_raw_updates is not None:
        if len(fixed_raw_updates) != len(update_tapes) or len(fixed_raw_terminals or ()) != len(terminal_tapes):
            raise ValueError("fixed raw replay length does not match tape manifest")
    initial = current
    scores: list[torch.Tensor] = []
    update_raw: list[FunctionalRawRollout] = []
    for index, tape in enumerate(update_tapes):
        fresh = (
            functional_fresh_rollout(learner, current, tape, env=env) if fixed_raw_updates is None
            else functional_replay_raw_rollout(fixed_raw_updates[index], current, learner)
        )
        current = functional_frozen_transaction(
            fresh.batch, fresh.state, responsibility_logits, learner,
            q=1, partial_gradient=True, fixed_lambda=fixed_lambda,
        )
        update_raw.append(fresh.raw)
        scores.append(fresh.behavior_joint_log_prob)

    terminal_scores: list[TerminalScore] = []
    terminal_raw: list[FunctionalRawRollout] = []
    for index, tape in enumerate(terminal_tapes):
        fresh_terminal = (
            functional_fresh_rollout(learner, current, tape, env=env) if fixed_raw_terminals is None
            else functional_replay_raw_rollout(fixed_raw_terminals[index], current, learner)
        )
        terminal_raw.append(fresh_terminal.raw)
        terminal_scores.append(_terminal_score(fresh_terminal.batch))

    direct = torch.stack([item.objective for item in terminal_scores]).mean()
    terminal_welfare = torch.stack([item.welfare for item in terminal_scores]).mean().detach()
    direct, correction, corrected = compose_score_terms(direct, terminal_welfare, scores, correction_spec=correction_spec)
    return DirectUnrollResult(
        initial, current, tuple(update_raw), tuple(terminal_raw), tuple(scores), tuple(terminal_scores),
        direct, correction, corrected,
    )


def _learner_parameter_dtype(learner: Any) -> torch.dtype:
    return next(learner.actor[0].parameters()).dtype


def _chart_logits_at_dtype(chart: ProbabilityTangentChart, eta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Compute ``chart.logits(eta)`` and cast to ``dtype`` (a differentiable op).

    ``ProbabilityTangentChart.__post_init__`` requires its center to sum to
    exactly one at ``atol=1e-12, rtol=0.0``.  A float32 uniform center's sum
    can be off by about one ULP (softmax rounding), so the chart is built and
    differentiated in float64, and only the resulting ``logits`` are cast down
    to the learner's float32 parameter dtype before entering the functional
    transaction.  The cast is a normal differentiable ``.to(dtype)``, so
    ``eta``'s gradient is unaffected; only its numerical precision is capped
    at whatever the downstream float32 computation itself allows.
    """
    logits = chart.logits(eta)
    return logits if logits.dtype == dtype else logits.to(dtype)


def execute_direct_unroll(
    learner: Any, chart: ProbabilityTangentChart, eta: torch.Tensor, *,
    update_tapes: Sequence[CategoricalTape], terminal_tapes: Sequence[CategoricalTape],
    state: FunctionalTransactionState | None = None, fixed_lambda: bool = False,
    env: CategoricalEnvLike | None = None,
) -> tuple[torch.Tensor, DirectUnrollResult]:
    """Direct-unroll (DU) only; no sampling-correction objective is formed.

    Returns ``(d(direct_objective)/d(eta), result)``.
    """
    logits = _chart_logits_at_dtype(chart, eta, _learner_parameter_dtype(learner))
    result = run_direct_unroll(
        learner, logits, update_tapes=update_tapes, terminal_tapes=terminal_tapes, state=state,
        fixed_lambda=fixed_lambda, sampling_correction=SamplingCorrectionSpec.disabled(), env=env,
    )
    return result.gradient(eta), result


def execute_direct_unroll_sampling_correction(
    learner: Any, chart: ProbabilityTangentChart, eta: torch.Tensor, *,
    update_tapes: Sequence[CategoricalTape], terminal_tapes: Sequence[CategoricalTape],
    state: FunctionalTransactionState | None = None, fixed_lambda: bool = False,
    env: CategoricalEnvLike | None = None,
) -> tuple[torch.Tensor, DirectUnrollResult]:
    """DU + score correction with a zero baseline (the leave-one-out baseline
    is applied afterwards by :mod:`lira.envs.harvest.meta_batch`)."""
    logits = _chart_logits_at_dtype(chart, eta, _learner_parameter_dtype(learner))
    result = run_direct_unroll(
        learner, logits, update_tapes=update_tapes, terminal_tapes=terminal_tapes, state=state,
        fixed_lambda=fixed_lambda, sampling_correction=SamplingCorrectionSpec(), env=env,
    )
    return result.gradient(eta, corrected=True), result


__all__ = [
    "N_CONSTRAINTS",
    "CategoricalEnvLike",
    "CategoricalTape",
    "DirectUnrollResult",
    "FunctionalBatch",
    "FunctionalFreshRollout",
    "FunctionalRawRollout",
    "FunctionalTransactionState",
    "execute_direct_unroll",
    "execute_direct_unroll_sampling_correction",
    "functional_fresh_rollout",
    "functional_replay_raw_rollout",
    "run_direct_unroll",
    "functional_frozen_transaction",
]
