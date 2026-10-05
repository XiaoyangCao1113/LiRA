"""Fixed-rollout direct-unroll (DU) estimator for the MetaDrive learner.

Differentiates a finite q-step PPO/Adam lookahead with respect to the K=1
responsibility logits while treating sampled trajectories as constants: the
environment only supplies the lookahead tapes, and no pathwise environment
derivative is used. The terminal welfare enters through the on-policy
likelihood-ratio score of a fixed evaluation rollout (the DU term). With
sampling correction enabled, the score of the q sampled lookahead rollouts is
added (the SC term); :func:`run_independent_du_sc` averages M independent
lookaheads and uses a leave-one-out (LOO) welfare baseline. Raw environment
samples remain detached; only policy log-probability scores are live.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from ...ppo import PPOBatch
from ...responsibility import simplex_from_logits
from .learner import MetaDriveStaticLearner


@dataclass(frozen=True)
class MetaDriveDUConfig:
    """Protocol of one MetaDrive direct-unroll lookahead."""

    q: int = 2
    sampling_correction: bool = False
    correction_baseline: str = "zero"
    response_method: str = "direct_unroll"
    target: str = "finite_rollout_welfare_score"

    def __post_init__(self) -> None:
        if self.q < 1:
            raise ValueError("q must be positive")
        if self.correction_baseline not in {"zero", "loo"}:
            raise ValueError("correction_baseline must be 'zero' or 'loo'")
        if self.response_method != "direct_unroll":
            raise ValueError("MetaDrive DU adapter requires response_method='direct_unroll'")
        if self.target != "finite_rollout_welfare_score":
            raise ValueError("unsupported MetaDrive DU target")


@dataclass(frozen=True)
class MetaDriveRolloutTape:
    """A detached fixed trajectory consumed by the direct-unroll lookahead."""

    obs: torch.Tensor
    final_obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor

    @classmethod
    def from_rollout(cls, rollout: Mapping[str, Any], dtype: torch.dtype) -> "MetaDriveRolloutTape":
        keys = ("obs", "final_obs", "actions", "old_log_probs", "rewards", "costs")
        if any(key not in rollout for key in keys):
            raise ValueError("rollout is missing a DU tape field")
        steps = int(rollout["steps"])
        if steps < 1:
            raise ValueError("DU tapes must contain at least one step")
        dones = torch.zeros(steps, dtype=torch.bool)
        if bool(rollout.get("done", False)):
            dones[-1] = True
        tape = cls(
            *(torch.as_tensor(rollout[key], dtype=dtype).detach().clone() for key in keys),
            dones=dones,
        )
        tape.validate()
        return tape

    def validate(self) -> None:
        t, n, d = self.obs.shape
        if self.final_obs.shape != (n, d) or self.actions.ndim != 3 or self.actions.shape[0] != t:
            raise ValueError("invalid MetaDrive DU tape shapes")
        if self.old_log_probs.shape != (t, n) or self.rewards.shape != (t, n):
            raise ValueError("invalid MetaDrive DU per-agent tape shapes")
        if self.costs.shape != (t, 1) or self.dones.shape != (t,):
            raise ValueError("invalid MetaDrive DU cost/done tape shapes")
        tensors = (self.obs, self.final_obs, self.actions, self.old_log_probs, self.rewards, self.costs)
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("DU tape contains non-finite values")

    def digest(self) -> str:
        payload = {name: getattr(self, name).detach().cpu().numpy().tobytes().hex() for name in (
            "obs", "final_obs", "actions", "old_log_probs", "rewards", "costs", "dones"
        )}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class MetaDriveDUResult:
    gradient: torch.Tensor
    terminal_welfare: float
    tape_digests: tuple[str, ...]
    evaluation_digest: str
    q: int
    estimator: str = "direct_unroll"
    sampling_correction: bool = False
    target: str = "finite_rollout_welfare_score"
    direct_gradient: torch.Tensor | None = None
    score_gradient: torch.Tensor | None = None
    score_value: float = 0.0
    baseline: float = 0.0


def capture_tapes(
    learner: MetaDriveStaticLearner,
    *,
    update_seeds: Sequence[int],
    evaluation_seed: int,
) -> tuple[tuple[MetaDriveRolloutTape, ...], MetaDriveRolloutTape]:
    """Collect fixed update/evaluation tapes while preserving all RNG streams."""
    if not update_seeds:
        raise ValueError("at least one update seed is required")
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    try:
        updates = tuple(MetaDriveRolloutTape.from_rollout(learner.rollout(int(seed)), learner.dtype) for seed in update_seeds)
        evaluation = MetaDriveRolloutTape.from_rollout(learner.rollout(int(evaluation_seed)), learner.dtype)
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
    return updates, evaluation


def _parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value for name, value in module.named_parameters()}


def _zeros_like_parameters(params: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value) for name, value in params.items()}


class _SafeSqrt(torch.autograd.Function):
    """``sqrt`` whose forward is bit-for-bit ``torch.sqrt`` and whose backward
    is zero (instead of singular) exactly at ``v == 0``.

    ``_adam_step`` below differentiates ``exp_avg_sq.sqrt()`` a second time
    (``MetaDriveDirectUnroll.run`` needs ``create_graph=True`` through the
    inner functional Adam steps). ``d(sqrt(x))/dx`` is singular at ``x == 0``,
    and for real MetaDrive observations some input feature can read exactly
    ``0.0`` for an entire tape (e.g. a lidar channel with nothing in range),
    so the matching first-layer weight column keeps ``exp_avg_sq == 0`` and a
    plain ``sqrt`` would produce NaN. Flooring the second moment would
    perturb the forward Adam update for every parameter; ``_SafeSqrt``
    instead keeps the forward pass exactly ``torch.sqrt`` and only redefines
    the derivative at the single undefined point.
    """

    @staticmethod
    def forward(ctx: Any, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v)
        return torch.sqrt(v)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (v,) = ctx.saved_tensors
        is_zero = v == 0
        safe_v = torch.where(is_zero, torch.ones_like(v), v)
        grad_v = 0.5 * grad_output / torch.sqrt(safe_v)
        return torch.where(is_zero, torch.zeros_like(grad_v), grad_v)


def _adam_step(
    params: Mapping[str, torch.Tensor],
    grads: Sequence[torch.Tensor],
    state: Mapping[str, tuple[torch.Tensor, torch.Tensor, int]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor, int]]]:
    """Differentiable Adam transition, including a graph through q updates."""
    updated: dict[str, torch.Tensor] = {}
    next_state: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for (name, parameter), grad in zip(params.items(), grads):
        old_m, old_v, step = state[name]
        step += 1
        m = beta1 * old_m + (1.0 - beta1) * grad
        v = beta2 * old_v + (1.0 - beta2) * grad.square()
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        denom = _SafeSqrt.apply(v) / (bias2**0.5) + eps
        updated[name] = parameter - lr * (m / bias1) / denom
        next_state[name] = (m, v, step)
    return updated, next_state


def _initial_adam_state(module: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, tuple[torch.Tensor, torch.Tensor, int]]:
    params = list(module.parameters())
    state_by_id = optimizer.state
    result = {}
    for name, parameter in module.named_parameters():
        raw = state_by_id.get(parameter, {})
        result[name] = (
            torch.as_tensor(raw.get("exp_avg", torch.zeros_like(parameter)), dtype=parameter.dtype),
            torch.as_tensor(raw.get("exp_avg_sq", torch.zeros_like(parameter)), dtype=parameter.dtype),
            int(raw.get("step", 0)),
        )
    return result


def _actor_distribution(actor: nn.Module, params: Mapping[str, torch.Tensor], obs: torch.Tensor) -> torch.distributions.Normal:
    """Functional call for ``_GaussianActor`` (which exposes distribution(), not forward())."""
    network_params = {name.removeprefix("net."): value for name, value in params.items() if name.startswith("net.")}
    mean = functional_call(actor.net, network_params, (obs,))
    return torch.distributions.Normal(mean, params["log_std"].exp())


def _gae(rewards: torch.Tensor, values: torch.Tensor, bootstrap: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    next_value = bootstrap.detach()
    next_adv = torch.zeros_like(bootstrap)
    for index in range(rewards.shape[0] - 1, -1, -1):
        mask = (~dones[index]).to(rewards.dtype)
        delta = rewards[index] + gamma * next_value * mask - values[index]
        next_adv = delta + gamma * lam * mask * next_adv
        advantages[index] = next_adv
        next_value = values[index]
    return advantages.detach(), (advantages + values).detach()


def _fixed_advantages(
    learner: MetaDriveStaticLearner,
    params_actors: Sequence[Mapping[str, torch.Tensor]],
    params_critic: Mapping[str, torch.Tensor],
    tape: MetaDriveRolloutTape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the learner's no-grad advantage/return construction."""
    cfg = learner.config
    with torch.no_grad():
        values = functional_call(learner.critic, params_critic, (tape.obs.reshape(tape.obs.shape[0], -1),))
        values = values.reshape(tape.obs.shape[0], cfg.n_agents, 1 + cfg.n_constraints)
        reward_values, cost_values = values[..., 0], values[..., 1:]
        final = functional_call(learner.critic, params_critic, (tape.final_obs.reshape(1, -1),))
        final = final.reshape(cfg.n_agents, 1 + cfg.n_constraints)
        bootstrap_r = torch.zeros(cfg.n_agents, dtype=learner.dtype) if tape.dones[-1] else final[:, 0]
        bootstrap_c = torch.zeros(cfg.n_agents, cfg.n_constraints, dtype=learner.dtype) if tape.dones[-1] else final[:, 1:]
        reward_adv, reward_ret = _gae(tape.rewards, reward_values, bootstrap_r, tape.dones, cfg.gamma, cfg.gae_lambda)
        expanded = tape.costs[:, None, :].expand(tape.costs.shape[0], cfg.n_agents, cfg.n_constraints)
        cost_adv, cost_ret = _gae(expanded, cost_values, bootstrap_c, tape.dones, cfg.gamma, cfg.gae_lambda)
    return reward_adv, reward_ret, cost_adv, cost_ret


def _terminal_score(learner: MetaDriveStaticLearner, actor_params: Sequence[Mapping[str, torch.Tensor]], tape: MetaDriveRolloutTape) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = learner.config
    log_probs = []
    for index, actor in enumerate(learner.actors):
        dist = _actor_distribution(actor, actor_params[index], tape.obs[:, index, :])
        log_probs.append(dist.log_prob(tape.actions[:, index, :]).sum(-1))
    joint = torch.stack(log_probs, dim=-1).sum(-1)
    valid = (~tape.dones).to(learner.dtype)
    if bool(tape.dones[-1]):
        valid[-1] = 1.0
    rewards = tape.rewards.detach().mean(dim=-1) * valid
    returns = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=learner.dtype)
    for index in range(rewards.numel() - 1, -1, -1):
        running = rewards[index] + (1.0 - tape.dones[index].to(learner.dtype)) * running
        returns[index] = running
    return rewards.sum().detach(), (returns * joint * valid).sum()


def _behavior_joint_score(
    learner: MetaDriveStaticLearner,
    actor_params: Sequence[Mapping[str, torch.Tensor]],
    tape: MetaDriveRolloutTape,
) -> torch.Tensor:
    """Live joint log-probability score for one sampled learner rollout.

    The tape's observations/actions are detached exogenous samples.  The
    actor parameters are the current functional response state, so after the
    first unroll transition this score carries the rho-to-policy path needed
    by sampling correction.  At the initial state it correctly has zero rho
    derivative because the initial policy is independent of rho.
    """
    scores = []
    for index, actor in enumerate(learner.actors):
        dist = _actor_distribution(actor, actor_params[index], tape.obs[:, index, :])
        scores.append(dist.log_prob(tape.actions[:, index, :]).sum(-1))
    valid = (~tape.dones).to(learner.dtype)
    if bool(tape.dones[-1]):
        valid[-1] = 1.0
    return torch.stack(scores, dim=-1).sum(dim=-1).mul(valid).sum()


class MetaDriveDirectUnroll:
    """Run one fixed-tape direct-unroll lookahead from the current learner state."""

    estimator_name = "direct_unroll"
    sampling_correction = False

    def __init__(self, learner: MetaDriveStaticLearner, config: MetaDriveDUConfig | None = None):
        self.learner = learner
        self.config = config or MetaDriveDUConfig()
        if learner.config.n_agents < 2 or learner.config.n_constraints != 1:
            raise ValueError("MetaDrive DU requires N>=2 and K=1")
        if learner.estimator_name != self.estimator_name or learner.sampling_correction is not False:
            raise ValueError("learner estimator metadata is incompatible with MetaDrive DU")

    def run(
        self,
        update_tapes: Sequence[MetaDriveRolloutTape],
        evaluation_tape: MetaDriveRolloutTape,
        *,
        baseline: float = 0.0,
    ) -> MetaDriveDUResult:
        cfg = self.learner.config
        if len(update_tapes) != self.config.q:
            raise ValueError(f"expected exactly q={self.config.q} update tapes")
        for tape in (*update_tapes, evaluation_tape):
            tape.validate()
            if tape.obs.shape[1] != cfg.n_agents or tape.obs.shape[2] != cfg.obs_dim:
                raise ValueError("tape does not match learner dimensions")
        # The graph starts at a detached S0 copy.  Only rho logits remain a
        # differentiable meta-variable; all sampled tape fields are constants.
        actor_params = [{name: value.detach().clone().requires_grad_(True) for name, value in _parameters(actor).items()} for actor in self.learner.actors]
        critic_params = {name: value.detach().clone().requires_grad_(True) for name, value in _parameters(self.learner.critic).items()}
        actor_states = [_initial_adam_state(actor, self.learner.optimizer) for actor in self.learner.actors]
        critic_state = _initial_adam_state(self.learner.critic, self.learner.optimizer)
        rho_logits = self.learner.simplex.logits.detach().clone().requires_grad_(True)
        rho = simplex_from_logits(rho_logits, self.learner.simplex.floor)
        inner_scores: list[torch.Tensor] = []
        for tape in update_tapes:
            if self.config.sampling_correction:
                inner_scores.append(_behavior_joint_score(self.learner, actor_params, tape))
            reward_adv, reward_ret, cost_adv, cost_ret = _fixed_advantages(self.learner, actor_params, critic_params, tape)
            critic_values = functional_call(self.learner.critic, critic_params, (tape.obs.reshape(tape.obs.shape[0], -1),))
            critic_values = critic_values.reshape(tape.obs.shape[0], cfg.n_agents, 1 + cfg.n_constraints)
            critic_loss = torch.nn.functional.mse_loss(critic_values[..., 0], reward_ret)
            critic_loss = critic_loss + torch.nn.functional.mse_loss(critic_values[..., 1:], cost_ret)
            actor_total = torch.zeros((), dtype=self.learner.dtype)
            for agent, actor in enumerate(self.learner.actors):
                dist = _actor_distribution(actor, actor_params[agent], tape.obs[:, agent, :])
                new_lp = dist.log_prob(tape.actions[:, agent, :]).sum(-1)
                batch = PPOBatch(
                    tape.old_log_probs[:, agent], reward_adv[:, agent], cost_adv[:, agent, :].T,
                    torch.ones(tape.obs.shape[0], dtype=self.learner.dtype),
                    torch.ones(tape.obs.shape[0], dtype=self.learner.dtype), dist.entropy().sum(-1),
                )
                actor_total = actor_total + self.learner.ppo.evaluate(new_lp, batch, rho, self.learner.shared_lambda.values, agent).total / cfg.n_agents
            total = actor_total + cfg.critic_loss_coefficient * critic_loss
            all_values = [*actor_params[0].values()]
            for mapping in actor_params[1:]:
                all_values.extend(mapping.values())
            all_values.extend(critic_params.values())
            grads = torch.autograd.grad(total, all_values, create_graph=True, allow_unused=True)
            cursor = 0
            new_actor_params = []
            new_actor_states = []
            for mapping, state in zip(actor_params, actor_states):
                count = len(mapping)
                new_mapping, new_state = _adam_step(mapping, grads[cursor:cursor + count], state, lr=cfg.actor_lr, beta1=0.9, beta2=0.999, eps=1e-8)
                new_actor_params.append(new_mapping)
                new_actor_states.append(new_state)
                cursor += count
            critic_params, critic_state = _adam_step(critic_params, grads[cursor:], critic_state, lr=cfg.critic_lr, beta1=0.9, beta2=0.999, eps=1e-8)
            actor_params, actor_states = new_actor_params, new_actor_states
        terminal_welfare, objective = _terminal_score(self.learner, actor_params, evaluation_tape)
        direct = torch.autograd.grad(
            objective, rho_logits, retain_graph=bool(self.config.sampling_correction), allow_unused=True,
        )[0]
        direct = torch.zeros_like(rho_logits) if direct is None else direct
        score_gradient = torch.zeros_like(rho_logits)
        score_value = torch.zeros((), dtype=self.learner.dtype)
        if self.config.sampling_correction:
            score_value = torch.stack(inner_scores).sum()
            score = torch.autograd.grad(score_value, rho_logits, allow_unused=True)[0]
            score_gradient = torch.zeros_like(rho_logits) if score is None else score
            gradient = direct + (
                torch.as_tensor(terminal_welfare - float(baseline), dtype=self.learner.dtype) * score_gradient
            )
        else:
            gradient = direct
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("MetaDrive DU gradient is non-finite")
        result = MetaDriveDUResult(
            gradient.detach(), float(terminal_welfare), tuple(tape.digest() for tape in update_tapes), evaluation_tape.digest(), self.config.q,
            estimator=("direct_unroll_sampling_correction" if self.config.sampling_correction else "direct_unroll"),
            sampling_correction=self.config.sampling_correction,
            direct_gradient=direct.detach(), score_gradient=score_gradient.detach(),
            score_value=float(score_value.detach()), baseline=float(baseline),
        )
        return result


def run_independent_du_sc(
    learner: MetaDriveStaticLearner,
    update_tapes: Sequence[Sequence[MetaDriveRolloutTape]],
    evaluation_tapes: Sequence[MetaDriveRolloutTape],
    *,
    q: int,
    baseline: str = "loo",
) -> MetaDriveDUResult:
    """Average independent DU graphs with zero or cross-fitted LOO-SC.

    Each replicate is built from disjoint update/evaluation tapes but starts
    from the identical post-PPO learner state.  The direct term is averaged
    across replicates.  For LOO, replicate ``m`` uses the detached mean welfare
    from the other ``M-1`` terminal tapes as its baseline, so no replicate's
    own welfare enters its score baseline.
    """
    if baseline not in {"zero", "loo"}:
        raise ValueError("baseline must be 'zero' or 'loo'")
    if not update_tapes or len(update_tapes) != len(evaluation_tapes):
        raise ValueError("DU+SC requires matching nonempty replicate tape sets")
    if baseline == "loo" and len(update_tapes) < 2:
        raise ValueError("LOO-SC requires at least two independent replicates")
    transaction = MetaDriveDirectUnroll(
        learner, MetaDriveDUConfig(q=q, sampling_correction=True),
    )
    results = [
        transaction.run(updates, evaluation, baseline=0.0)
        for updates, evaluation in zip(update_tapes, evaluation_tapes, strict=True)
    ]
    welfare = torch.as_tensor([item.terminal_welfare for item in results], dtype=learner.dtype)
    direct_rows = torch.stack([item.direct_gradient for item in results])
    score_rows = torch.stack([item.score_gradient for item in results])
    if baseline == "loo":
        baselines = (welfare.sum() - welfare) / (len(results) - 1)
    else:
        baselines = torch.zeros_like(welfare)
    correction_rows = (welfare - baselines).reshape(-1, 1, 1) * score_rows
    direct_gradient = direct_rows.mean(dim=0)
    correction_gradient = correction_rows.mean(dim=0)
    gradient = direct_gradient + correction_gradient
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("MetaDrive DU+SC gradient is non-finite")
    return MetaDriveDUResult(
        gradient.detach(), float(welfare.mean()),
        tuple(digest for result in results for digest in result.tape_digests),
        hashlib.sha256("".join(item.evaluation_digest for item in results).encode()).hexdigest(), q,
        estimator="direct_unroll_sampling_correction", sampling_correction=True,
        direct_gradient=direct_gradient.detach(), score_gradient=correction_gradient.detach(),
        score_value=float(torch.stack([torch.as_tensor(item.score_value, dtype=learner.dtype) for item in results]).mean()),
        baseline=float(baselines.mean()),
    )


__all__ = [
    "MetaDriveDUConfig", "MetaDriveDUResult", "MetaDriveDirectUnroll", "MetaDriveRolloutTape",
    "capture_tapes", "run_independent_du_sc",
]
