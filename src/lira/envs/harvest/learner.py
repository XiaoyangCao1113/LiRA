"""CPU masked-categorical PPO learner with a shared cost constraint.

Used for the seven-player Commons Harvest task.  One shared K=1 constraint,
categorical actions with a pre-step mask, and a reset-boundary update
("transaction"): each call collects one episode, then takes one critic step,
one PPO actor step per agent, and one projected shared-dual step.

Two arm semantics are implemented:

* ``uniform`` (also used by LiRA, whose allocation ``rho`` is updated
  externally by the lookahead in :mod:`lira.envs.harvest.meta_batch`): every
  agent's cost coefficient is ``N * lambda * rho_i`` with one shared dual.
* ``pal`` (per-agent Lagrangian): one dual per agent, each driven by the same
  team cost signal, and no responsibility simplex.

Episode cost ("damage") is accumulated in native units.  PPO cost advantages
are computed from the per-step shared cost with the configured discount; the
episode-sum accounting used by the dual update is kept separate and is never
replaced by a discounted sum.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from lira.responsibility import SharedLambda, StaticSimplex

from .env import N_AGENTS
from .interfaces import AdapterStep, StepObservation


N_CONSTRAINTS = 1
ARM_UNIFORM = "uniform"
ARM_PAL = "pal"
ARMS = (ARM_UNIFORM, ARM_PAL)


def _seeded_module(factory: Callable[[], nn.Module], seed: int) -> nn.Module:
    """Construct a deterministically initialized module without perturbing global RNG."""
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        return factory()
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


class MaskedCategoricalPolicy(nn.Module):
    """One categorical action per agent, with invalid categories at -inf."""

    def __init__(self, obs_dim: int, categories: int, hidden_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        if obs_dim < 1 or categories < 2 or hidden_dim < 1:
            raise ValueError("invalid categorical policy dimensions")
        self.categories = int(categories)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, categories)
        ).to(dtype=dtype)

    def masked_logits(self, observations: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or masks.shape != (observations.shape[0], self.categories):
            raise ValueError("categorical observations/masks have incompatible shapes")
        if masks.dtype != torch.bool or not torch.all(masks.any(dim=-1)):
            raise ValueError("each categorical mask must be bool with at least one legal action")
        logits = self.net(observations)
        return logits.masked_fill(~masks, float("-inf"))

    def log_prob_entropy(
        self, observations: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim != 1 or actions.dtype != torch.int64:
            raise ValueError("categorical actions must be a one-dimensional int64 tensor")
        logits = self.masked_logits(observations, masks)
        if torch.any(actions < 0) or torch.any(actions >= self.categories):
            raise ValueError("categorical action is out of range")
        if not torch.all(masks.gather(1, actions[:, None]).squeeze(1)):
            raise ValueError("categorical action selects a masked category")
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy()

    def sample_with_uniforms(
        self, observations: torch.Tensor, masks: torch.Tensor, uniforms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if uniforms.shape != (observations.shape[0],) or not torch.isfinite(uniforms).all() or torch.any((uniforms < 0) | (uniforms >= 1)):
            raise ValueError("categorical uniforms must be finite and lie in [0,1)")
        logits = self.masked_logits(observations, masks)
        probs = torch.softmax(logits, dim=-1)
        actions = (uniforms[:, None] >= probs.cumsum(dim=-1)).sum(dim=-1).clamp(max=self.categories - 1)
        actions = actions.to(torch.int64)
        logp, _ = self.log_prob_entropy(observations, actions, masks)
        return actions, logp


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        ).to(dtype=dtype)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


@dataclass(frozen=True)
class CategoricalLearnerConfig:
    obs_dim: int = 17
    categories: int = 8
    hidden_dim: int = 32
    horizon: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    lambda_lr: float = 1e-3
    rho_lr: float = 1e-3
    damage_limit: float = 50.0
    lambda_init: float = 0.1
    rho_floor: float = 0.0
    seed: int = 1001
    device: str = "cpu"
    dtype: str = "float32"
    # Number of inner PPO updates per lookahead replicate (recorded for
    # provenance; the training script passes ``q`` explicitly).
    meta_du_sc_q: int = 2

    def __post_init__(self) -> None:
        if self.obs_dim < 1 or self.categories < 2 or self.hidden_dim < 1 or self.horizon < 1:
            raise ValueError("learner dimensions must be positive")
        if self.meta_du_sc_q < 1:
            raise ValueError("meta_du_sc_q must be a positive integer")
        if self.gamma <= 0 or self.gamma > 1 or self.gae_lambda <= 0 or self.gae_lambda > 1:
            raise ValueError("discount and GAE factors must lie in (0,1]")
        if self.actor_lr <= 0 or self.critic_lr <= 0 or self.lambda_lr <= 0 or self.rho_lr <= 0:
            raise ValueError("learner rates must be positive")
        if not np.isfinite(self.damage_limit) or self.damage_limit < 0:
            raise ValueError("damage limit must be finite and nonnegative")
        if not np.isfinite(self.lambda_init) or self.lambda_init < 0:
            raise ValueError("lambda_init must be finite and nonnegative")
        if self.device != "cpu" or self.dtype != "float32":
            raise ValueError("the categorical learner is CPU float32 only")

    def canonical_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update({"n_agents": N_AGENTS, "n_constraints": N_CONSTRAINTS})
        return data


@dataclass(frozen=True)
class CategoricalRollout:
    observations: torch.Tensor  # T x N x D
    actions: torch.Tensor  # T x N
    action_masks: torch.Tensor  # T x N x C
    old_log_probs: torch.Tensor  # T x N
    rewards: torch.Tensor  # T x N
    shared_damage: torch.Tensor  # T, native per-step team damage
    per_agent_damage: torch.Tensor  # T x N, diagnostic only
    dones: torch.Tensor  # T
    executed_rho: torch.Tensor  # T x N
    episode_damage: float
    episode_agent_damage: torch.Tensor  # N
    death_events: torch.Tensor  # T x N
    timeout: bool
    terminated: bool

    def validate(self, cfg: CategoricalLearnerConfig) -> None:
        expected = {
            "observations": (cfg.horizon, N_AGENTS, cfg.obs_dim),
            "actions": (cfg.horizon, N_AGENTS),
            "action_masks": (cfg.horizon, N_AGENTS, cfg.categories),
            "old_log_probs": (cfg.horizon, N_AGENTS),
            "rewards": (cfg.horizon, N_AGENTS),
            "shared_damage": (cfg.horizon,),
            "per_agent_damage": (cfg.horizon, N_AGENTS),
            "dones": (cfg.horizon,),
            "executed_rho": (cfg.horizon, N_AGENTS),
            "episode_agent_damage": (N_AGENTS,),
            "death_events": (cfg.horizon, N_AGENTS),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} has shape {tuple(getattr(self, name).shape)}, expected {shape}")
        if self.actions.dtype != torch.int64 or self.action_masks.dtype != torch.bool:
            raise ValueError("categorical rollout uses int64 actions and bool masks")
        for name in expected:
            value = getattr(self, name)
            if not torch.isfinite(value.to(torch.float32)).all():
                raise FloatingPointError(f"non-finite rollout field {name}")
        if not torch.all(self.action_masks.any(dim=-1)):
            raise ValueError("rollout contains an all-invalid action mask")
        if not torch.all(self.action_masks.gather(-1, self.actions[..., None]).squeeze(-1)):
            raise ValueError("rollout action is masked")
        if abs(float(self.episode_damage) - float(self.shared_damage.sum())) > 1e-4:
            raise ValueError("episode-sum damage does not equal raw step damage")

    def discounted_cost_returns(self, cfg: CategoricalLearnerConfig) -> torch.Tensor:
        """PPO cost targets; unlike ``episode_damage``, these are discounted."""
        returns = torch.zeros_like(self.shared_damage)
        running = torch.zeros((), dtype=self.shared_damage.dtype, device=self.shared_damage.device)
        for t in range(cfg.horizon - 1, -1, -1):
            running = self.shared_damage[t] + cfg.gamma * (1.0 - self.dones[t]) * running
            returns[t] = running
        return returns


class CategoricalLearner:
    """N-agent learner with ``uniform`` (shared dual, rho simplex) and ``pal`` arms."""

    def __init__(
        self,
        env: Any,
        config: CategoricalLearnerConfig | None = None,
        *,
        arm: str = ARM_UNIFORM,
    ) -> None:
        self.env = env
        self.config = config or CategoricalLearnerConfig()
        if getattr(env, "n_agents", None) != N_AGENTS or getattr(env, "n_constraints", None) != N_CONSTRAINTS:
            raise ValueError(f"learner requires an N={N_AGENTS}/K=1 environment")
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        if self.config.categories < 2:
            raise ValueError("categorical action space must have at least two categories")
        self.arm = arm
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.environment_digest = self._environment_digest()
        self._torch_generator = torch.Generator(device="cpu")
        self._torch_generator.manual_seed(self.config.seed + 10_000)
        self._reset_boundary = True
        self.rollout_counter = 0
        self.transaction_counter = 0
        self.actor = nn.ModuleList(
            [_seeded_module(lambda: MaskedCategoricalPolicy(self.config.obs_dim, self.config.categories, self.config.hidden_dim, self.dtype), self.config.seed + i) for i in range(N_AGENTS)]
        )
        self.reward_critic = nn.ModuleList(
            [_seeded_module(lambda: ValueCritic(self.config.obs_dim, self.config.hidden_dim, self.dtype), self.config.seed + 100 + i) for i in range(N_AGENTS)]
        )
        self.cost_critic = nn.ModuleList(
            [_seeded_module(lambda: ValueCritic(self.config.obs_dim, self.config.hidden_dim, self.dtype), self.config.seed + 200 + i) for i in range(N_AGENTS)]
        )
        self.actor_optimizers = [torch.optim.Adam(policy.parameters(), lr=self.config.actor_lr) for policy in self.actor]
        self.reward_optimizers = [torch.optim.Adam(critic.parameters(), lr=self.config.critic_lr) for critic in self.reward_critic]
        self.cost_optimizers = [torch.optim.Adam(critic.parameters(), lr=self.config.critic_lr) for critic in self.cost_critic]
        self.shared_lambda = SharedLambda(torch.tensor([self.config.lambda_init], dtype=self.dtype))
        self.rho = StaticSimplex.uniform(1, N_AGENTS, floor=self.config.rho_floor)
        self.rho.logits = self.rho.logits.to(dtype=self.dtype)
        self.duplicated_lambdas = torch.full((N_AGENTS,), self.config.lambda_init, dtype=self.dtype)
        self.last_rollout: CategoricalRollout | None = None

    def _environment_digest(self) -> str:
        provenance = getattr(self.env, "provenance", None)
        payload = asdict(provenance) if provenance is not None else {}
        payload.update({"n_agents": N_AGENTS, "n_constraints": 1})
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list, separators=(",", ":")).encode()).hexdigest()

    @property
    def rho_value(self) -> torch.Tensor:
        return self.rho.rho[0]

    @property
    def lambda_value(self) -> torch.Tensor:
        return self.shared_lambda.values[0]

    def effective_penalties(self) -> torch.Tensor:
        """Return the actor coefficients and keep arm semantics explicit."""
        if self.arm == ARM_PAL:
            return self.duplicated_lambdas.clone()
        return N_AGENTS * self.lambda_value * self.rho_value

    def _observation(self, value: StepObservation) -> tuple[torch.Tensor, torch.Tensor]:
        observations = torch.as_tensor(value.observation_matrix, dtype=self.dtype)
        raw_masks = value.action_mask_matrix
        if raw_masks.ndim != 2 or raw_masks.shape[0] != N_AGENTS or raw_masks.shape[1] > self.config.categories:
            raise ValueError("action mask shape exceeds the categorical contract")
        masks = torch.zeros((N_AGENTS, self.config.categories), dtype=torch.bool)
        masks[:, : raw_masks.shape[1]] = torch.as_tensor(raw_masks, dtype=torch.bool)
        if observations.shape != (N_AGENTS, self.config.obs_dim) or not torch.isfinite(observations).all():
            raise ValueError("observation shape/dtype is outside the categorical contract")
        if not torch.all(masks.any(dim=-1)):
            raise ValueError("environment emitted an all-invalid action mask")
        return observations, masks

    def _action_uniforms(self) -> torch.Tensor:
        return torch.rand((N_AGENTS,), generator=self._torch_generator, dtype=self.dtype)

    def collect(self, *, seed: int | None = None) -> CategoricalRollout:
        if not self._reset_boundary:
            raise RuntimeError("collector is already inside a transaction")
        reset_seed = self.config.seed + self.rollout_counter if seed is None else int(seed)
        # The adapter must seed its backend, but a learner transaction cannot
        # consume the caller's process-wide Python/NumPy/Torch streams.  Save
        # and restore those streams around the complete environment episode;
        # policy exploration uses the learner-local Torch generator below.
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        current = self.env.reset(seed=reset_seed)
        self.rollout_counter += 1
        self._reset_boundary = False
        obs_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        logprob_rows: list[torch.Tensor] = []
        reward_rows: list[torch.Tensor] = []
        shared_damage_rows: list[torch.Tensor] = []
        agent_damage_rows: list[torch.Tensor] = []
        done_rows: list[torch.Tensor] = []
        rho_rows: list[torch.Tensor] = []
        death_rows: list[torch.Tensor] = []
        episode_damage = 0.0
        agent_damage = torch.zeros(N_AGENTS, dtype=self.dtype)
        terminated = False
        timeout = False
        try:
            for _ in range(self.config.horizon):
                observations, masks = self._observation(current)
                obs_rows.append(observations)
                mask_rows.append(masks)
                actions_list: list[torch.Tensor] = []
                logprobs_list: list[torch.Tensor] = []
                uniforms = self._action_uniforms()
                for agent in range(N_AGENTS):
                    action, logprob = self.actor[agent].sample_with_uniforms(
                        observations[agent : agent + 1], masks[agent : agent + 1], uniforms[agent : agent + 1]
                    )
                    actions_list.append(action[0])
                    # Behavior log-probabilities are frozen PPO data, never a
                    # live graph into the pre-update actor parameters.
                    logprobs_list.append(logprob[0].detach())
                actions = torch.stack(actions_list).to(torch.int64)
                result = self.env.step(actions.tolist())
                if not isinstance(result, AdapterStep):
                    raise TypeError("adapter must return AdapterStep")
                action_rows.append(actions)
                logprob_rows.append(torch.stack(logprobs_list))
                rewards = torch.as_tensor(result.rewards, dtype=self.dtype)
                if rewards.shape != (N_AGENTS,) or not torch.isfinite(rewards).all():
                    raise ValueError("categorical adapter emitted an invalid per-agent reward vector")
                reward_rows.append(rewards)
                transition = result.transition
                per_agent = torch.as_tensor(transition.per_agent_damage, dtype=self.dtype)
                team_damage = torch.tensor(float(transition.team_damage), dtype=self.dtype)
                shared_damage_rows.append(team_damage)
                agent_damage_rows.append(per_agent)
                done_rows.append(torch.tensor(float(result.terminated), dtype=self.dtype))
                rho_rows.append(self.rho_value.detach().clone())
                death_rows.append(torch.as_tensor(transition.death_events, dtype=torch.bool))
                episode_damage += float(transition.team_damage)
                agent_damage += per_agent
                current = result.observation
                terminated = bool(result.terminated)
                timeout = bool(transition.timeout)
                if terminated:
                    break
            if not terminated:
                # The adapter should report timeout at max_steps; a nonterminal
                # horizon remains a valid truncation but is not a death.
                timeout = True
        finally:
            self._reset_boundary = True
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
        steps = len(action_rows)
        def pad(rows: list[torch.Tensor], shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
            if len(rows) == self.config.horizon:
                return torch.stack(rows)
            zero = torch.zeros(shape, dtype=dtype)
            return torch.stack(rows + [zero] * (self.config.horizon - len(rows)))
        rollout = CategoricalRollout(
            observations=pad(obs_rows, (N_AGENTS, self.config.obs_dim), dtype=self.dtype),
            actions=pad(action_rows, (N_AGENTS,), dtype=torch.int64),
            action_masks=pad(mask_rows, (N_AGENTS, self.config.categories), dtype=torch.bool),
            old_log_probs=pad(logprob_rows, (N_AGENTS,), dtype=self.dtype),
            rewards=pad(reward_rows, (N_AGENTS,), dtype=self.dtype),
            shared_damage=pad(shared_damage_rows, (), dtype=self.dtype),
            per_agent_damage=pad(agent_damage_rows, (N_AGENTS,), dtype=self.dtype),
            dones=pad(done_rows, (), dtype=self.dtype),
            executed_rho=pad(rho_rows, (N_AGENTS,), dtype=self.dtype),
            episode_damage=float(episode_damage), episode_agent_damage=agent_damage,
            death_events=pad(death_rows, (N_AGENTS,), dtype=torch.bool), timeout=timeout, terminated=terminated,
        )
        # Padding is never legal data; give it an explicit no-op mask and zero
        # weights by marking the rows done.  The transaction slices active rows.
        if steps < self.config.horizon:
            rollout.dones[steps:] = 1.0
            rollout.action_masks[steps:] = rollout.action_masks[max(steps - 1, 0)]
            rollout.actions[steps:] = 0
            rollout.executed_rho[steps:] = rollout.executed_rho[max(steps - 1, 0)]
        rollout.validate(self.config)
        self.last_rollout = rollout
        return rollout

    def _advantages(self, rollout: CategoricalRollout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.config.horizon
        reward_adv = torch.zeros((t, N_AGENTS), dtype=self.dtype)
        reward_returns = torch.zeros_like(reward_adv)
        cost_adv = torch.zeros_like(reward_adv)
        cost_returns = rollout.discounted_cost_returns(self.config)
        with torch.no_grad():
            values_r = torch.stack([self.reward_critic[i](rollout.observations[:, i]) for i in range(N_AGENTS)], dim=1)
            values_c = torch.stack([self.cost_critic[i](rollout.observations[:, i]) for i in range(N_AGENTS)], dim=1)
        next_r = torch.zeros(N_AGENTS, dtype=self.dtype)
        next_c = torch.zeros(N_AGENTS, dtype=self.dtype)
        gae_r = torch.zeros(N_AGENTS, dtype=self.dtype)
        gae_c = torch.zeros(N_AGENTS, dtype=self.dtype)
        for step in range(t - 1, -1, -1):
            not_done = 1.0 - rollout.dones[step]
            delta_r = rollout.rewards[step] + self.config.gamma * not_done * next_r - values_r[step]
            delta_c = rollout.shared_damage[step] + self.config.gamma * not_done * next_c - values_c[step]
            gae_r = delta_r + self.config.gamma * self.config.gae_lambda * not_done * gae_r
            gae_c = delta_c + self.config.gamma * self.config.gae_lambda * not_done * gae_c
            reward_adv[step], cost_adv[step] = gae_r, gae_c
            reward_returns[step], _ = gae_r + values_r[step], gae_c + values_c[step]
            next_r, next_c = values_r[step], values_c[step]
        return reward_adv, cost_adv, reward_returns, cost_returns

    def _actor_loss(self, agent: int, rollout: CategoricalRollout, reward_adv: torch.Tensor, cost_adv: torch.Tensor) -> torch.Tensor:
        active = (rollout.dones == 0).to(self.dtype)
        logp, entropy = self.actor[agent].log_prob_entropy(
            rollout.observations[:, agent], rollout.actions[:, agent], rollout.action_masks[:, agent]
        )
        ratio = torch.exp(logp - rollout.old_log_probs[:, agent])
        if self.arm == ARM_PAL:
            coefficient = self.duplicated_lambdas[agent]
        else:
            coefficient = N_AGENTS * self.lambda_value * self.rho_value[agent]
        advantage = reward_adv[:, agent] - coefficient * cost_adv[:, agent]
        clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
        surrogate = torch.minimum(ratio * advantage.detach(), clipped * advantage.detach())
        denom = active.sum().clamp_min(1.0)
        return -(surrogate * active).sum() / denom - self.config.entropy_coefficient * (entropy * active).sum() / denom

    def _update_duals(self, episode_damage: float) -> tuple[float, ...]:
        if self.arm == ARM_PAL:
            before = self.duplicated_lambdas.clone()
            # This scientifically identifiable control has one duplicated
            # lambda coordinate per agent and no responsibility simplex.  It
            # observes the same K=1 team damage signal in each coordinate, so
            # equality is expected and is not a heterogeneous-dual claim.
            self.duplicated_lambdas = torch.clamp(
                self.duplicated_lambdas + self.config.lambda_lr * (float(episode_damage) - self.config.damage_limit), min=0.0
            )
            return tuple(float(v) for v in before)
        before = self.shared_lambda.values.clone()
        self.shared_lambda.projected_update_(torch.tensor([float(episode_damage) - self.config.damage_limit], dtype=self.dtype), self.config.lambda_lr)
        return tuple(float(v) for v in before)

    def transaction(self, *, seed: int | None = None) -> "TransactionResult":
        """Collect one episode, then update critics, actors, and duals once."""
        rollout = self.collect(seed=seed)
        reward_adv, cost_adv, reward_returns, cost_returns = self._advantages(rollout)
        active = (rollout.dones == 0).to(self.dtype)
        for agent in range(N_AGENTS):
            obs = rollout.observations[:, agent]
            active_count = active.sum().clamp_min(1.0)
            self.reward_optimizers[agent].zero_grad(set_to_none=True)
            reward_loss = (((self.reward_critic[agent](obs) - reward_returns[:, agent]) ** 2) * active).sum() / active_count
            reward_loss.backward()
            self.reward_optimizers[agent].step()
            self.cost_optimizers[agent].zero_grad(set_to_none=True)
            cost_loss = (((self.cost_critic[agent](obs) - cost_returns) ** 2) * active).sum() / active_count
            cost_loss.backward()
            self.cost_optimizers[agent].step()
            self.actor_optimizers[agent].zero_grad(set_to_none=True)
            loss = self._actor_loss(agent, rollout, reward_adv, cost_adv)
            loss.backward()
            self.actor_optimizers[agent].step()
        self._update_duals(rollout.episode_damage)
        self.transaction_counter += 1
        return TransactionResult(rollout=rollout)

    def close(self) -> None:
        """Close the environment adapter."""
        backend = getattr(self.env, "env", self.env)
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "CategoricalLearner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


@dataclass(frozen=True)
class TransactionResult:
    rollout: CategoricalRollout


__all__ = [
    "ARMS", "ARM_PAL", "ARM_UNIFORM", "N_AGENTS", "N_CONSTRAINTS",
    "CategoricalLearner", "CategoricalLearnerConfig", "CategoricalRollout",
    "MaskedCategoricalPolicy", "TransactionResult", "ValueCritic",
]
