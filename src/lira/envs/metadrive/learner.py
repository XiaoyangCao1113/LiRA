"""MetaDrive N-agent / K=1 PPO learner with a responsibility simplex and shared dual.

The actor loss reuses the canonical :class:`lira.ppo.PPOObjective`
(effective cost penalties ``N * lambda * rho``). Within one ``update()`` the
responsibility allocation (rho) and the shared dual (lambda) are frozen
inputs; the outer responsibility update and the live dual update are applied
by :class:`lira.envs.metadrive.online.MetaDriveOnlineLearner`. The checkpoint
metadata declares the outer-estimator identity (``direct_unroll``,
``sampling_correction=False``) so that checkpoints from incompatible
learners are rejected.

The native per-agent action space is ``Box(-1, 1, (2,))`` (steering,
throttle). Sampled Gaussian actions are stored raw for the PPO update but
clipped to the native actuator bounds before being sent to the environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ...ppo import PPOBatch, PPOConfig, PPOObjective
from ...responsibility import SharedLambda, StaticSimplex


@dataclass(frozen=True)
class MetaDriveLearnerConfig:
    n_agents: int = 4
    n_constraints: int = 1
    obs_dim: int = 91
    action_dim: int = 2
    horizon: int = 1000
    hidden: int = 32
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.0
    critic_loss_coefficient: float = 0.5
    rho_floor: float = 0.0
    lambda_init: float = 1.0
    response_method: str = "disabled"

    def __post_init__(self) -> None:
        if self.n_agents < 2 or self.n_constraints != 1:
            raise ValueError("MetaDrive learner requires N>=2 and K=1")
        if self.response_method != "disabled":
            raise ValueError("the inner PPO learner keeps rho fixed; response_method must be 'disabled'")
        if self.horizon < 1 or self.obs_dim < 1 or self.action_dim < 1:
            raise ValueError("invalid horizon/obs_dim/action_dim")
        if self.actor_lr <= 0 or self.critic_lr <= 0 or self.lambda_init < 0:
            raise ValueError("invalid learning rate or lambda init")


class _GaussianActor(nn.Module):
    """One small MLP mean head plus a state-independent log-std, per agent."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=dtype))
        self.to(dtype)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(obs)
        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std)


class MetaDriveStaticLearner:
    """Actor/critic/optimizer/rho/lambda/RNG/counters, all checkpointable together."""

    estimator_name = "direct_unroll"
    sampling_correction = False

    def __init__(self, env: Any, config: MetaDriveLearnerConfig | None = None, dtype: torch.dtype = torch.float64):
        self.env = env
        self.config = config or MetaDriveLearnerConfig()
        self.dtype = dtype
        cfg = self.config
        self.actors = nn.ModuleList(
            [_GaussianActor(cfg.obs_dim, cfg.action_dim, cfg.hidden, dtype) for _ in range(cfg.n_agents)]
        )
        self.critic = nn.Sequential(
            nn.Linear(cfg.n_agents * cfg.obs_dim, cfg.hidden),
            nn.Tanh(),
            nn.Linear(cfg.hidden, cfg.n_agents * (1 + cfg.n_constraints)),
        ).to(dtype)
        self.optimizer = torch.optim.Adam(
            list(self.actors.parameters()) + list(self.critic.parameters()), lr=cfg.actor_lr
        )
        self.simplex = StaticSimplex.uniform(cfg.n_constraints, cfg.n_agents, floor=cfg.rho_floor)
        self.simplex.logits = self.simplex.logits.to(dtype=dtype)
        self.shared_lambda = SharedLambda(torch.full((cfg.n_constraints,), cfg.lambda_init, dtype=dtype))
        self.ppo = PPOObjective(PPOConfig(cfg.clip_ratio, cfg.entropy_coefficient, 0.0))
        self.update_count = 0
        self.rollout_counter = 0

    def _critic_values(self, obs_TND: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t, n, d = obs_TND.shape
        out = self.critic(obs_TND.reshape(t, n * d)).reshape(t, n, 1 + self.config.n_constraints)
        return out[..., 0], out[..., 1:]

    @staticmethod
    def _gae(rewards, values, bootstrap, dones, gamma, lam):
        t = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        next_value = bootstrap
        next_adv = torch.zeros_like(bootstrap)
        for i in range(t - 1, -1, -1):
            mask = 0.0 if bool(dones[i]) else 1.0
            delta = rewards[i] + gamma * next_value * mask - values[i]
            next_adv = delta + gamma * lam * mask * next_adv
            advantages[i] = next_adv
            next_value = values[i]
        return advantages, advantages + values

    def rollout(self, seed: int) -> dict[str, Any]:
        cfg = self.config
        obs0 = self.env.reset(seed)
        obs = [torch.as_tensor(obs0, dtype=self.dtype)]
        raw_actions, oldlp, rewards, costs, native_costs = [], [], [], [], []
        done = False
        accounting_last: dict | None = None
        for _ in range(cfg.horizon):
            o = obs[-1]
            step_raw = torch.empty((cfg.n_agents, cfg.action_dim), dtype=self.dtype)
            step_logp = torch.empty((cfg.n_agents,), dtype=self.dtype)
            for i in range(cfg.n_agents):
                dist = self.actors[i].distribution(o[i])
                a = dist.sample()
                step_raw[i] = a
                step_logp[i] = dist.log_prob(a).sum()
            env_actions = torch.clamp(step_raw, -1.0, 1.0).detach().to(torch.float32).numpy()
            next_obs, native_reward, done, accounting = self.env.step(env_actions)
            obs.append(torch.as_tensor(next_obs, dtype=self.dtype))
            raw_actions.append(step_raw)
            # PPO's old policy is a frozen behavior-policy snapshot.  Keeping
            # this tensor attached to the actor graph makes the new/old
            # log-ratio differentiate as log_pi - log_pi == 0, silently
            # cancelling the entire actor gradient on every update.  The
            # resulting trainer still updates its critic, which can look
            # healthy while the policy remains at initialization.  Detach at
            # collection time so the ratio differentiates only through the
            # current-policy log probability.
            oldlp.append(step_logp.detach())
            rewards.append(torch.as_tensor(native_reward, dtype=self.dtype))
            costs.append(torch.as_tensor([accounting["shared_cost"]], dtype=self.dtype))
            native_costs.append(torch.as_tensor(accounting["native_cost"], dtype=self.dtype))
            accounting_last = accounting
            if done:
                break
        self.rollout_counter += 1
        T = len(raw_actions)
        return {
            "obs": torch.stack(obs[:-1]),
            "final_obs": obs[-1],
            "actions": torch.stack(raw_actions),
            "old_log_probs": torch.stack(oldlp),
            "rewards": torch.stack(rewards),
            "costs": torch.stack(costs),
            # Per-agent native cost, shape (T, n_agents) -- additive alongside
            # ``costs`` (the team/shared broadcast cost). Existing callers that
            # only read ``costs`` are unaffected.
            "native_costs": torch.stack(native_costs),
            "done": done,
            "steps": T,
            "accounting": accounting_last,
        }

    def update(self, seed: int) -> dict[str, Any]:
        """One rollout plus one PPO gradient step; rho/lambda stay frozen (no outer update)."""
        cfg = self.config
        batch_data = self.rollout(seed)
        O, A, LP, R, C = (
            batch_data["obs"], batch_data["actions"], batch_data["old_log_probs"],
            batch_data["rewards"], batch_data["costs"],
        )
        T, done = batch_data["steps"], batch_data["done"]

        with torch.no_grad():
            v_reward_t, v_cost_t = self._critic_values(O)
            final_reward_b, final_cost_b = self._critic_values(batch_data["final_obs"].unsqueeze(0))
            bootstrap_reward = torch.zeros(cfg.n_agents, dtype=self.dtype) if done else final_reward_b[0]
            bootstrap_cost = torch.zeros(cfg.n_agents, cfg.n_constraints, dtype=self.dtype) if done else final_cost_b[0]
            dones = torch.zeros(T, dtype=torch.bool)
            if done:
                dones[-1] = True
            adv_reward, ret_reward = self._gae(R, v_reward_t, bootstrap_reward, dones, cfg.gamma, cfg.gae_lambda)
            # The team shared cost is broadcast identically to every agent's
            # cost-value target; there is no per-agent cost decomposition.
            cost_expanded = C.unsqueeze(1).expand(T, cfg.n_agents, cfg.n_constraints)
            adv_cost, ret_cost = self._gae(cost_expanded, v_cost_t, bootstrap_cost, dones, cfg.gamma, cfg.gae_lambda)

        self.optimizer.zero_grad()
        v_reward, v_cost = self._critic_values(O)
        rho, lam = self.simplex.rho, self.shared_lambda.values
        actor_total = torch.zeros((), dtype=self.dtype)
        for i in range(cfg.n_agents):
            dist = self.actors[i].distribution(O[:, i, :])
            nlp = dist.log_prob(A[:, i, :]).sum(-1)
            ent = dist.entropy().sum(-1)
            ones = torch.ones(T, dtype=self.dtype)
            batch = PPOBatch(LP[:, i], adv_reward[:, i], adv_cost[:, i, :].T, ones, ones, ent)
            actor_total = actor_total + self.ppo.evaluate(nlp, batch, rho, lam, i).total / cfg.n_agents
        critic_loss = torch.nn.functional.mse_loss(v_reward, ret_reward) + torch.nn.functional.mse_loss(v_cost, ret_cost)
        total = actor_total + cfg.critic_loss_coefficient * critic_loss
        total.backward()
        self.optimizer.step()
        self.update_count += 1
        return {
            "steps": T,
            "loss": float(total.detach()),
            "actor_loss": float(actor_total.detach()),
            "critic_loss": float(critic_loss.detach()),
            "rho": rho.tolist(),
            "shared_lambda": lam.tolist(),
            "effective_penalty_n_lambda_rho": self.simplex.effective_penalties(self.shared_lambda).tolist(),
            "raw_shared_cost": float(C.sum()),
            "raw_reward_per_agent": R.sum(0).tolist(),
            "accounting": batch_data["accounting"],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "estimator_name": self.estimator_name,
            "sampling_correction": self.sampling_correction,
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rho": self.simplex.state_dict(),
            "shared_lambda": self.shared_lambda.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "update_count": self.update_count,
            "rollout_counter": self.rollout_counter,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("estimator_name") != self.estimator_name or state.get("sampling_correction") is not False:
            raise ValueError("incompatible MetaDrive learner checkpoint")
        for actor, actor_state in zip(self.actors, state["actors"]):
            actor.load_state_dict(actor_state)
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.simplex.load_state_dict(state["rho"], target_dtype=self.dtype)
        self.shared_lambda.load_state_dict(state["shared_lambda"], target_dtype=self.dtype)
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        self.update_count = int(state["update_count"])
        self.rollout_counter = int(state["rollout_counter"])

    def checkpoint(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    def load_checkpoint(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(Path(path), weights_only=False))
