"""Online, finite-horizon MetaDrive learner: Uniform / LiRA / PAL arms.

All three arms share the identical actor/critic/PPO machinery owned by
``MetaDriveStaticLearner``; this module only adds the live outer-loop
mechanics on top of it:

* ``arm="uniform"``: one ``MetaDriveStaticLearner.update()`` (rollout + one
  PPO/critic gradient step, rho/lambda frozen for that inner step) per
  ``.step()``, plus a live dual-ascent update of the shared lambda using the
  realized team cost. Responsibility (rho) stays uniform.
* ``arm="lira"``: identical to ``uniform``, plus a direct-unroll (DU) outer
  update of the responsibility simplex every ``rho_interval`` steps using
  ``MetaDriveDirectUnroll`` (optionally with LOO sampling correction over
  ``sc_replicates`` independent lookaheads). The DU gradient is a
  welfare-*ascent* gradient; it is applied through the descent primitive
  ``StaticSimplex.apply_tangent_gradient_`` with one sign flip.
* ``arm="pal"`` (per-agent Lagrangian): no rho, no shared lambda. N
  independent lambdas, each driven only by that agent's own realized native
  cost against an equal share ``cost_budget / N`` of the budget. The
  canonical PPO surrogate (``PPOObjective.evaluate``) is reused by passing a
  frozen-uniform ``rho`` row together with the agent's own scalar lambda, so
  that ``n_agents * shared_lambda * rho[:, agent_index]`` reduces to exactly
  that agent's own lambda.

``run_heldout_episode``/``heldout_evaluate`` are a read-only evaluation
utility (no gradient, no optimizer/dual/rho mutation): a deterministic
mean-action rollout of the current actors reporting native episode
return/welfare, arrival/success, and full native cost, for held-out seeds
distinct from the training seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...ppo import PPOBatch
from .direct_unroll import MetaDriveDUConfig, MetaDriveDirectUnroll, capture_tapes, run_independent_du_sc
from .learner import MetaDriveLearnerConfig, MetaDriveStaticLearner

ARMS = ("uniform", "lira", "pal")

@dataclass(frozen=True)
class MetaDriveOnlineConfig:
    arm: str
    cost_budget: float
    lambda_lr: float
    rho_lr: float = 0.0
    rho_interval: int = 5
    du_q: int = 2
    lambda_init: float = 1.0
    sampling_correction: bool = False
    sc_baseline: str = "zero"
    sc_replicates: int = 1
    # MetaDrive uses the seed as a bounded scenario index, so the independent
    # LOO replicate stride must keep every lookahead seed inside the
    # scenario bank. With 512 scenarios, a stride of 17 keeps the q=2 tapes of
    # different replicates disjoint.
    sc_seed_stride: int = 17

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if not (self.cost_budget > 0):
            raise ValueError("cost_budget must be positive")
        if not (self.lambda_lr > 0):
            raise ValueError("lambda_lr must be positive")
        if self.rho_lr < 0:
            raise ValueError("rho_lr must be nonnegative")
        if self.rho_interval < 1:
            raise ValueError("rho_interval must be a positive integer")
        if self.du_q < 1:
            raise ValueError("du_q must be a positive integer")
        if self.lambda_init < 0:
            raise ValueError("lambda_init must be nonnegative")
        if self.sc_baseline not in {"zero", "loo"}:
            raise ValueError("sc_baseline must be 'zero' or 'loo'")
        if self.sc_replicates < 1 or self.sc_seed_stride < 1:
            raise ValueError("sc_replicates and sc_seed_stride must be positive")
        if self.sampling_correction and self.sc_baseline == "loo" and self.sc_replicates < 2:
            raise ValueError("LOO sampling correction requires at least two replicates")


class MetaDriveOnlineLearner:
    """Wraps one ``MetaDriveStaticLearner`` with a live outer-loop arm."""

    def __init__(
        self,
        env: Any,
        learner_config: MetaDriveLearnerConfig,
        online_config: MetaDriveOnlineConfig,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.env = env
        self.learner_config = learner_config
        self.config = online_config
        self.learner = MetaDriveStaticLearner(env, learner_config, dtype=dtype)
        self.step_count = 0
        self.pal_lambdas = torch.full(
            (learner_config.n_agents,), online_config.lambda_init, dtype=self.learner.dtype
        )

    def _derive_du_seeds(self, seed: int) -> tuple[tuple[int, ...], int]:
        """Deterministic fresh seeds for the q update tapes + 1 evaluation tape.

        Small consecutive offsets from the step's own rollout seed -- distinct
        from it (so the DU tapes are not literally identical to the live
        update's rollout) but still small enough to stay inside a real
        MetaDrive env's ``[start_seed, start_seed + num_scenarios)`` range for
        ordinary step seeds. ``capture_tapes`` save/restores RNG state around
        these calls regardless, so this choice has no effect on
        reproducibility of the caller's ambient RNG stream.
        """
        base = int(seed) + 1
        q = self.config.du_q
        update_seeds = tuple(base + i for i in range(q))
        evaluation_seed = base + q
        return update_seeds, evaluation_seed

    def step(self, seed: int) -> dict[str, Any]:
        self.step_count += 1
        if self.config.arm == "pal":
            return self._pal_step(seed)
        return self._dual_arm_step(seed)

    def _dual_arm_step(self, seed: int) -> dict[str, Any]:
        """Shared body for arm="uniform" and arm="lira": one update + live dual."""
        cfg = self.config
        result = self.learner.update(seed)
        # Dual gradient = mean realized per-step cost - budget, averaged over
        # the rollout's T steps (not summed).
        realized_shared_cost_mean = result["raw_shared_cost"] / result["steps"]
        dual_gradient = torch.tensor(
            [realized_shared_cost_mean - cfg.cost_budget], dtype=self.learner.dtype
        )
        self.learner.shared_lambda.projected_update_(dual_gradient, cfg.lambda_lr)

        rho_update_applied = False
        du_gradient_norm: float | None = None
        du_direct_gradient_norm: float | None = None
        du_sc_gradient_norm: float | None = None
        if cfg.arm == "lira" and self.step_count % cfg.rho_interval == 0:
            if cfg.rho_lr > 0:
                if cfg.sampling_correction:
                    replicate_updates = []
                    replicate_evaluations = []
                    update_seeds, evaluation_seed = self._derive_du_seeds(seed)
                    for replicate in range(cfg.sc_replicates):
                        offset = replicate * cfg.sc_seed_stride
                        updates, evaluation = capture_tapes(
                            self.learner,
                            update_seeds=tuple(item + offset for item in update_seeds),
                            evaluation_seed=evaluation_seed + offset,
                        )
                        replicate_updates.append(updates)
                        replicate_evaluations.append(evaluation)
                    du_result = run_independent_du_sc(
                        self.learner, replicate_updates, replicate_evaluations,
                        q=cfg.du_q, baseline=cfg.sc_baseline,
                    )
                else:
                    update_seeds, evaluation_seed = self._derive_du_seeds(seed)
                    update_tapes, evaluation_tape = capture_tapes(
                        self.learner, update_seeds=update_seeds, evaluation_seed=evaluation_seed
                    )
                    transaction = MetaDriveDirectUnroll(self.learner, MetaDriveDUConfig(q=cfg.du_q))
                    du_result = transaction.run(update_tapes, evaluation_tape)
                gradient = du_result.gradient
                if du_result.direct_gradient is not None:
                    du_direct_gradient_norm = float(du_result.direct_gradient.norm())
                if du_result.score_gradient is not None:
                    du_sc_gradient_norm = float(du_result.score_gradient.norm())
                # MetaDriveDirectUnroll.run() already raises FloatingPointError
                # on a non-finite gradient; enforce the additional
                # nonzero-estimator requirement here.
                # A zero DU+SC vector is a valid stationary/saturated response
                # (especially after LOO cancellation), so only the DU-only
                # estimator treats an exactly-zero gradient as an error.
                if not cfg.sampling_correction and torch.allclose(gradient, torch.zeros_like(gradient)):
                    raise RuntimeError(
                        "MetaDrive DU outer gradient is exactly zero; "
                        "nonzero-estimator requirement violated"
                    )
                # welfare-ascent gradient -> descent primitive requires one
                # negation.
                self.learner.simplex.apply_tangent_gradient_(-gradient, cfg.rho_lr)
                rho_update_applied = True
                du_gradient_norm = float(gradient.norm())
            # else: rho_lr<=0 means no meaningful outer step is possible
            # (apply_tangent_gradient_ requires step_size>0), and the DU
            # transaction is nontrivially expensive, so it is skipped
            # entirely rather than computed-and-discarded. With rho_lr=0 the
            # LiRA arm therefore does no extra RNG/graph work relative to the
            # uniform arm.

        return {
            "arm": cfg.arm,
            "seed": seed,
            "steps": result["steps"],
            "loss": result["loss"],
            "shared_lambda": self.learner.shared_lambda.values.tolist(),
            "rho": self.learner.simplex.rho.tolist(),
            "realized_shared_cost_mean": realized_shared_cost_mean,
            "rho_update_applied": rho_update_applied,
            "du_gradient_norm": du_gradient_norm,
            "du_direct_gradient_norm": du_direct_gradient_norm,
            "du_sc_gradient_norm": du_sc_gradient_norm,
            "sampling_correction": cfg.sampling_correction,
            "sc_baseline": cfg.sc_baseline if cfg.sampling_correction else None,
            "sc_replicates": cfg.sc_replicates if cfg.sampling_correction else None,
        }

    def heldout_evaluate(self, seeds: tuple[int, ...], horizon: int) -> dict[str, Any]:
        """Read-only, multi-seed heldout evaluation of the learner's current policy.

        Convenience wrapper around :func:`heldout_evaluate` binding this
        instance's ``self.learner`` and ``self.env``.
        """
        return heldout_evaluate(self.learner, self.env, seeds, horizon)

    def _pal_step(self, seed: int) -> dict[str, Any]:
        cfg = self.learner_config
        online = self.config
        learner = self.learner
        batch_data = learner.rollout(seed)
        O, A, LP, R = (
            batch_data["obs"], batch_data["actions"], batch_data["old_log_probs"], batch_data["rewards"],
        )
        native_costs = batch_data["native_costs"]  # (T, N)
        T, done = batch_data["steps"], batch_data["done"]

        with torch.no_grad():
            v_reward_t, v_cost_t = learner._critic_values(O)
            final_reward_b, final_cost_b = learner._critic_values(batch_data["final_obs"].unsqueeze(0))
            bootstrap_reward = torch.zeros(cfg.n_agents, dtype=learner.dtype) if done else final_reward_b[0]
            bootstrap_cost = (
                torch.zeros(cfg.n_agents, cfg.n_constraints, dtype=learner.dtype) if done else final_cost_b[0]
            )
            dones = torch.zeros(T, dtype=torch.bool)
            if done:
                dones[-1] = True
            adv_reward, ret_reward = learner._gae(R, v_reward_t, bootstrap_reward, dones, cfg.gamma, cfg.gae_lambda)
            # Per-agent cost advantage, keyed by that agent's OWN native cost
            # (not the team shared_cost broadcast used by uniform/du).
            cost_expanded = native_costs.unsqueeze(-1)  # (T, N, K=1)
            adv_cost, ret_cost = learner._gae(cost_expanded, v_cost_t, bootstrap_cost, dones, cfg.gamma, cfg.gae_lambda)

        learner.optimizer.zero_grad()
        v_reward, v_cost = learner._critic_values(O)
        # Frozen-uniform rho row: with agent_index picking one column, this
        # makes PPOObjective.evaluate's ``n_agents * shared_lambda *
        # rho[:, agent_index]`` reduce to exactly ``pal_lambdas[i]`` -- reusing
        # the one canonical PPO surrogate with no second implementation.
        rho_uniform = torch.full((1, cfg.n_agents), 1.0 / cfg.n_agents, dtype=learner.dtype)
        actor_total = torch.zeros((), dtype=learner.dtype)
        for i in range(cfg.n_agents):
            dist = learner.actors[i].distribution(O[:, i, :])
            nlp = dist.log_prob(A[:, i, :]).sum(-1)
            ent = dist.entropy().sum(-1)
            ones = torch.ones(T, dtype=learner.dtype)
            batch = PPOBatch(LP[:, i], adv_reward[:, i], adv_cost[:, i, :].T, ones, ones, ent)
            lam_i = self.pal_lambdas[i : i + 1]
            actor_total = actor_total + learner.ppo.evaluate(nlp, batch, rho_uniform, lam_i, i).total / cfg.n_agents
        critic_loss = torch.nn.functional.mse_loss(v_reward, ret_reward) + torch.nn.functional.mse_loss(
            v_cost, ret_cost
        )
        total = actor_total + cfg.critic_loss_coefficient * critic_loss
        total.backward()
        learner.optimizer.step()
        learner.update_count += 1

        per_agent_mean_native_cost = native_costs.mean(dim=0)  # (N,)
        dual_gradient = per_agent_mean_native_cost - (online.cost_budget / cfg.n_agents)
        self.pal_lambdas = torch.clamp(self.pal_lambdas + online.lambda_lr * dual_gradient, min=0.0)

        realized_shared_cost_mean = float(native_costs.sum(dim=-1).mean(dim=0))
        return {
            "arm": "pal",
            "seed": seed,
            "steps": T,
            "loss": float(total.detach()),
            "pal_lambdas": self.pal_lambdas.tolist(),
            "rho": learner.simplex.rho.tolist(),
            "realized_shared_cost_mean": realized_shared_cost_mean,
            "rho_update_applied": False,
            "du_gradient_norm": None,
        }


@torch.no_grad()
def run_heldout_episode(learner: MetaDriveStaticLearner, env: Any, seed: int, horizon: int) -> dict[str, Any]:
    """One deterministic, no-gradient rollout of ``learner``'s current actors.

    Held-out in the sense of "not used for any gradient/dual/rho update": this
    only reads the actors (mean action, not a stochastic sample, so the
    reported metrics reflect the policy's current operating point rather than
    sampling noise) and calls ``env.step``/``env.reset``; it never touches
    ``learner.optimizer``, ``learner.shared_lambda``, or ``learner.simplex``.
    Caller is responsible for picking ``seed`` values distinct from the
    seeds used to train ``learner`` (see ``scripts/train_metadrive.py``).
    """
    cfg = learner.config
    obs = torch.as_tensor(env.reset(seed), dtype=learner.dtype)
    agent_ids = env.agent_ids
    episode_reward = torch.zeros(cfg.n_agents, dtype=learner.dtype)
    episode_native_cost = torch.zeros(cfg.n_agents, dtype=learner.dtype)
    arrived = [False] * cfg.n_agents
    crashed = [False] * cfg.n_agents
    out_of_road = [False] * cfg.n_agents
    other_crash = [False] * cfg.n_agents
    max_route_completion = [0.0] * cfg.n_agents
    steps = 0
    for _ in range(horizon):
        actions = torch.empty((cfg.n_agents, cfg.action_dim), dtype=learner.dtype)
        for i in range(cfg.n_agents):
            actions[i] = learner.actors[i].distribution(obs[i]).mean
        env_actions = torch.clamp(actions, -1.0, 1.0).to(torch.float32).numpy()
        next_obs, native_reward, done, accounting = env.step(env_actions)
        obs = torch.as_tensor(next_obs, dtype=learner.dtype)
        episode_reward += torch.as_tensor(native_reward, dtype=learner.dtype)
        episode_native_cost += torch.as_tensor(accounting["native_cost"], dtype=learner.dtype)
        for i, aid in enumerate(agent_ids):
            if accounting["arrive_dest"].get(aid, False):
                arrived[i] = True
            if accounting["crash_vehicle"].get(aid, False):
                crashed[i] = True
            if accounting.get("out_of_road", {}).get(aid, False):
                out_of_road[i] = True
            # "crash" is the aggregate MetaDrive termination state (vehicle,
            # object, building, or sidewalk); a True here not already
            # explained by crashed[i] means a non-vehicle collision.
            if accounting.get("crash", {}).get(aid, False) and not accounting["crash_vehicle"].get(aid, False):
                other_crash[i] = True
            max_route_completion[i] = max(
                max_route_completion[i],
                float(accounting.get("route_completion", {}).get(aid, 0.0)),
            )
        steps += 1
        if done:
            break
    return {
        "seed": seed,
        "steps": steps,
        "episode_return_per_agent": episode_reward.tolist(),
        "episode_welfare": float(episode_reward.sum()),
        "episode_native_cost_per_agent": episode_native_cost.tolist(),
        "episode_native_cost_total": float(episode_native_cost.sum()),
        "arrived_per_agent": arrived,
        "success_rate": float(sum(arrived)) / cfg.n_agents,
        "crashed_per_agent": crashed,
        "out_of_road_per_agent": out_of_road,
        "other_crash_per_agent": other_crash,
        "max_route_completion_per_agent": max_route_completion,
        "mean_max_route_completion": float(sum(max_route_completion) / cfg.n_agents),
    }


def heldout_evaluate(
    learner: MetaDriveStaticLearner, env: Any, seeds: tuple[int, ...], horizon: int
) -> dict[str, Any]:
    """Read-only heldout evaluation over several episodes; per-episode + mean summary."""
    episodes = [run_heldout_episode(learner, env, seed, horizon) for seed in seeds]
    n = len(episodes)
    return {
        "seeds": list(seeds),
        "episodes": episodes,
        "mean_episode_welfare": sum(e["episode_welfare"] for e in episodes) / n,
        "mean_success_rate": sum(e["success_rate"] for e in episodes) / n,
        "mean_episode_native_cost_total": sum(e["episode_native_cost_total"] for e in episodes) / n,
        "mean_max_route_completion": sum(e["mean_max_route_completion"] for e in episodes) / n,
    }


__all__ = [
    "MetaDriveOnlineConfig",
    "MetaDriveOnlineLearner",
    "ARMS",
    "run_heldout_episode",
    "heldout_evaluate",
]
