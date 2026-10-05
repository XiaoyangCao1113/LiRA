"""Read-only, full-episode CityLearn held-out evaluator.

Training scores policies on short (``rollout_steps``-length) batches whose
advantages are normalized in-batch. ``evaluate_heldout_arm`` instead runs one
full CityLearn episode (the environment is stepped until ``done``, i.e. the
schema's real horizon: 719 steps for ``citylearn_challenge_2023_phase_1``) on
a fresh environment seeded with an *evaluation* seed distinct from the
training seed, and reports two native, un-normalized episode totals:

- ``welfare_sum``: sum over all buildings and steps of the native per-step
  ``ComfortReward``.
- ``native_shared_cost_sum``: sum over all steps of the native district
  cap-excess cost ``max(0, district_net_electricity - cap_kwh_per_step)``
  (``accounting["C_cap"]`` from ``CityLearnDistributedEnv.step``).

Read-only: runs entirely under ``torch.no_grad()``, calls no optimizer or
``.backward()``, and never writes to any actor parameter. A deterministic
``eval_seed`` (default ``training_seed + EVAL_SEED_OFFSET``) makes the
evaluation identical across every arm and hyperparameter setting trained from
the same training seed, so cross-arm comparisons are paired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .env import CityLearnDistributedEnv, CityLearnEnvConfig
from .ppo_update import sample_actions

EVAL_SEED_OFFSET = 5000
MAX_EPISODE_STEPS_SAFEGUARD = 4000  # generous upper bound; real schema horizon is 720 steps


def eval_seed_for(training_seed: int, *, offset: int = EVAL_SEED_OFFSET) -> int:
    """Deterministic training_seed -> eval_seed mapping (identical across every
    arm and hyperparameter cell trained from the same ``training_seed``)."""
    return int(training_seed) + int(offset)


@dataclass(frozen=True)
class HeldoutEvalResult:
    dataset_name: str
    eval_seed: int
    steps: int
    welfare_sum: float
    native_shared_cost_sum: float
    agent_welfare_sums: list[float]

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "eval_seed": self.eval_seed,
            "steps": self.steps,
            "welfare_sum": self.welfare_sum,
            "native_shared_cost_sum": self.native_shared_cost_sum,
            "agent_welfare_sums": self.agent_welfare_sums,
        }


def evaluate_heldout_arm(
    *,
    actors: Sequence[nn.Module],
    dataset_name: str,
    eval_seed: int,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    on_step: Callable[[CityLearnDistributedEnv, np.ndarray, np.ndarray, dict], None] | None = None,
) -> HeldoutEvalResult:
    """Run one full episode on a fresh, held-out-seeded env; read-only.

    ``actors`` are evaluated in ``torch.no_grad()`` with a fresh env whose
    ``random_seed=eval_seed`` (never the training seed), stepped until
    ``done`` -- the schema's real full-episode horizon, not a short training
    snippet. No parameter of ``actors`` is read via anything that could
    create autograd state, and nothing is written to them.
    """
    was_training = [actor.training for actor in actors]
    for actor in actors:
        actor.eval()
    try:
        torch.manual_seed(eval_seed)
        env = CityLearnDistributedEnv(CityLearnEnvConfig(dataset_name=dataset_name, seed=eval_seed))
        obs, _state = env.reset()
        welfare_sum = 0.0
        agent_welfare_sums = np.zeros(len(actors), dtype=np.float64)
        native_shared_cost_sum = 0.0
        steps = 0
        done = False
        with torch.no_grad():
            while not done:
                if steps >= MAX_EPISODE_STEPS_SAFEGUARD:
                    raise RuntimeError(
                        "CityLearn heldout episode did not terminate within "
                        f"{MAX_EPISODE_STEPS_SAFEGUARD} steps"
                    )
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_t, _log_prob_t = sample_actions(actors, obs_t, action_scale, action_bias)
                next_obs, _next_state, welfare, done, accounting = env.step(action_t.squeeze(0).numpy())
                welfare_arr = np.asarray(welfare, dtype=np.float64)
                if on_step is not None:
                    on_step(env, action_t.squeeze(0).numpy(), welfare_arr, accounting)
                welfare_sum += float(welfare_arr.sum())
                agent_welfare_sums += welfare_arr
                native_shared_cost_sum += float(accounting["C_cap"])
                obs = next_obs
                steps += 1
        if not np.isfinite(welfare_sum) or not np.isfinite(native_shared_cost_sum):
            raise FloatingPointError("CityLearn heldout evaluation produced a non-finite total")
        return HeldoutEvalResult(
            dataset_name=dataset_name, eval_seed=eval_seed, steps=steps,
            welfare_sum=welfare_sum, native_shared_cost_sum=native_shared_cost_sum,
            agent_welfare_sums=agent_welfare_sums.tolist(),
        )
    finally:
        for actor, mode in zip(actors, was_training):
            actor.train(mode)
