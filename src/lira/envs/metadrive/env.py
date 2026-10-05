"""Official MetaDrive multi-agent Intersection adapter (N=4 agents, K=1 shared cost).

Wraps ``metadrive.envs.marl_envs.marl_intersection.MultiAgentIntersectionEnv``
from the unmodified ``metadrive-simulator`` release without editing any
source file. The official default multi-agent config leaves
``force_seed_spawn_manager=False`` and ``random_spawn_lane_index=True``,
which makes repeated resets at a fixed seed non-reproducible across fresh
instances. Both keys are pinned here so a fixed-seed rollout is exactly
reproducible.

The native scalar ``cost`` field is per-agent (1.0 on ``crash_vehicle`` /
``crash_object``, 0.0 on ``out_of_road`` in the multi-agent default). This
adapter reports that native per-agent cost alongside a declared team-level
K=1 aggregation (the sum across the fixed agent set); the aggregation is not
claimed to be a native official team constraint.

The official per-agent dict API drops an agent's key the step it
individually terminates (crash/out-of-road/arrival), even while the team
``__all__`` flag is still False -- ``allow_respawn=False`` means that agent
never reappears. Treating a single agent's drop as the end of the team
episode would discard the remaining agents' rollout as soon as the first
agent exits. This adapter instead holds a dropped agent's last observation
and reports zero reward/cost for it, but keeps forwarding actions for the
agents still active until the official env reports the team boundary
(``__all__`` True) or every agent has dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetaDriveIntersectionConfig:
    n_agents: int = 4
    n_constraints: int = 1
    horizon: int = 1000
    num_scenarios: int = 256
    start_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_agents < 2 or self.n_constraints != 1:
            raise ValueError("MetaDrive Intersection adapter requires N>=2 and K=1")
        if self.horizon < 1 or self.num_scenarios < 1:
            raise ValueError("horizon and num_scenarios must be positive")

    def native_config(self) -> dict:
        return dict(
            num_agents=self.n_agents,
            allow_respawn=False,
            crash_done=True,
            out_of_road_done=True,
            horizon=self.horizon,
            use_render=False,
            num_scenarios=self.num_scenarios,
            start_seed=self.start_seed,
            force_seed_spawn_manager=True,
            random_spawn_lane_index=False,
        )


class MetaDriveIntersectionEnv:
    """Fixed-identity numpy adapter over the official multi-agent Intersection env."""

    def __init__(self, config: MetaDriveIntersectionConfig | None = None):
        from metadrive.envs.marl_envs.marl_intersection import MultiAgentIntersectionEnv

        self.config = config or MetaDriveIntersectionConfig()
        self.env = MultiAgentIntersectionEnv(self.config.native_config())
        self.agent_ids: list[str] | None = None
        self._obs_dim: int | None = None
        self._action_dim: int | None = None
        self._last_obs: np.ndarray | None = None
        self._alive: set[str] = set()
        self.step_count = 0

    def reset(self, seed: int) -> np.ndarray:
        obs, _info = self.env.reset(seed=seed)
        self.agent_ids = sorted(obs.keys())
        if len(self.agent_ids) != self.config.n_agents:
            raise RuntimeError("MetaDrive reset did not return N stable agent identities")
        stacked = np.stack([np.asarray(obs[a], dtype=np.float32) for a in self.agent_ids])
        self._obs_dim = stacked.shape[-1]
        self._action_dim = int(self.env.action_space[self.agent_ids[0]].shape[0])
        self._last_obs = stacked
        self._alive = set(self.agent_ids)
        self.step_count = 0
        return stacked

    def step(self, actions: np.ndarray):
        """One step; ``actions`` is (N, action_dim). Returns the fixed-identity contract."""
        if self.agent_ids is None:
            raise RuntimeError("reset() must be called before step()")
        if not self._alive:
            raise RuntimeError("step() called after every agent already dropped/terminated")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.config.n_agents, self._action_dim):
            raise ValueError("actions must have shape (n_agents, action_dim)")
        # Only forward actions for agents the official env still considers
        # active; a dropped agent (allow_respawn=False) never reappears.
        action_dict = {aid: actions[i] for i, aid in enumerate(self.agent_ids) if aid in self._alive}
        obs, rew, term, trunc, info = self.env.step(action_dict)
        self.step_count += 1
        present = set(obs.keys())
        newly_dropped = sorted(self._alive - present)
        self._alive = present

        next_obs_rows = []
        native_reward = np.zeros(self.config.n_agents, dtype=np.float64)
        native_cost = np.zeros(self.config.n_agents, dtype=np.float64)
        crash = {}
        arrive = {}
        route_completion = {}
        # Under this env's config (``crash_done=True``, ``out_of_road_done=True``)
        # a vehicle crash is only one of several ways an agent's episode
        # individually ends, so ``out_of_road`` and the aggregate ``crash``
        # state (crash_object/building/sidewalk) are surfaced as well. See
        # metadrive.constants.TerminationState for the native key names.
        out_of_road = {}
        crash_any = {}
        for i, aid in enumerate(self.agent_ids):
            if aid in present:
                next_obs_rows.append(np.asarray(obs[aid], dtype=np.float32))
                native_reward[i] = float(rew[aid])
                native_cost[i] = float(info[aid].get("cost", 0.0))
                crash[aid] = bool(info[aid].get("crash_vehicle", False))
                arrive[aid] = bool(info[aid].get("arrive_dest", False))
                route_completion[aid] = float(info[aid].get("route_completion", 0.0))
                out_of_road[aid] = bool(info[aid].get("out_of_road", False))
                crash_any[aid] = bool(info[aid].get("crash", False))
            else:
                # Hold the last observation and report zero reward/cost for
                # an agent that has already individually terminated, rather
                # than inventing state the official API no longer exposes.
                next_obs_rows.append(self._last_obs[i])
                crash[aid] = False
                arrive[aid] = False
                route_completion[aid] = 0.0
                out_of_road[aid] = False
                crash_any[aid] = False
        next_obs = np.stack(next_obs_rows)
        self._last_obs = next_obs
        done = bool(term.get("__all__", False) or trunc.get("__all__", False)) or not present
        accounting = {
            "shared_cost": float(native_cost.sum()),
            "native_cost": native_cost,
            "crash_vehicle": crash,
            "arrive_dest": arrive,
            "route_completion": route_completion,
            "out_of_road": out_of_road,
            "crash": crash_any,
            "step": self.step_count,
            "identity_dropped": newly_dropped,
        }
        return next_obs, native_reward, done, accounting

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            raise RuntimeError("call reset() before reading obs_dim")
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
            raise RuntimeError("call reset() before reading action_dim")
        return self._action_dim

    def close(self) -> None:
        self.env.close()
