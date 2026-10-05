"""Learning adapter for Melting Pot Commons Harvest.

The trainable categorical policies receive a compact observation containing
the action proposed by Melting Pot's pretrained Harvest bot
(``commons_harvest__open__free_0``, the "teacher"), the native ready-to-zap
signal, normalized live-apple stock, and a fixed player-slot identity.  The
bot proposal is an action prior shared by every compared arm; all executed
actions and PPO likelihoods still come from the trainable policy, which is
warm-started to follow the proposal (``initialize_teacher_prior_``).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import itertools
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .env import CommonsHarvestEnv, INITIAL_APPLES, N_ACTIONS, N_AGENTS
from .interfaces import AdapterStep, CostTransition, StepObservation


OBS_DIM = N_ACTIONS + 1 + 1 + N_AGENTS
BOT_NAME = "commons_harvest__open__free_0"


def counterbalanced_arm_order(names: Sequence[str], *, seed: int, phase: int) -> tuple[str, ...]:
    """Cycle through every arm permutation, offset by seed, without favoritism."""
    names = tuple(names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("counterbalance names must be distinct")
    permutations = tuple(itertools.permutations(names))
    return permutations[(int(seed) + int(phase)) % len(permutations)]


def heldout_arm_order(names: Sequence[str], *, seed: int, rollout_index: int) -> tuple[str, ...]:
    """Counterbalance evaluation without coupling order to training horizon."""
    if rollout_index < 0:
        raise ValueError("rollout_index must be nonnegative")
    return counterbalanced_arm_order(names, seed=seed, phase=100_000 + rollout_index)


def teacher_tape_digest(cache: Mapping[str, tuple[int, Any]]) -> str:
    """Hash the shared teacher action/recurrent-state tape in stable key order."""
    digest = hashlib.sha256()
    for key in sorted(cache):
        digest.update(key.encode("utf-8"))
        action, state = cache[key]
        digest.update(int(action).to_bytes(8, "little", signed=True))
        _update_digest_value(digest, state)
    return digest.hexdigest()


def _update_digest_value(digest: Any, value: Any) -> None:
    """Canonicalize nested NumPy/TensorFlow SavedModel state for hashing."""
    if isinstance(value, np.ndarray):
        digest.update(b"ndarray")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence")
        digest.update(len(value).to_bytes(8, "little"))
        for item in value:
            _update_digest_value(digest, item)
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=repr):
            _update_digest_value(digest, key)
            _update_digest_value(digest, value[key])
    elif hasattr(value, "numpy"):
        _update_digest_value(digest, np.asarray(value.numpy()))
    elif value is None or isinstance(value, (bool, int, float, str, bytes)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
    elif hasattr(value, "_fields"):
        digest.update(type(value).__name__.encode("utf-8"))
        _update_digest_value(digest, tuple(getattr(value, field) for field in value._fields))
    else:
        raise TypeError(f"unsupported teacher-state type for digest: {type(value)!r}")


def seeded_teacher_initial_states(policies: Sequence[Any], *, seed: int) -> list[Any]:
    """Create repeatable SavedModel initial states without perturbing caller RNG.

    Melting Pot's SavedModelPolicy draws its explicit recurrent PRNG key via
    ``random.getrandbits``; the returned state then carries the key through
    each stochastic action step. Pinning this boundary seeds the teacher tape.
    """
    state = random.getstate()
    try:
        random.seed(int(seed))
        return [policy.initial_state() for policy in policies]
    finally:
        random.setstate(state)


@dataclass(frozen=True)
class CommonsHarvestLearningProvenance:
    domain: str = "melting_pot"
    substrate: str = "commons_harvest__open"
    n_agents: int = N_AGENTS
    cost_name: str = "ecological_resource_depletion_k1"
    cost_basis: str = "one_minus_native_live_apple_stock_over_64"
    notes: str = (
        "Instrumentation exposes the native live-apple state without changing "
        "reward or transition dynamics. Fixed slots 0-1 use insideSpawnPoints; "
        "slots 2-6 use spawnPoints."
    )


class CommonsHarvestLearningAdapter:
    """N=7/K=1 typed adapter consumed by the categorical learner and lookahead."""

    n_agents = N_AGENTS
    n_constraints = 1
    categories = N_ACTIONS
    obs_dim = OBS_DIM
    agent_ids = tuple(f"player_{index}" for index in range(N_AGENTS))
    provenance = CommonsHarvestLearningProvenance()

    def __init__(
        self, source_path: str | Path, *, max_steps: int = 100,
        shared_policies: Sequence[Any] | None = None,
        shared_initial_states: Sequence[Any] | None = None,
        teacher_cache: dict[str, tuple[int, Any]] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.source_path = Path(source_path)
        self.max_steps = int(max_steps)
        self._env: CommonsHarvestEnv | None = None
        if shared_policies is not None and len(shared_policies) != N_AGENTS:
            raise ValueError("shared_policies must contain exactly seven policies")
        if shared_initial_states is not None and len(shared_initial_states) != N_AGENTS:
            raise ValueError("shared_initial_states must contain exactly seven states")
        self._policies = list(shared_policies or ())
        self._owns_policies = shared_policies is None
        self._shared_initial_states = (
            tuple(copy.deepcopy(value) for value in shared_initial_states)
            if shared_initial_states is not None else None
        )
        self._teacher_cache = teacher_cache
        self._teacher_states: list[Any] = []
        self._step = 0
        self._seed: int | None = None
        self._cumulative_cost = 0.0
        self._last: StepObservation | None = None
        self._last_native: Any | None = None

    def _ensure_policies(self) -> None:
        if self._policies:
            return
        from meltingpot import bot
        self._policies = [bot.build(BOT_NAME) for _ in range(N_AGENTS)]

    def _teacher_actions(self) -> np.ndarray:
        assert self._env is not None
        next_states: list[Any] = []
        actions: list[int] = []
        for index, policy in enumerate(self._policies):
            local = self._env.native_timestep._replace(
                observation=self._env.native_timestep.observation[index]
            )
            # Treat the pretrained bot's proposal as an exogenous prior tape:
            # the first visit for (env seed, step, player) materializes the
            # action and recurrent successor, and every matched arm replays
            # that same record. This shares the *action* stream across arms
            # only. Each arm still owns a separately constructed/reset
            # CommonsHarvestEnv, and the dmlab2d engine does not return the
            # same reset observation for two independently built environments
            # given the same env_seed, even within one process. Arms in a
            # matched run therefore never share native game state, only the
            # cached teacher-action tape; welfare/cost differences between
            # arms are not a common-random-numbers comparison.
            key = f"{self._seed}:{self._step}:{index}"
            cached = None if self._teacher_cache is None else self._teacher_cache.get(key)
            if cached is None:
                action, state = policy.step(local, self._teacher_states[index])
                if self._teacher_cache is not None:
                    self._teacher_cache[key] = (int(action), copy.deepcopy(state))
            else:
                action, state = cached[0], copy.deepcopy(cached[1])
            actions.append(int(action))
            next_states.append(state)
        self._teacher_states = next_states
        result = np.asarray(actions, dtype=np.int64)
        if result.shape != (N_AGENTS,) or np.any(result < 0) or np.any(result >= N_ACTIONS):
            raise ValueError("pretrained Harvest bot emitted an invalid action")
        return result

    def _observation(self, native: Any, teacher_actions: np.ndarray) -> StepObservation:
        live = float(native.info["live_apples"])
        observations: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        for index, agent_id in enumerate(self.agent_ids):
            row = np.zeros(OBS_DIM, dtype=np.float32)
            row[int(teacher_actions[index])] = 1.0
            row[N_ACTIONS] = float(native.ready[index])
            row[N_ACTIONS + 1] = live / INITIAL_APPLES
            row[N_ACTIONS + 2 + index] = 1.0
            observations[agent_id] = row
            masks[agent_id] = np.ones(N_ACTIONS, dtype=np.uint8)
        resource = np.full(N_AGENTS, live / INITIAL_APPLES, dtype=np.float32)
        return StepObservation(
            agent_ids=self.agent_ids,
            observations=observations,
            world_state=np.asarray([live / INITIAL_APPLES], dtype=np.float32),
            action_masks=masks,
            unit_types=np.arange(N_AGENTS, dtype=np.int64),
            unit_health=resource,
            unit_alive=np.ones(N_AGENTS, dtype=bool),
            step=self._step,
        )

    def reset(self, seed: int) -> StepObservation:
        if self._env is not None:
            self._env.close()
        self._env = CommonsHarvestEnv(self.source_path, env_seed=int(seed))
        self._ensure_policies()
        if self._shared_initial_states is not None:
            self._teacher_states = [copy.deepcopy(value) for value in self._shared_initial_states]
        else:
            py_state = random.getstate()
            try:
                random.seed(int(seed) + 7919)
                self._teacher_states = [policy.initial_state() for policy in self._policies]
            finally:
                random.setstate(py_state)
        self._step = 0
        self._seed = int(seed)
        self._cumulative_cost = 0.0
        native = self._env.reset()
        self._last_native = native
        self._last = self._observation(native, self._teacher_actions())
        return self._last

    def step(self, actions: Sequence[int]) -> AdapterStep:
        if self._env is None or self._last is None or self._last_native is None:
            raise RuntimeError("reset() must be called before step()")
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (N_AGENTS,):
            raise ValueError("Harvest requires exactly seven categorical actions")
        before = self._last
        native = self._env.step(action_array)
        self._step += 1
        terminated = bool(native.done or self._step >= self.max_steps)
        cost = float(native.costs[0])
        self._cumulative_cost += cost
        after = self._observation(native, self._teacher_actions())
        per_agent = np.full(N_AGENTS, cost / N_AGENTS, dtype=np.float32)
        transition = CostTransition(
            per_agent_damage=per_agent,
            team_damage=cost,
            cumulative_team_damage=self._cumulative_cost,
            death_events=np.zeros(N_AGENTS, dtype=bool),
            unit_types=after.unit_types.copy(),
            unit_health_before=before.unit_health.copy(),
            unit_health_after=after.unit_health.copy(),
            unit_alive_before=before.unit_alive.copy(),
            unit_alive_after=after.unit_alive.copy(),
            step_before=before.step,
            step_after=after.step,
            done=terminated,
            timeout=bool(self._step >= self.max_steps and not native.done),
            death_termination=False,
            battle_won=None,
            provenance=self.provenance,
        )
        self._last_native = native
        self._last = after
        rewards = np.asarray(native.rewards, dtype=np.float32)
        return AdapterStep(
            observation=after,
            team_reward=float(rewards.mean()),
            rewards=rewards,
            transition=transition,
            terminated=terminated,
            info=dict(native.info),
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._owns_policies:
            for policy in self._policies:
                policy.close()
        self._policies = []
        self._teacher_states = []


def initialize_teacher_prior_(learner: Any, *, strength: float = 8.0) -> None:
    """Initialize every trainable policy to follow its teacher-action feature."""
    if strength <= 0 or not np.isfinite(strength):
        raise ValueError("teacher-prior strength must be finite and positive")
    with torch.no_grad():
        for actor in learner.actor:
            first = actor.net[0]
            last = actor.net[2]
            first.weight.zero_()
            first.bias.zero_()
            last.weight.zero_()
            last.bias.zero_()
            width = min(N_ACTIONS, first.weight.shape[0])
            first.weight[:width, :width] = torch.eye(width, dtype=first.weight.dtype)
            last.weight[:width, :width] = torch.eye(width, dtype=last.weight.dtype) * float(strength)


__all__ = [
    "BOT_NAME",
    "CommonsHarvestLearningAdapter",
    "CommonsHarvestLearningProvenance",
    "OBS_DIM",
    "counterbalanced_arm_order",
    "heldout_arm_order",
    "initialize_teacher_prior_",
    "seeded_teacher_initial_states",
    "teacher_tape_digest",
]
