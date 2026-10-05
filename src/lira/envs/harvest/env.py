"""Melting Pot ``commons_harvest__open`` environment wrapper.

The shared constraint is ecological depletion, not reward: a small
instrumentation patch (``patches/meltingpot_apple_count.patch``) mirrors the
number of live apples from the native ``apples`` state group into each
player's observation without changing rewards or transitions.  The
instantaneous cost is ``1 - live_apples / 64``, where 64 is the number of
apple sites in the substrate map.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


# Melting Pot source revision the patch and all results were produced with.
SOURCE_COMMIT = "817f8c1974863a91909c04c7a69dd33993199ec6"
# Fallback for source trees that are not git checkouts (e.g. an unpacked
# archive): set this variable to the revision the tree was taken from.
SOURCE_COMMIT_ENV = "MELTINGPOT_SOURCE_COMMIT"
SUBSTRATE = "commons_harvest__open"
N_AGENTS = 7
N_ACTIONS = 8
N_CONSTRAINTS = 1
INITIAL_APPLES = 64
APPLE_COUNT_KEY = "RESOURCE_APPLE_COUNT"


class CommonsHarvestContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class HarvestStep:
    rgb: np.ndarray
    ready: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    done: bool
    info: Mapping[str, Any]


def _source_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except FileNotFoundError:
        return os.environ.get(SOURCE_COMMIT_ENV, "")


class CommonsHarvestEnv:
    """Seed-bound seven-player ``commons_harvest__open`` environment."""

    n_agents = N_AGENTS
    n_constraints = N_CONSTRAINTS
    n_actions = N_ACTIONS

    def __init__(self, source_path: str | Path, *, env_seed: int) -> None:
        source_path = Path(source_path).resolve()
        if _source_commit(source_path) != SOURCE_COMMIT:
            raise CommonsHarvestContractError(
                f"Melting Pot source at {source_path} is not at revision {SOURCE_COMMIT}"
            )
        source = str(source_path)
        if source not in sys.path:
            sys.path.insert(0, source)
        substrate = importlib.import_module("meltingpot.substrate")
        substrate_builder = importlib.import_module(
            "meltingpot.utils.substrates.builder"
        )
        config_module = importlib.import_module(
            "meltingpot.configs.substrates.commons_harvest__open"
        )
        # The apple-count key is instrumentation-only and is included in the
        # individual observation contract so no rendered-RGB proxy is needed.
        config = substrate.get_config(SUBSTRATE)
        if APPLE_COUNT_KEY not in config.individual_observation_names:
            raise CommonsHarvestContractError(
                "live-apple instrumentation is absent; apply patches/meltingpot_apple_count.patch"
            )
        if tuple(config.default_player_roles) != ("default",) * N_AGENTS:
            raise CommonsHarvestContractError("seven-player role contract drifted")
        if config_module.ASCII_MAP.count("A") != INITIAL_APPLES:
            raise CommonsHarvestContractError("apple-site count drifted")

        # Melting Pot binds the Lua seed when the substrate is built. Its
        # public factory obtains that seed from Python's random module.
        # Saving/restoring the caller state makes paired environments
        # reproducible without leaking this seed choice to learner RNG.
        state = random.getstate()
        original_builder = substrate_builder.builder
        try:
            random.seed(int(env_seed))
            # The public substrate factory does not expose ``env_seed``.
            # Inject it at the builder call, matching the upstream
            # determinism test instead of relying on the factory's
            # process-global random seed selection.
            def pinned_builder(*args: Any, **kwargs: Any) -> Any:
                kwargs["env_seed"] = int(env_seed)
                return original_builder(*args, **kwargs)
            substrate_builder.builder = pinned_builder
            self._env = substrate.build(SUBSTRATE, roles=("default",) * N_AGENTS)
        finally:
            substrate_builder.builder = original_builder
            random.setstate(state)
        self.env_seed = int(env_seed)
        self._last_native = None
        self._closed = False

    def _convert(self, timestep: Any) -> HarvestStep:
        obs = timestep.observation
        if len(obs) != N_AGENTS:
            raise CommonsHarvestContractError("native observation count drifted")
        rgb = np.stack([np.asarray(row["RGB"], dtype=np.uint8) for row in obs])
        ready = np.asarray([float(np.asarray(row["READY_TO_SHOOT"]).reshape(())) for row in obs], dtype=np.float32)
        counts = np.asarray([float(np.asarray(row[APPLE_COUNT_KEY]).reshape(())) for row in obs])
        if not np.allclose(counts, counts[0]):
            raise CommonsHarvestContractError("shared live-apple count differs across players")
        live = float(counts[0])
        if not 0.0 <= live <= INITIAL_APPLES:
            raise CommonsHarvestContractError("live-apple count outside map capacity")
        rewards = np.asarray(timestep.reward, dtype=np.float32)
        if rewards.shape != (N_AGENTS,):
            raise CommonsHarvestContractError("native reward vector shape drifted")
        events = tuple(self._env.events())
        eaten = sum(name == "edible_consumed" for name, _ in events)
        zaps = sum(name == "zap" for name, _ in events)
        cost = np.asarray([1.0 - live / INITIAL_APPLES], dtype=np.float32)
        done = bool(getattr(timestep, "last", lambda: False)())
        self._last_native = timestep
        return HarvestStep(
            rgb, ready, rewards, cost, done,
            {"live_apples": live, "resource_depletion": float(cost[0]),
             "apple_events": int(eaten), "zap_events": int(zaps), "events": events},
        )

    def reset(self) -> HarvestStep:
        if self._closed:
            raise RuntimeError("environment is closed")
        return self._convert(self._env.reset())

    def step(self, actions: np.ndarray) -> HarvestStep:
        actions = np.asarray(actions)
        if actions.shape != (N_AGENTS,) or not np.issubdtype(actions.dtype, np.integer):
            raise CommonsHarvestContractError("actions must be integer shape [7]")
        if np.any(actions < 0) or np.any(actions >= N_ACTIONS):
            raise CommonsHarvestContractError("action outside the 8-category action set")
        return self._convert(self._env.step(actions.astype(np.int32).tolist()))

    @property
    def native_timestep(self) -> Any:
        if self._last_native is None:
            raise RuntimeError("native timestep requested before reset")
        return self._last_native

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True
