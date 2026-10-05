"""Official CityLearn v2.5.0 ``citylearn_challenge_2023_phase_1`` distributed adapter.

Three agents (``Building_1..3``), each controlling the action vector
``(dhw_storage, electrical_storage, cooling_device)``; per-building welfare is
CityLearn's native ``ComfortReward``. The single shared cost is the positive
excess of district net electricity over a fixed ``10.0 kWh/step`` grid cap.
The real ``citylearn.citylearn.CityLearnEnv`` is imported lazily inside
``__init__`` so this module stays importable without the (heavy, optional)
``citylearn`` dependency installed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


DEFAULT_DATASET_NAME = "citylearn_challenge_2023_phase_1"
EXPECTED_BUILDING_IDS = ("Building_1", "Building_2", "Building_3")
EXPECTED_ACTION_ORDER = ("dhw_storage", "electrical_storage", "cooling_device")
CAP_KWH_PER_STEP = 10.0


def attribute_global_cap_excess(net_electricity: np.ndarray, cap_kwh_per_step: float) -> tuple[float, np.ndarray]:
    """Conservatively attribute native district excess by positive building load.

    Returns ``C=max(0,sum(e)-cap)`` unchanged plus ``c_i=C*[e_i]_+/sum_j[e_j]_+``.
    The attribution is zero when the positive-load denominator is zero and
    otherwise sums to C up to floating-point precision.
    """
    net = np.asarray(net_electricity, dtype=np.float64)
    if net.ndim != 1 or not np.isfinite(net).all() or not np.isfinite(cap_kwh_per_step):
        raise ValueError("net electricity must be a finite vector and cap must be finite")
    if cap_kwh_per_step < 0:
        raise ValueError("cap_kwh_per_step must be nonnegative")
    global_excess = float(max(0.0, float(net.sum()) - cap_kwh_per_step))
    positive = np.maximum(net, 0.0)
    denominator = float(positive.sum())
    local = global_excess * positive / denominator if denominator > 0.0 else np.zeros_like(positive)
    return global_excess, local


@dataclass(frozen=True)
class CityLearnEnvConfig:
    dataset_name: str = DEFAULT_DATASET_NAME
    seed: int = 0
    cap_kwh_per_step: float = CAP_KWH_PER_STEP


class CityLearnDistributedEnv:
    """N=3 building, K=1 (district capacity) distributed CityLearn adapter."""

    n_agents = 3
    n_constraints = 1

    def __init__(self, config: CityLearnEnvConfig):
        from citylearn.citylearn import CityLearnEnv
        from citylearn.data import DataSet
        from citylearn.reward_function import ComfortReward

        self.config = config
        # Resolve the dataset name to its cached local schema.json path rather
        # than passing the bare name through to CityLearnEnv. CityLearnEnv's
        # schema setter, given a bare string, always calls
        # DataSet.get_dataset_names() first, which makes an unconditional
        # GitHub API request (and is subject to that API's unauthenticated
        # rate limit) even when the dataset is already cached locally.
        # DataSet.get_dataset() only touches the network on a genuine cache
        # miss, so resolving the path here once keeps every subsequent
        # CityLearnEnv construction (each seed, arm and lookahead replicate)
        # offline after the first download.
        schema_path = DataSet().get_dataset(config.dataset_name)
        self.env = CityLearnEnv(
            schema_path,
            central_agent=False,
            reward_function=ComfortReward,
            random_seed=config.seed,
        )
        if self.env.central_agent:
            raise RuntimeError("CityLearn adapter requires central_agent=False")
        building_ids = tuple(b.name for b in self.env.buildings)
        if building_ids != EXPECTED_BUILDING_IDS:
            raise RuntimeError(
                f"CityLearn adapter expects buildings {EXPECTED_BUILDING_IDS}, got {building_ids}"
            )
        if len(self.env.buildings) != self.n_agents:
            raise RuntimeError("CityLearn adapter requires three distributed buildings")
        action_names = self.env.action_names[0]
        if tuple(action_names) != EXPECTED_ACTION_ORDER:
            raise RuntimeError(
                f"CityLearn adapter expects action order {EXPECTED_ACTION_ORDER}, got {tuple(action_names)}"
            )
        obs_names = self.env.observation_names[0]
        if "net_electricity_consumption" not in obs_names:
            raise RuntimeError("CityLearn observations must expose net_electricity_consumption")
        self._obs_dim = int(len(obs_names))
        self._action_dim = int(len(action_names))
        low = np.stack([np.asarray(space.low, dtype=np.float64) for space in self.env.action_space])
        high = np.stack([np.asarray(space.high, dtype=np.float64) for space in self.env.action_space])
        if low.shape != (self.n_agents, self._action_dim) or high.shape != low.shape:
            raise RuntimeError("CityLearn adapter found an unexpected per-agent action-space shape")
        self._action_low = low
        self._action_high = high
        self.steps = 0
        # citylearn_challenge_2023_phase_1's buildings are LSTMDynamicsBuilding:
        # indoor_dry_bulb_temperature (and therefore ComfortReward) is driven by
        # an LSTM that needs `lookback` real post-reset steps of input history
        # before an executed action can change its output at all. Empirically
        # verified directly against citylearn==2.5.0: with a fresh reset, an
        # action of 0.0 vs. 0.9 held for the same buildings produces bit-identical
        # per-step reward through step index `lookback` inclusive, first
        # diverging at `lookback + 1`. Buildings without a dynamics model (no
        # `lookback` attribute) contribute 0.
        lookbacks = [
            int(getattr(getattr(b, "dynamics", None), "lookback", 0) or 0)
            for b in self.env.buildings
        ]
        self._dynamics_warmup_steps = (max(lookbacks) + 1) if lookbacks else 0

    def reset(self):
        obs, _info = self.env.reset()
        self.steps = 0
        obs = np.asarray(obs, dtype=np.float32)
        state = obs.reshape(-1).astype(np.float32)
        return obs, state

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float64).reshape(self.n_agents, self._action_dim)
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        if np.any(actions < self._action_low - 1e-6) or np.any(actions > self._action_high + 1e-6):
            raise ValueError("actions must lie within CityLearn's per-agent action bounds")
        clipped = np.clip(actions, self._action_low, self._action_high)
        # CityLearnEnv.step() executes apply_actions() (writes net electricity
        # consumption into each building's own array at the CURRENT,
        # pre-increment time_step) -> next_time_step() (increments time_step)
        # -> returns self.observations (built AFTER the increment, at the NEW
        # time_step, whose net_electricity_consumption slot has not been
        # written yet). Reading net_electricity_consumption from the returned
        # `obs` therefore reads a one-step-stale (always 0.0) value. The
        # correct value for the transition just completed lives in each
        # building's own native array at the pre-increment index (verified
        # against citylearn==2.5.0).
        pre_step_index = self.steps
        obs, reward, terminated, truncated, info = self.env.step(clipped.tolist())
        if int(self.env.time_step) != pre_step_index + 1:
            raise RuntimeError(
                "CityLearn adapter's own step counter fell out of sync with "
                "env.time_step; the pre-increment net_electricity_consumption "
                "index would be wrong"
            )
        obs = np.asarray(obs, dtype=np.float32)
        state = obs.reshape(-1).astype(np.float32)
        welfare = np.asarray(reward, dtype=np.float32)  # each building's own native ComfortReward
        net_electricity = np.array(
            [float(b.net_electricity_consumption[pre_step_index]) for b in self.env.buildings],
            dtype=np.float64,
        )
        cap_cost, local_cap_costs = attribute_global_cap_excess(
            net_electricity, self.config.cap_kwh_per_step,
        )
        self.steps += 1
        done = bool(terminated or truncated)
        accounting = {
            "C_cap": cap_cost,
            # Diagnostic attribution of the unchanged native district excess
            # to buildings by positive load; not used for training.
            "C_cap_local": local_cap_costs.astype(np.float64),
            "district_net_electricity_kwh": float(net_electricity.sum()),
            "welfare_sum": float(welfare.sum()),
            "full_horizon_step": self.steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }
        return obs, state, welfare, done, accounting

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._obs_dim * self.n_agents

    @property
    def action_low(self) -> np.ndarray:
        return self._action_low

    @property
    def action_high(self) -> np.ndarray:
        return self._action_high

    @property
    def dynamics_warmup_steps(self) -> int:
        """Steps needed after ``reset()`` before an action can affect any
        dynamics-model-driven observation (see ``__init__``'s note)."""
        return self._dynamics_warmup_steps
