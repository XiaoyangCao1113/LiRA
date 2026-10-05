"""Held-out rollout evaluation of frozen MABIM checkpoints.

The frozen shared actor is rolled out on the native ReplenishmentEnv test
split (2021-09-01 .. 2021-10-30, 60 days), which is disjoint from both the
training split used for learning and the validation split used to calibrate
the budgets.  Evaluation contract:

* The actor forward pass reproduces training exactly (observation
  construction, zero resource context, ``h=None`` on every step).
* For each action seed, one array of per-(day, agent) uniforms is drawn with
  ``numpy.random.default_rng(seed)`` and turned into categories by
  inverse-CDF sampling; the same seeds are used for every arm
  (common random numbers across Uniform, PAL and LiRA checkpoints).
* Category 0/1/2 multiplies the official (s, S) = (4, 20) base-stock order by
  0.8/1.0/1.2.
* Welfare is the native total profit; cost ``k`` is the native rejected
  arriving inventory (``arrived - accepted``) of warehouse ``k``.  Every
  native field is read from ``agent_states`` and non-finite values raise.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .env import CONFIG_NAME, REPLENISHMENT_ENV_COMMIT, import_replenishment_env, resolve_env_root
from .learner import (
    SharedStaticCategoricalRunner,
    SharedStaticConfig,
    load_shared_static_checkpoint,
    _CHECKPOINT_SCHEMA,
    _sha256 as _sha256_file,
)

TEST_MODE = "test"
TEST_STEPS = 60
TEST_DATES = ("2021-09-01", "2021-10-30")
TEST_SEEDS = tuple(range(950001, 950009))
WAREHOUSE_CAPACITY = 5000.0
TRAINING_COST_SCALE = 1_000_000.0
CATEGORY_MULTIPLIERS = (0.8, 1.0, 1.2)
SS_BASELINE_S = 20.0
SS_BASELINE_s = 4.0
N_WAREHOUSES = 2
N_SKUS = 200
N_AGENTS = N_WAREHOUSES * N_SKUS
N_CONSTRAINTS = 2
# Index 0 = store2, index 1 = store1, matching the config's warehouse list;
# constraint k is warehouse k's rejected inventory.
WAREHOUSE_NAMES = ("store2", "store1")
# "P" is the learned policy; the others are reference/intervention arms.
ARM_NAMES = ("P", "Random", "sS", "W0-low", "W0-high", "W1-low", "W1-high")


def _require_finite(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise RuntimeError(f"native field {name!r} is empty")
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"native field {name!r} contains non-finite values")
    return array


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return None


def make_heldout_env(*, env_root=None, mode: str = TEST_MODE):
    """Construct the native MABIM environment on the requested data split.

    If ``env_root`` is a git checkout, its HEAD must match the pinned
    ReplenishmentEnv commit; an installed package cannot be checked and only
    triggers a warning.
    """
    root_path = resolve_env_root(env_root)
    actual_commit = None if root_path is None else _git_commit(root_path)
    if actual_commit is None:
        warnings.warn(
            f"cannot verify the ReplenishmentEnv commit (expected {REPLENISHMENT_ENV_COMMIT})",
            stacklevel=2,
        )
    elif actual_commit != REPLENISHMENT_ENV_COMMIT:
        raise RuntimeError(
            f"ReplenishmentEnv git HEAD mismatch: expected {REPLENISHMENT_ENV_COMMIT!r}, "
            f"got {actual_commit!r} at {root_path}"
        )

    make_env = import_replenishment_env(env_root)
    env = make_env(CONFIG_NAME, wrapper_names=["OracleWrapper"], mode=mode)
    base = env.env

    assert base.warehouse_list == list(WAREHOUSE_NAMES), (
        f"warehouse_list order mismatch: expected {list(WAREHOUSE_NAMES)}, got {base.warehouse_list}"
    )

    for wh_config in base.config["warehouse"]:
        wh_name = wh_config["name"]
        wh_capacity = wh_config["capacity"]
        assert wh_capacity == WAREHOUSE_CAPACITY, (
            f"warehouse {wh_name!r} capacity mismatch: expected {WAREHOUSE_CAPACITY}, got {wh_capacity}"
        )

    return env, base


@dataclass
class CheckpointMeta:
    checkpoint_path: str
    checkpoint_sha256: str
    config: dict[str, Any]
    counters: dict[str, int]
    source_digests: dict[str, str]


def load_frozen_actor(
    checkpoint_path: str | Path,
    *,
    expected_source_digests: Mapping[str, str] | None = None,
):
    """Strictly restore the frozen actor from a MABIM shared-static checkpoint.

    Read-only: only ever reads ``checkpoint_path``; never writes to it, never
    calls ``.update()``/``.backward()``/optimizer ``.step()``, and freezes
    every actor parameter (``requires_grad_(False)``) as defense in depth.

    ``expected_source_digests`` defaults to the digests recorded in the
    checkpoint itself (the checkpoint loader requires an exact match).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint_sha256_before = _sha256_file(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or raw.get("schema") != _CHECKPOINT_SCHEMA:
        raise RuntimeError(f"not a valid MABIM shared-static checkpoint: {path}")

    cfg_dict = raw["config"]
    config = SharedStaticConfig(**cfg_dict)
    if config.n_warehouses != N_WAREHOUSES or config.n_skus != N_SKUS:
        raise NotImplementedError(
            "this evaluator's warehouse/sku agent-id convention is pinned to "
            f"n_warehouses={N_WAREHOUSES}, n_skus={N_SKUS}; got "
            f"n_warehouses={config.n_warehouses}, n_skus={config.n_skus}"
        )
    if config.device != "cpu":
        raise RuntimeError("this evaluator is CPU-only")

    if expected_source_digests is None:
        expected = dict(raw["source_digests"])
    else:
        expected = dict(expected_source_digests)

    warehouse_ids = [0] * config.n_skus + [1] * config.n_skus
    sku_ids = list(range(config.n_skus)) * config.n_warehouses
    runner = SharedStaticCategoricalRunner(
        None, config, raw["rho"], raw["support_mask"], warehouse_ids, sku_ids
    )
    counters = load_shared_static_checkpoint(runner, path, expected_source_digests=expected)

    checkpoint_sha256_after = _sha256_file(path)
    if checkpoint_sha256_before != checkpoint_sha256_after:
        raise RuntimeError("checkpoint file mutated during load -- refusing to proceed")

    for parameter in runner.actor.parameters():
        parameter.requires_grad_(False)
    runner.actor.eval()

    meta = CheckpointMeta(
        checkpoint_path=str(path),
        checkpoint_sha256=checkpoint_sha256_after,
        config=cfg_dict,
        counters=dict(counters),
        source_digests=dict(raw["source_digests"]),
    )
    return runner.actor, meta


def build_warehouse_sku_ids() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    warehouse_ids = torch.as_tensor([0] * N_SKUS + [1] * N_SKUS, dtype=torch.long)
    sku_ids = torch.as_tensor(list(range(N_SKUS)) * N_WAREHOUSES, dtype=torch.long)
    warehouse_of_agent = np.asarray([0] * N_SKUS + [1] * N_SKUS, dtype=np.int64)
    return warehouse_ids, sku_ids, warehouse_of_agent


def build_observation(env) -> np.ndarray:
    """Reproduce the exact 4-d observation the training adapter feeds the actor.

    Native getters return ``[n_warehouses, n_skus]``; flattened row-major
    (warehouse-major) to match the pinned agent-id convention
    ``[0]*n_skus + [1]*n_skus``, exactly as ``MABIMSharedCostEnv.obs()`` does.
    """
    demand_mean = _require_finite(np.asarray(env.get_demand_mean()).reshape(-1), "demand_mean")
    in_stock = _require_finite(np.asarray(env.get_in_stock()).reshape(-1), "in_stock")
    in_transit = _require_finite(np.asarray(env.get_in_transit()).reshape(-1), "in_transit")
    if demand_mean.shape != (N_AGENTS,) or in_stock.shape != (N_AGENTS,) or in_transit.shape != (N_AGENTS,):
        raise RuntimeError("unexpected native observation shape from MABIM env")
    zeros = np.zeros(N_AGENTS, dtype=np.float64)
    obs = np.stack([demand_mean / 100.0, (in_stock + in_transit) / 5000.0, zeros, zeros], axis=1)
    return obs


def ss_baseline_action(env) -> tuple[np.ndarray, np.ndarray]:
    """Official sS_policy inventory-position ratio (S=20, s=4), pre-multiplier.

    Returns ``(q_base, demand_mean)`` both shape ``(N_AGENTS,)``; the caller
    multiplies ``q_base`` by the category multiplier and lets
    ``ReplenishmentEnv.replenish`` apply the ``demand_mean_continuous``
    conversion (which multiplies by ``demand_mean`` again internally --
    this evaluator must not double-apply that conversion itself).
    """
    demand_mean = _require_finite(np.asarray(env.get_demand_mean()).reshape(-1), "demand_mean")
    in_stock = _require_finite(np.asarray(env.get_in_stock()).reshape(-1), "in_stock")
    in_transit = _require_finite(np.asarray(env.get_in_transit()).reshape(-1), "in_transit")
    position_ratio = (in_stock + in_transit) / (demand_mean + 1e-4)
    q_base = np.where(position_ratio < SS_BASELINE_s, SS_BASELINE_S - position_ratio, 0.0)
    return q_base, demand_mean


def actor_category_probs(actor, obs: np.ndarray, warehouse_ids: torch.Tensor, sku_ids: torch.Tensor) -> np.ndarray:
    """Frozen-actor forward pass for one step; ``h`` is never carried across steps."""
    zero_context = torch.zeros((1, N_CONSTRAINTS), dtype=torch.float64)
    obs_t = torch.as_tensor(obs, dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        logits, _ = actor(obs_t, warehouse_ids, sku_ids, zero_context)  # h=None every call
    probs = torch.softmax(logits[0], dim=-1).numpy()
    return probs


def category_probs_for_arm(
    arm: str, *, p_probs: np.ndarray, warehouse_of_agent: np.ndarray,
) -> np.ndarray:
    """Per-agent category-3 probability rows for the named arm."""
    if arm == "P":
        return p_probs
    if arm == "Random":
        return np.full((N_AGENTS, 3), 1.0 / 3.0, dtype=np.float64)
    if arm == "sS":
        onehot = np.zeros((N_AGENTS, 3), dtype=np.float64)
        onehot[:, 1] = 1.0
        return onehot

    forced = {
        "W0-low": (0, 0), "W0-high": (0, 2),
        "W1-low": (1, 0), "W1-high": (1, 2),
    }.get(arm)
    if forced is None:
        raise ValueError(f"unknown arm {arm!r}")
    forced_warehouse, forced_category = forced
    probs = p_probs.copy()
    mask = warehouse_of_agent == forced_warehouse
    probs[mask] = 0.0
    probs[mask, forced_category] = 1.0
    return probs


def inverse_cdf_categories(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Turn shared per-agent uniforms into categories via inverse-CDF."""
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise RuntimeError(f"probs must be [n,3], got {probs.shape}")
    if u.shape != (probs.shape[0],):
        raise RuntimeError(f"u must be [{probs.shape[0]}], got {u.shape}")
    if not np.all(np.isfinite(probs)) or np.any(probs < -1e-9):
        raise RuntimeError("category probabilities must be finite and nonnegative")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise RuntimeError("category probability rows must sum to 1")
    cumulative = np.cumsum(probs, axis=1)
    categories = (u[:, None] >= cumulative[:, :-1]).sum(axis=1)
    return categories.astype(np.int64)


@dataclass
class StepRecord:
    date: str
    actions: list[int]
    realized_order_by_warehouse: list[float]
    order_change_by_warehouse: list[float]
    cost_by_warehouse: list[float]
    total_profit_step: float
    per_agent_profit_step: list[float]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "actions": self.actions,
            "realized_order_by_warehouse": self.realized_order_by_warehouse,
            "order_change_by_warehouse": self.order_change_by_warehouse,
            "cost_by_warehouse": self.cost_by_warehouse,
            "total_profit_step": self.total_profit_step,
            "per_agent_profit_step": self.per_agent_profit_step,
        }


@dataclass
class RolloutResult:
    arm: str
    action_seed: int
    checkpoint: CheckpointMeta
    env_mode: str
    env_start_date: str
    env_end_date: str
    env_steps: int
    terminated: bool
    native_total_profit: float
    per_agent_total_profit: list[float]
    raw_rejection_by_constraint: list[float]
    capacity_normalized_cost_by_constraint: list[float]
    training_scaled_cost_by_constraint: list[float]
    warehouse_totals: dict[str, dict[str, float]]
    fill_rate: float
    lost_sales: float
    step_records: list[StepRecord]
    evaluator_git_commit: str | None
    evaluator_source_sha256: str
    config: dict[str, Any]
    command: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "action_seed": self.action_seed,
            "checkpoint": {
                "checkpoint_path": self.checkpoint.checkpoint_path,
                "checkpoint_sha256": self.checkpoint.checkpoint_sha256,
                "config": self.checkpoint.config,
                "counters": self.checkpoint.counters,
                "source_digests": self.checkpoint.source_digests,
            },
            "env_mode": self.env_mode,
            "env_start_date": self.env_start_date,
            "env_end_date": self.env_end_date,
            "env_steps": self.env_steps,
            "terminated": self.terminated,
            "native_total_profit": self.native_total_profit,
            "per_agent_total_profit": self.per_agent_total_profit,
            "raw_rejection_by_constraint": self.raw_rejection_by_constraint,
            "capacity_normalized_cost_by_constraint": self.capacity_normalized_cost_by_constraint,
            "training_scaled_cost_by_constraint": self.training_scaled_cost_by_constraint,
            "warehouse_totals": self.warehouse_totals,
            "fill_rate": self.fill_rate,
            "lost_sales": self.lost_sales,
            "step_records": [record.to_json_dict() for record in self.step_records],
            "evaluator_git_commit": self.evaluator_git_commit,
            "evaluator_source_sha256": self.evaluator_source_sha256,
            "config": self.config,
            "command": self.command,
        }


def _evaluator_source_sha256() -> str:
    return _sha256_file(Path(__file__))


def _build_provenance_config(mode: str, env_root) -> dict[str, Any]:
    root = resolve_env_root(env_root)
    return {
        "config_name": CONFIG_NAME,
        "mode": mode,
        "mabim_source_commit": REPLENISHMENT_ENV_COMMIT,
        "replenishment_env_root": None if root is None else str(root),
        "capacity_per_warehouse": WAREHOUSE_CAPACITY,
        "training_cost_scale": TRAINING_COST_SCALE,
        "category_multipliers": list(CATEGORY_MULTIPLIERS),
        "sS_baseline_S": SS_BASELINE_S,
        "sS_baseline_s": SS_BASELINE_s,
        "warehouse_names": list(WAREHOUSE_NAMES),
    }


def normalize_costs(raw_rejection: Sequence[float], n_steps: int) -> tuple[list[float], list[float]]:
    """Raw K=2 rejection -> (capacity-normalized, training-scaled) costs.

    capacity-normalized: Ck / (WAREHOUSE_CAPACITY * n_steps)
    training-scaled:     Ck / TRAINING_COST_SCALE  (matches the training adapter's /1e6)
    """
    raw = np.asarray(raw_rejection, dtype=np.float64)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    capacity_normalized = (raw / (WAREHOUSE_CAPACITY * n_steps)).tolist()
    training_scaled = (raw / TRAINING_COST_SCALE).tolist()
    return capacity_normalized, training_scaled


def run_single_rollout(
    actor,
    checkpoint_meta: CheckpointMeta,
    arm: str,
    seed: int,
    *,
    env_root=None,
    mode: str = TEST_MODE,
    n_steps: int = TEST_STEPS,
) -> RolloutResult:
    """Run one full native-window MABIM rollout for one arm/action seed."""
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARM_NAMES}")

    env, base = make_heldout_env(env_root=env_root, mode=mode)
    env.reset()
    if base.mode != mode:
        raise RuntimeError(f"env constructed with mode={base.mode!r}, expected {mode!r}")
    if base.durations != n_steps:
        raise RuntimeError(f"{mode} window has durations={base.durations}, expected {n_steps}")

    warehouse_ids, sku_ids, warehouse_of_agent = build_warehouse_sku_ids()
    rng = np.random.default_rng(seed)
    uniforms = rng.random((n_steps, N_AGENTS))

    total_profit = 0.0
    per_agent_profit_total = np.zeros(N_AGENTS, dtype=np.float64)
    raw_rejection = np.zeros(N_CONSTRAINTS, dtype=np.float64)
    warehouse_arrived = np.zeros(N_WAREHOUSES, dtype=np.float64)
    warehouse_accepted = np.zeros(N_WAREHOUSES, dtype=np.float64)
    warehouse_orders = np.zeros(N_WAREHOUSES, dtype=np.float64)
    warehouse_inventory = np.zeros(N_WAREHOUSES, dtype=np.float64)
    warehouse_demand = np.zeros(N_WAREHOUSES, dtype=np.float64)
    warehouse_sale = np.zeros(N_WAREHOUSES, dtype=np.float64)
    step_records: list[StepRecord] = []

    start_date = base.picked_start_date
    steps_run = 0
    for step in range(n_steps):
        obs = build_observation(env)
        p_probs = actor_category_probs(actor, obs, warehouse_ids, sku_ids)
        arm_probs = category_probs_for_arm(arm, p_probs=p_probs, warehouse_of_agent=warehouse_of_agent)
        categories = inverse_cdf_categories(arm_probs, uniforms[step])

        q_base, demand_mean = ss_baseline_action(env)
        multipliers = np.asarray(CATEGORY_MULTIPLIERS, dtype=np.float64)[categories]
        action_flat = q_base * multipliers
        neutral_flat = q_base * CATEGORY_MULTIPLIERS[1]
        action = action_flat.reshape(N_WAREHOUSES, N_SKUS)

        base.replenish(action)
        base.sell()
        base.receive_sku()
        profit, _reward_info = base.get_reward()
        profit = _require_finite(profit, "profit")
        if profit.shape != (N_WAREHOUSES, N_SKUS):
            raise RuntimeError(f"unexpected native profit shape {profit.shape}")
        base.balance = base.balance + profit.sum(axis=1)
        base.per_balance = base.per_balance + profit.flatten()

        arrived = _require_finite(base.agent_states["all_warehouses", "arrived"], "arrived")
        accepted = _require_finite(base.agent_states["all_warehouses", "accepted"], "accepted")
        demand = _require_finite(base.agent_states["all_warehouses", "demand"], "demand")
        sale = _require_finite(base.agent_states["all_warehouses", "sale"], "sale")
        in_stock = _require_finite(base.agent_states["all_warehouses", "in_stock"], "in_stock")
        realized_order = _require_finite(base.agent_states["all_warehouses", "replenish"], "replenish")
        for name, value in (("arrived", arrived), ("accepted", accepted), ("demand", demand),
                            ("sale", sale), ("in_stock", in_stock), ("replenish", realized_order)):
            if value.shape != (N_WAREHOUSES, N_SKUS):
                raise RuntimeError(f"unexpected native shape for {name}: {value.shape}")

        cost_step = (arrived - accepted).sum(axis=1)
        realized_order_by_wh = realized_order.sum(axis=1)
        neutral_order = neutral_flat.reshape(N_WAREHOUSES, N_SKUS) * demand_mean.reshape(N_WAREHOUSES, N_SKUS)
        if getattr(base, "integerization_sku", False):
            # ReplenishmentEnv.replenish() floors realized orders when this
            # config flag is set; replicate it so the category-1 (sS) arm's
            # "change" is exactly zero rather than an artifact of comparing
            # a floored realized order to an unfloored counterfactual.
            neutral_order = np.floor(neutral_order)
        neutral_order_by_wh = neutral_order.sum(axis=1)
        order_change_by_wh = realized_order_by_wh - neutral_order_by_wh

        total_profit += float(profit.sum())
        per_agent_profit_total += profit.flatten()
        raw_rejection += cost_step
        warehouse_arrived += arrived.sum(axis=1)
        warehouse_accepted += accepted.sum(axis=1)
        warehouse_orders += realized_order_by_wh
        warehouse_inventory += in_stock.sum(axis=1)
        warehouse_demand += demand.sum(axis=1)
        warehouse_sale += sale.sum(axis=1)

        step_records.append(StepRecord(
            date=(start_date + timedelta(days=step)).strftime("%Y-%m-%d"),
            actions=[int(c) for c in categories],
            realized_order_by_warehouse=[float(v) for v in realized_order_by_wh],
            order_change_by_warehouse=[float(v) for v in order_change_by_wh],
            cost_by_warehouse=[float(v) for v in cost_step],
            total_profit_step=float(profit.sum()),
            per_agent_profit_step=[float(v) for v in profit.flatten()],
        ))

        base.next_step()
        steps_run += 1
        if base.current_step >= base.durations:
            break

    terminated = base.current_step >= base.durations
    if steps_run != n_steps or not terminated:
        raise RuntimeError(
            f"rollout did not complete the full native {mode} window: "
            f"steps_run={steps_run}, expected={n_steps}, terminated={terminated}"
        )

    if warehouse_demand.sum() <= 0:
        raise RuntimeError("native demand totals are non-positive; refusing to report fill_rate")
    fill_rate = float(warehouse_sale.sum() / warehouse_demand.sum())
    lost_sales = float((warehouse_demand - warehouse_sale).sum())

    warehouse_totals = {
        name: {
            "arrived": float(warehouse_arrived[i]),
            "accepted": float(warehouse_accepted[i]),
            "orders": float(warehouse_orders[i]),
            "inventory_mean": float(warehouse_inventory[i] / n_steps),
            "demand": float(warehouse_demand[i]),
            "sale": float(warehouse_sale[i]),
        }
        for i, name in enumerate(WAREHOUSE_NAMES)
    }

    capacity_normalized, training_scaled = normalize_costs(raw_rejection, n_steps)

    return RolloutResult(
        arm=arm,
        action_seed=seed,
        checkpoint=checkpoint_meta,
        env_mode=base.mode,
        env_start_date=base.picked_start_date.strftime("%Y-%m-%d"),
        env_end_date=base.picked_end_date.strftime("%Y-%m-%d"),
        env_steps=n_steps,
        terminated=terminated,
        native_total_profit=total_profit,
        per_agent_total_profit=[float(v) for v in per_agent_profit_total],
        raw_rejection_by_constraint=[float(v) for v in raw_rejection],
        capacity_normalized_cost_by_constraint=[float(v) for v in capacity_normalized],
        training_scaled_cost_by_constraint=[float(v) for v in training_scaled],
        warehouse_totals=warehouse_totals,
        fill_rate=fill_rate,
        lost_sales=lost_sales,
        step_records=step_records,
        evaluator_git_commit=_git_commit(Path(__file__).resolve().parents[2]),
        evaluator_source_sha256=_evaluator_source_sha256(),
        config=_build_provenance_config(mode, env_root),
        command=" ".join(sys.argv),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_test_split(
    checkpoint: str | Path,
    output: str | Path,
    *,
    seeds: Sequence[int] = TEST_SEEDS,
    env_root=None,
) -> dict[str, Any]:
    """Evaluate the learned policy ("P") on the test split for each action seed."""
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if raw.get("counters", {}).get("steps", 0) < 3000:
        raise RuntimeError("test evaluation expects a policy trained from the 3000-step initial checkpoint")
    actor, checkpoint_meta = load_frozen_actor(
        checkpoint, expected_source_digests=raw["source_digests"],
    )
    seeds = tuple(int(value) for value in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("test action seeds must be nonempty and unique")
    rows = []
    for seed in seeds:
        result = run_single_rollout(
            actor, checkpoint_meta, "P", seed, env_root=env_root, mode=TEST_MODE, n_steps=TEST_STEPS,
        )
        row = result.to_json_dict()
        if row["env_mode"] != TEST_MODE or row["env_steps"] != TEST_STEPS:
            raise RuntimeError("test split or episode length contract failed")
        if (row["env_start_date"], row["env_end_date"]) != TEST_DATES:
            raise RuntimeError("unexpected test date window")
        rows.append(row)

    welfare = np.asarray([row["native_total_profit"] for row in rows], dtype=np.float64)
    costs = np.asarray([row["raw_rejection_by_constraint"] for row in rows], dtype=np.float64)
    payload = {
        "schema": "mabim-test-split-eval-v1",
        "data_protocol": "train split -> validation-calibrated budgets -> test split",
        "test_mode": TEST_MODE,
        "test_dates": list(TEST_DATES),
        "steps": TEST_STEPS,
        "seeds": list(seeds),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_digests": raw["source_digests"],
        "native_total_profit_mean": float(welfare.mean()),
        "native_total_profit_sd_population": float(welfare.std(ddof=0)),
        "raw_rejection_by_constraint_mean": costs.mean(0).tolist(),
        "raw_rejection_by_constraint_sd_population": costs.std(0, ddof=0).tolist(),
        "rollouts": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
