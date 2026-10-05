"""MABIM (ReplenishmentEnv) adapter with N=400 SKU agents and K=2 shared costs.

The adapter wraps the ``sku200.2_stores.lowest_capacity`` scenario of
ReplenishmentEnv (https://github.com/VictorYXL/ReplenishmentEnv, commit
``e667565615461ecd4102a60ad1ecd6b772e357d6``).  Each of the 2 x 200
warehouse/SKU pairs is one agent that picks a categorical multiplier
(0.8, 1.0, 1.2) of the official (s, S) = (4, 20) base-stock order.  The two
shared costs are the rejected arriving inventory (``arrived - accepted``) of
warehouse 0 (store2) and warehouse 1 (store1); rewards are the native
per-agent profits.  Rewards and costs are divided by 1e6 for learning.

The environment package is not vendored.  Either install it (``pip install
-e <ReplenishmentEnv checkout>``) or point ``REPLENISHMENT_ENV_ROOT`` (or the
``env_root`` argument / ``--env-root`` CLI flag) at a checkout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ENV_ROOT_VARIABLE = "REPLENISHMENT_ENV_ROOT"
REPLENISHMENT_ENV_COMMIT = "e667565615461ecd4102a60ad1ecd6b772e357d6"
CONFIG_NAME = "sku200.2_stores.lowest_capacity"


def resolve_env_root(env_root: str | os.PathLike | None = None) -> Path | None:
    """Return the explicit checkout path, else ``$REPLENISHMENT_ENV_ROOT``, else None."""
    value = env_root if env_root is not None else os.environ.get(ENV_ROOT_VARIABLE)
    return None if value in (None, "") else Path(value)


def import_replenishment_env(env_root: str | os.PathLike | None = None):
    """Import ``ReplenishmentEnv.make_env`` from a checkout or the installed package."""
    root = resolve_env_root(env_root)
    if root is not None:
        if not root.is_dir():
            raise FileNotFoundError(f"ReplenishmentEnv checkout not found at {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from ReplenishmentEnv import make_env  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the optional install
        raise ImportError(
            "ReplenishmentEnv is required for MABIM; install it or set "
            f"{ENV_ROOT_VARIABLE} to a checkout of commit {REPLENISHMENT_ENV_COMMIT}"
        ) from exc
    return make_env


class MABIMSharedCostEnv:
    """All-agent MABIM environment with K=2 warehouse rejection costs."""

    n_agents = 400
    n_constraints = 2

    def __init__(self, env_root: str | os.PathLike | None = None):
        make_env = import_replenishment_env(env_root)
        # The wrapper is built in ``test`` mode; callers switch the underlying
        # environment's ``mode`` (train/validation/test) before the first reset.
        self.env = make_env(CONFIG_NAME, wrapper_names=["OracleWrapper"], mode="test")
        self.base = self.env.env
        self.steps = 0
        self.action_masks = np.ones((400, 1, 3), dtype=bool)
        self.account = {"raw_profit": 0.0, "raw_cost": [0.0, 0.0], "steps": 0}

    def obs(self):
        m = np.asarray(self.env.get_demand_mean())
        s = np.asarray(self.env.get_in_stock())
        t = np.asarray(self.env.get_in_transit())
        return np.stack([m.reshape(-1) / 100.0, (s + t).reshape(-1) / 5000.0, np.zeros(400), np.zeros(400)], 1)

    def reset(self):
        self.env.reset()
        self.steps = 0
        self.account = {"raw_profit": 0.0, "raw_cost": [0.0, 0.0], "steps": 0}
        return self.obs()

    def step(self, a):
        m = np.asarray(self.env.get_demand_mean())
        s = np.asarray(self.env.get_in_stock())
        t = np.asarray(self.env.get_in_transit())
        q = (s + t) / (m + 1e-4)
        q = np.where(q < 4.0, 20.0 - q, 0.0)
        q = q * np.asarray([0.8, 1.0, 1.2])[np.asarray(a).reshape(2, 200)]
        b = self.base
        b.replenish(q)
        b.sell()
        b.receive_sku()
        p, _ = b.get_reward()
        p = np.asarray(p)
        arr = np.asarray(b.agent_states["all_warehouses", "arrived"])
        acc = np.asarray(b.agent_states["all_warehouses", "accepted"])
        c = np.sum(arr - acc, 1)
        b.balance += p.sum(1)
        b.per_balance += p.flatten()
        b.next_step()
        self.steps += 1
        done = b.current_step >= b.durations
        self.account["raw_profit"] += float(p.sum())
        self.account["raw_cost"] = [self.account["raw_cost"][0] + float(c[0]), self.account["raw_cost"][1] + float(c[1])]
        self.account["steps"] = self.steps
        return (
            (self.obs() if not done else np.zeros((400, 4))),
            torch.tensor(p.reshape(-1), dtype=torch.float64) / 1e6,
            torch.tensor(c, dtype=torch.float64) / 1e6,
            done,
            {},
        )
