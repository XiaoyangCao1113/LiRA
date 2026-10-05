#!/usr/bin/env python3
r"""Held-out MABIM evaluation of a trained checkpoint on the test split.

Rolls out the frozen policy on 2021-09-01..2021-10-30 (60 days) once per
action seed (common random numbers across arms) and writes per-rollout native
profit and per-warehouse rejected inventory to ``--output``.

Example
-------
    python scripts/eval_mabim.py --checkpoint runs/mabim/lira_seed1101/checkpoint.pt \
        --output runs/mabim/lira_seed1101/test_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

TEST_SEEDS = tuple(range(950001, 950009))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default=",".join(map(str, TEST_SEEDS)),
                        help="comma-separated action seeds")
    parser.add_argument("--env-root", type=Path, default=None,
                        help="ReplenishmentEnv checkout (default: $REPLENISHMENT_ENV_ROOT or the installed package)")
    args = parser.parse_args()

    from lira.envs.mabim import evaluation

    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    payload = evaluation.evaluate_test_split(args.checkpoint, args.output, seeds=seeds, env_root=args.env_root)
    print(json.dumps({key: payload[key] for key in
                      ("test_mode", "test_dates", "steps", "seeds",
                       "native_total_profit_mean", "raw_rejection_by_constraint_mean")}, indent=2))


if __name__ == "__main__":
    main()
