#!/usr/bin/env python3
"""Train Uniform / PAL / LiRA on MABIM (N=400 SKU agents, K=2 shared costs).

Each (arm, seed) cell writes to ``<output-dir>/<arm>_seed<seed>/``:
``checkpoint.pt`` (final learner), ``checkpoint.json`` (training log),
``checkpoint.resume.pt`` (per-cycle resume state) and, with ``--evaluate``,
``test_eval.json`` (held-out test-split rollouts).  An interrupted cell is
resumed automatically from its resume checkpoint.

Examples
--------
    # all arms and seeds of the main table, one after another
    python scripts/train_mabim.py --config configs/mabim.yaml --evaluate
    # one cell
    python scripts/train_mabim.py --config configs/mabim.yaml --arm lira --seed 1101 --evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

ARMS = ("uniform", "pal", "lira", "lira_frozen_rho")
INIT_CHECKPOINT_URL = "https://github.com/XiaoyangCao1113/LiRA/releases/download/v1.0/mabim_init_policy.pt"


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    import yaml

    with Path(path).open() as stream:
        return yaml.safe_load(stream) or {}


def build_parser(config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="YAML config (e.g. configs/mabim.yaml)")
    parser.add_argument("--arm", choices=ARMS, default=None,
                        help="arm to train (default: every arm listed in the config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="training seed (default: every seed listed in the config)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mabim"))
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help=f"initial policy checkpoint (download: {INIT_CHECKPOINT_URL})")
    parser.add_argument("--budget-file", type=Path, default=None,
                        help="JSON with a 'budgets' list (per-episode shared-cost budgets in 1e6 units)")
    parser.add_argument("--env-root", type=Path, default=None,
                        help="ReplenishmentEnv checkout (default: $REPLENISHMENT_ENV_ROOT or the installed package)")
    parser.add_argument("--training-mode", choices=("train", "test"), default="train")
    parser.add_argument("--eta", type=float, default=0.01, help="shared-dual step size")
    parser.add_argument("--q", type=int, default=5, help="committed updates per cycle and lookahead length")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--outer-lr", type=float, default=0.05, help="responsibility-logit step size")
    parser.add_argument("--outer-objective-scale", type=float, default=None,
                        help="scale of the LiRA outer gradient (default: --lira-outer-objective-scale for "
                             "LiRA arms, 1 otherwise)")
    parser.add_argument("--lira-outer-objective-scale", type=float, default=400.0)
    parser.add_argument("--m-replicates", type=int, default=6)
    parser.add_argument("--dual-init", type=float, default=0.01)
    parser.add_argument("--dual-max", type=float, default=100.0)
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="write a resume checkpoint every N outer cycles (0 disables)")
    parser.add_argument("--evaluate", action="store_true",
                        help="run the held-out test-split evaluation after training")
    known = {action.dest for action in parser._actions}
    defaults = dict(config.get("args", {}))
    unknown = sorted(set(defaults) - known)
    if unknown:
        raise SystemExit(f"unknown keys in config args: {unknown}")
    parser.set_defaults(**defaults)
    return parser


def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    config = _load_config(pre.parse_known_args(argv)[0].config)
    args = build_parser(config).parse_args(argv)

    arms = [args.arm] if args.arm else list(config.get("arms", ("uniform", "pal", "lira")))
    seeds = [args.seed] if args.seed is not None else list(config.get("seeds", ()))
    if not seeds:
        raise SystemExit("pass --seed or a config listing seeds")
    if args.init_checkpoint is None or not Path(args.init_checkpoint).exists():
        raise SystemExit(f"initial checkpoint not found: {args.init_checkpoint}; download it from "
                         f"{INIT_CHECKPOINT_URL}")
    if args.budget_file is None or not Path(args.budget_file).exists():
        raise SystemExit(f"budget file not found: {args.budget_file}")
    eval_seeds = tuple(config.get("eval", {}).get("seeds", ())) or None

    from lira.envs.mabim import trainer

    for arm in arms:
        for seed in seeds:
            is_lira = arm in ("lira", "lira_frozen_rho")
            scale = args.outer_objective_scale
            if scale is None:
                scale = args.lira_outer_objective_scale if is_lira else 1.0
            cell = args.output_dir / f"{arm}_seed{seed}"
            final = cell / "checkpoint.pt"
            if (cell / "checkpoint.json").exists():
                print(f"[skip] {cell} already has a completed run")
            else:
                cell.mkdir(parents=True, exist_ok=True)
                resume = cell / "checkpoint.resume.pt"
                print(f"[train] arm={arm} seed={seed} -> {cell}")
                trainer.run(
                    checkpoint=Path(args.init_checkpoint), budget_file=Path(args.budget_file), output=final,
                    arm=arm, seed=seed, eta=args.eta, q=args.q, cycles=args.cycles, outer_lr=args.outer_lr,
                    outer_objective_scale=scale, m_replicates=args.m_replicates,
                    dual_init=args.dual_init, dual_max=args.dual_max,
                    checkpoint_every=args.checkpoint_every,
                    resume_from=resume if resume.exists() and resume.stat().st_size > 0 else None,
                    training_mode=args.training_mode, env_root=args.env_root,
                )
            if args.evaluate and not (cell / "test_eval.json").exists():
                from lira.envs.mabim import evaluation

                kwargs = {"seeds": eval_seeds} if eval_seeds else {}
                payload = evaluation.evaluate_test_split(final, cell / "test_eval.json",
                                                         env_root=args.env_root, **kwargs)
                print(json.dumps({key: payload[key] for key in (
                    "native_total_profit_mean", "raw_rejection_by_constraint_mean")}))


if __name__ == "__main__":
    main()
