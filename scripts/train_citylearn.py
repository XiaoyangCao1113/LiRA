#!/usr/bin/env python3
r"""Train and evaluate Uniform, PAL and LiRA on CityLearn (N=3 buildings, K=1).

Each training seed runs the requested arms one after another (every arm
re-seeds Python/NumPy/Torch with the training seed, so arms are independent
of the order in which they run), evaluates each trained policy on one full
719-step held-out episode (evaluation seed = training seed + 5000), and
writes ``<output-dir>/seed<seed>.json`` with per-cycle training diagnostics
and, per arm, ``results.<arm>.heldout_eval.{welfare_sum,native_shared_cost_sum}``.

Reproduce the paper's CityLearn row of the main results table (seeds
1101-1103; about 8 minutes per seed for all three arms on 4 CPU cores, no
GPU)::

    python scripts/train_citylearn.py --config configs/citylearn.yaml \
        --output-dir runs/citylearn

or a single seed / subset of arms::

    python scripts/train_citylearn.py --config configs/citylearn.yaml \
        --seed 1101 --arms lira --output-dir runs/citylearn

Command-line flags override values from ``--config``; without ``--config``
the parser defaults are the paper's settings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

PAPER_SEEDS = (1101, 1102, 1103)
ARM_CHOICES = ("uniform", "pal", "lira")


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("domain", "citylearn") != "citylearn":
        raise ValueError(f"{path} is not a CityLearn config (domain={config.get('domain')!r})")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="YAML config (configs/citylearn.yaml)")
    parser.add_argument("--output-dir", required=True, help="directory for seed<seed>.json results")
    parser.add_argument(
        "--arms", default="uniform,pal,lira",
        help="comma-separated subset of uniform,pal,lira",
    )
    parser.add_argument("--seed", type=int, default=None, help="run a single training seed")
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in PAPER_SEEDS),
        help="comma-separated training seeds (ignored when --seed is given)",
    )
    parser.add_argument("--dataset", default="citylearn_challenge_2023_phase_1")
    parser.add_argument("--q", type=int, default=10,
                        help="live PPO batches per outer cycle (the paper's h)")
    parser.add_argument("--cycles", type=int, default=30, help="outer (allocation + dual) cycles")
    parser.add_argument("--rollout-steps", type=int, default=12,
                        help="environment steps per PPO batch (live and lookahead)")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per live batch")
    parser.add_argument(
        "--outer-lr", type=float, default=0.05,
        help="fallback step size used for --rho-lr and --lambda-lr when those are not given",
    )
    parser.add_argument("--m-replicates", type=int, default=8,
                        help="LiRA: independent lookahead replicates per outer cycle (M)")
    parser.add_argument("--rho-lr", type=float, default=0.05,
                        help="LiRA: outer Adam step size on the responsibility logits (eta_rho)")
    parser.add_argument("--rho-grad-clip", type=float, default=1.0,
                        help="LiRA: max L2 norm of the outer gradient (None/negative disables)")
    parser.add_argument("--q-meta", type=int, default=2,
                        help="LiRA: differentiable lookahead steps per replicate (q >= 2)")
    parser.add_argument("--terminal-rollout-steps", type=int, default=12,
                        help="LiRA: length of each replicate's terminal rollout")
    parser.add_argument("--lambda-init", type=float, default=2.0,
                        help="initial shared dual (and each PAL per-agent dual)")
    parser.add_argument("--lambda-lr", type=float, default=0.05,
                        help="projected dual-ascent step size (all arms)")
    parser.add_argument(
        "--cost-budget", type=float, default=0.04,
        help="per-step district cap-excess budget d in kWh (dual gradient = mean cost - d)",
    )
    parser.add_argument(
        "--eval-seed", type=int, default=None,
        help="held-out evaluation seed; default training seed + 5000 (paired across arms)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    config = _load_config(known.config)

    parser = build_parser()
    defaults = dict(config.get("args", {}) or {})
    if "seeds" in config:
        defaults["seeds"] = ",".join(str(s) for s in config["seeds"])
    if "arms" in config:
        defaults["arms"] = ",".join(config["arms"])
    unknown = sorted(set(defaults) - {action.dest for action in parser._actions})
    if unknown:
        parser.error(f"unknown keys in {known.config}: {unknown}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    bad = [a for a in arms if a not in ARM_CHOICES]
    if bad or not arms:
        parser.error(f"--arms must be a non-empty subset of {ARM_CHOICES}; got {arms}")
    seeds = [args.seed] if args.seed is not None else [int(s) for s in str(args.seeds).split(",") if s.strip()]
    rho_grad_clip = None if args.rho_grad_clip is None or args.rho_grad_clip < 0 else args.rho_grad_clip

    from lira.envs.citylearn.trainer import run_arm, run_lira_arm

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time()
        results = {}
        for arm in arms:
            if arm == "lira":
                results[arm] = run_lira_arm(
                    dataset_name=args.dataset, seed=seed, q=args.q, cycles=args.cycles,
                    rollout_steps=args.rollout_steps, epochs=args.epochs, outer_lr=args.outer_lr,
                    m_replicates=args.m_replicates, eval_seed=args.eval_seed,
                    rho_lr=args.rho_lr, rho_grad_clip=rho_grad_clip,
                    lambda_init=args.lambda_init, lambda_lr=args.lambda_lr, cost_budget=args.cost_budget,
                    q_meta=args.q_meta, terminal_rollout_steps=args.terminal_rollout_steps,
                )
            else:
                results[arm] = run_arm(
                    arm=arm, dataset_name=args.dataset, seed=seed, q=args.q, cycles=args.cycles,
                    rollout_steps=args.rollout_steps, epochs=args.epochs, outer_lr=args.outer_lr,
                    eval_seed=args.eval_seed,
                    lambda_init=args.lambda_init, lambda_lr=args.lambda_lr, cost_budget=args.cost_budget,
                )
        elapsed = time.time() - t0
        out = {
            "protocol": "CityLearn Uniform / PAL / LiRA training with full-episode held-out evaluation",
            "elapsed_seconds": elapsed,
            "args": {**vars(args), "seed": seed, "arms": list(arms)},
            "results": results,
        }
        out_path = output_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({"out": str(out_path), "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
