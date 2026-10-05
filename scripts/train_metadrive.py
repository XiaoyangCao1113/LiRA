#!/usr/bin/env python3
"""Train and evaluate the Uniform / PAL / LiRA arms on MetaDrive Intersection (N=4, K=1).

For every requested (arm, seed) cell this script builds the official
``MultiAgentIntersectionEnv`` (horizon 1000), warm-starts the actors and the
critic from a shared initial checkpoint, runs ``--steps`` online learner
steps, and then evaluates the final policy on held-out episode seeds with
deterministic mean actions. One JSON file is written per invocation with the
per-cell training trajectories (loss, dual, rho) and held-out episodes.

Reproduce one paper seed (all three arms)::

    python scripts/train_metadrive.py --config configs/metadrive.yaml \\
        --seeds 211 --out outputs/metadrive/seed211/result.json

Seed ranges: with ``--steps 100`` the training loop of a cell with base seed
``s`` consumes environment seeds ``s .. s+99``; the LiRA lookahead tapes use
seeds up to ``s + steps + q + (M-1) * sc_seed_stride``. The official env
requires ``start_seed <= seed < start_seed + num_scenarios``, hence
``--num-scenarios 512``. All training seeds are >= the lowest base seed (211),
so held-out seeds below 211 are never used for training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import lira  # noqa: F401
except ImportError:  # allow running from a source checkout without installation
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lira.envs.metadrive.env import MetaDriveIntersectionConfig, MetaDriveIntersectionEnv  # noqa: E402
from lira.envs.metadrive.learner import MetaDriveLearnerConfig  # noqa: E402
from lira.envs.metadrive.online import MetaDriveOnlineConfig, MetaDriveOnlineLearner, heldout_evaluate  # noqa: E402

DEFAULT_NUM_SCENARIOS = 512
ARM_ALIASES = {"du": "lira"}  # "du" (direct unroll) is accepted as an alias of LiRA
CHECKPOINT_URL = "https://github.com/XiaoyangCao1113/LiRA/releases/download/v1.0/metadrive_init_actor.pt"


def load_actor_critic_only(online: MetaDriveOnlineLearner, checkpoint_path: str) -> None:
    """Warm-start only the actor/critic weights from a full learner checkpoint.

    Optimizer state, rho/shared lambda, PAL lambdas, and RNG state are left
    untouched: every arm starts from the same policy but runs its own
    allocator dynamics from its own neutral start.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    learner = online.learner
    for actor, actor_state in zip(learner.actors, checkpoint["actors"]):
        actor.load_state_dict(actor_state)
    learner.critic.load_state_dict(checkpoint["critic"])


def run_cell(
    arm: str,
    *,
    seed: int,
    steps: int,
    rho_interval: int,
    du_q: int,
    rho_lr: float,
    cost_budget: float,
    lambda_lr: float,
    num_scenarios: int,
    heldout_seeds: tuple[int, ...],
    heldout_horizon: int,
    actor_lr: float = 3e-4,
    entropy_coefficient: float = 0.0,
    init_checkpoint: str | None = None,
    sampling_correction: bool = False,
    sc_baseline: str = "zero",
    sc_replicates: int = 1,
    sc_seed_stride: int = 17,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    env = MetaDriveIntersectionEnv(
        MetaDriveIntersectionConfig(n_agents=4, horizon=1000, num_scenarios=num_scenarios)
    )
    env.reset(seed=seed)
    learner_config = MetaDriveLearnerConfig(
        n_agents=4, n_constraints=1, obs_dim=env.obs_dim, action_dim=env.action_dim, horizon=1000,
        actor_lr=actor_lr, entropy_coefficient=entropy_coefficient,
    )
    online_config = MetaDriveOnlineConfig(
        arm=arm, cost_budget=cost_budget, lambda_lr=lambda_lr, rho_lr=rho_lr,
        rho_interval=rho_interval, du_q=du_q,
        sampling_correction=sampling_correction, sc_baseline=sc_baseline,
        sc_replicates=sc_replicates, sc_seed_stride=sc_seed_stride,
    )
    online = MetaDriveOnlineLearner(env, learner_config, online_config)
    if init_checkpoint is not None:
        load_actor_critic_only(online, init_checkpoint)

    losses, dual_trajectory, rho_trajectory = [], [], []
    du_gradient_norms, du_direct_gradient_norms, du_sc_gradient_norms = [], [], []
    t0 = time.monotonic()
    try:
        for i in range(steps):
            # Small offsets from ``seed`` keep the derived lookahead tape seeds
            # (see MetaDriveOnlineLearner._derive_du_seeds) inside the env's
            # [start_seed, start_seed + num_scenarios) range.
            result = online.step(seed=seed + i)
            losses.append(result["loss"])
            dual_trajectory.append(result["pal_lambdas"] if arm == "pal" else result["shared_lambda"])
            rho_trajectory.append(result["rho"])
            du_gradient_norms.append(result.get("du_gradient_norm"))
            du_direct_gradient_norms.append(result.get("du_direct_gradient_norm"))
            du_sc_gradient_norms.append(result.get("du_sc_gradient_norm"))
        heldout = heldout_evaluate(online.learner, env, heldout_seeds, heldout_horizon)
    finally:
        env.close()
    elapsed = time.monotonic() - t0

    all_finite = bool(np.isfinite(losses).all())
    return {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "rho_lr": rho_lr,
        "elapsed_seconds": elapsed,
        "losses": losses,
        "final_loss": losses[-1],
        "all_losses_finite": all_finite,
        "dual_trajectory": dual_trajectory,
        "final_dual": dual_trajectory[-1],
        "rho_trajectory": rho_trajectory,
        "final_rho": rho_trajectory[-1],
        "du_gradient_norms": du_gradient_norms,
        "du_direct_gradient_norms": du_direct_gradient_norms,
        "du_sc_gradient_norms": du_sc_gradient_norms,
        "heldout": heldout,
    }


def _parse_arms(text: str) -> list[str]:
    arms = [ARM_ALIASES.get(a.strip(), a.strip()) for a in text.split(",") if a.strip()]
    unknown = sorted(set(arms) - {"uniform", "lira", "pal"})
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from uniform, lira, pal")
    return arms


def _parse_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Matched arms x seeds at ``args.rho_lr``."""
    arms = _parse_arms(args.arms)
    seeds = [int(s) for s in str(args.seeds).split(",")]
    return [{"arm": arm, "seed": seed, "rho_lr": args.rho_lr} for arm in arms for seed in seeds]


def _as_csv(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None, help="YAML config (e.g. configs/metadrive.yaml); CLI flags override it.")
    parser.add_argument("--arms", default="uniform,lira,pal", help="Comma-separated subset of uniform,lira,pal.")
    parser.add_argument("--seeds", default="211", help="Comma-separated training seeds.")
    parser.add_argument("--steps", type=int, default=100, help="Online learner steps (one rollout + PPO update each).")
    parser.add_argument("--rho-interval", type=int, default=5, help="LiRA: responsibility update every this many steps.")
    parser.add_argument("--q", "--du-q", dest="du_q", type=int, default=2, help="LiRA: lookahead depth q.")
    parser.add_argument("--rho-lr", type=float, default=0.02, help="LiRA: responsibility step size.")
    parser.add_argument("--cost-budget", type=float, default=0.1, help="Shared cost budget per actual env step.")
    parser.add_argument("--lambda-lr", type=float, default=0.05, help="Dual learning rate (shared for all arms).")
    parser.add_argument("--num-scenarios", type=int, default=DEFAULT_NUM_SCENARIOS)
    parser.add_argument("--heldout-seeds", default="100,130,160", help="Comma-separated held-out episode seeds.")
    parser.add_argument("--heldout-horizon", type=int, default=1000)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--sampling-correction", action=argparse.BooleanOptionalAction, default=False,
                        help="LiRA: add the score-function sampling correction (SC) to the DU gradient.")
    parser.add_argument("--sc-baseline", choices=("zero", "loo"), default="zero")
    parser.add_argument("--sc-replicates", type=int, default=1, help="LiRA: number M of independent lookaheads.")
    parser.add_argument("--sc-seed-stride", type=int, default=17, help="LiRA: seed offset between lookahead replicates.")
    parser.add_argument(
        "--init-checkpoint", default=None,
        help=f"Initial learner checkpoint; only actor+critic weights are loaded (download: {CHECKPOINT_URL}).",
    )
    parser.add_argument("--output-dir", default="outputs/metadrive", help="Used when --out is not given.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <output-dir>/result_seed<seeds>.json).")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        import yaml

        config = yaml.safe_load(Path(pre.config).read_text()) or {}
        defaults = {key: _as_csv(value) for key, value in (config.get("args") or {}).items()}
        for key in ("arms", "seeds"):
            if key in config:
                defaults[key] = _as_csv(config[key])
        known = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - known)
        if unknown:
            raise SystemExit(f"unknown config keys in {pre.config}: {unknown}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.out is None:
        tag = str(args.seeds).replace(",", "_")
        args.out = str(Path(args.output_dir) / f"result_seed{tag}.json")
    if args.init_checkpoint is not None and not Path(args.init_checkpoint).is_file():
        raise SystemExit(
            f"initial checkpoint not found: {args.init_checkpoint}\n"
            f"download it with: wget -P checkpoints {CHECKPOINT_URL}"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cell_specs = _parse_cells(args)
    heldout_seeds = tuple(int(s) for s in str(args.heldout_seeds).split(","))

    cells = []
    t_start = time.monotonic()
    for spec in cell_specs:
        arm, seed, rho_lr = spec["arm"], spec["seed"], spec["rho_lr"]
        print(f"[metadrive] running arm={arm} seed={seed} rho_lr={rho_lr} steps={args.steps}", flush=True)
        cell = run_cell(
            arm, seed=seed, steps=args.steps, rho_interval=args.rho_interval, du_q=args.du_q,
            rho_lr=rho_lr, cost_budget=args.cost_budget, lambda_lr=args.lambda_lr,
            num_scenarios=args.num_scenarios, heldout_seeds=heldout_seeds,
            heldout_horizon=args.heldout_horizon, actor_lr=args.actor_lr,
            entropy_coefficient=args.entropy_coefficient,
            init_checkpoint=args.init_checkpoint,
            sampling_correction=args.sampling_correction,
            sc_baseline=args.sc_baseline,
            sc_replicates=args.sc_replicates,
            sc_seed_stride=args.sc_seed_stride,
        )
        print(
            f"[metadrive]   -> final_loss={cell['final_loss']:.6f} "
            f"all_losses_finite={cell['all_losses_finite']} "
            f"heldout_mean_welfare={cell['heldout']['mean_episode_welfare']:.4f} "
            f"heldout_mean_success_rate={cell['heldout']['mean_success_rate']:.4f} "
            f"elapsed={cell['elapsed_seconds']:.1f}s",
            flush=True,
        )
        cells.append(cell)
    total_elapsed = time.monotonic() - t_start

    def _finite_nested(value: Any) -> bool:
        arr = np.asarray(value, dtype=np.float64)
        return bool(np.isfinite(arr).all())

    all_losses_finite = all(cell["all_losses_finite"] for cell in cells)
    all_duals_finite = all(_finite_nested(cell["dual_trajectory"]) for cell in cells)
    all_rho_finite = all(_finite_nested(cell["rho_trajectory"]) for cell in cells)
    all_heldout_welfare_finite = all(np.isfinite(cell["heldout"]["mean_episode_welfare"]) for cell in cells)
    all_heldout_cost_finite = all(np.isfinite(cell["heldout"]["mean_episode_native_cost_total"]) for cell in cells)
    lira_rho_moved = any(
        cell["arm"] == "lira" and not np.allclose(np.asarray(cell["rho_trajectory"][0]), np.asarray(cell["final_rho"]))
        for cell in cells
    )

    summary = {
        "schema": "lira-metadrive-v1",
        "env_class": "metadrive.envs.marl_envs.marl_intersection.MultiAgentIntersectionEnv",
        "horizon": 1000,
        "n_agents": 4,
        "num_scenarios": args.num_scenarios,
        "cell_specs": cell_specs,
        "init_checkpoint": args.init_checkpoint,
        "steps": args.steps,
        "rho_interval": args.rho_interval,
        "du_q": args.du_q,
        "cost_budget": args.cost_budget,
        "lambda_lr": args.lambda_lr,
        "heldout_seeds": list(heldout_seeds),
        "heldout_horizon": args.heldout_horizon,
        "actor_lr": args.actor_lr,
        "entropy_coefficient": args.entropy_coefficient,
        "sampling_correction": args.sampling_correction,
        "sc_baseline": args.sc_baseline,
        "sc_replicates": args.sc_replicates,
        "sc_seed_stride": args.sc_seed_stride,
        "cells": cells,
        "total_elapsed_seconds": total_elapsed,
        "all_losses_finite": all_losses_finite,
        "all_duals_finite": all_duals_finite,
        "all_rho_finite": all_rho_finite,
        "all_heldout_welfare_finite": all_heldout_welfare_finite,
        "all_heldout_cost_finite": all_heldout_cost_finite,
        "lira_rho_moved_any_seed": lira_rho_moved,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({
        "out": args.out,
        "all_losses_finite": all_losses_finite,
        "all_duals_finite": all_duals_finite,
        "all_rho_finite": all_rho_finite,
        "all_heldout_welfare_finite": all_heldout_welfare_finite,
        "all_heldout_cost_finite": all_heldout_cost_finite,
        "lira_rho_moved_any_seed": lira_rho_moved,
        "total_elapsed_seconds": total_elapsed,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
