#!/usr/bin/env python3
"""Train Uniform, PAL, and LiRA on Melting Pot Commons Harvest (N=7, K=1).

One process trains all three arms for one training seed from the same initial
learner state and writes one JSON result.  LiRA and Uniform share the same
learner (a single shared dual with cost coefficient ``N * lambda * rho_i``);
LiRA additionally updates the responsibility allocation ``rho`` with the
q-step lookahead gradient (direct unroll + leave-one-out score correction,
``M`` replicates) every ``--outer-every`` response updates.  PAL keeps one
dual per agent driven by the same team cost.

Before training, the script checks that the lookahead leaves the live
learner untouched and that Uniform and LiRA stay bit-identical through one
matched update with the allocation step disabled.

Example (paper setting, one seed)::

    python scripts/train_harvest.py --config configs/harvest.yaml \\
        --meltingpot-source /path/to/meltingpot --seed 10411 \\
        --output-dir runs/harvest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

try:
    import lira  # noqa: F401
except ImportError:  # allow running from a source checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lira.envs.harvest.adapter import (  # noqa: E402
    CommonsHarvestLearningAdapter,
    heldout_arm_order,
    OBS_DIM,
    counterbalanced_arm_order,
    initialize_teacher_prior_,
    seeded_teacher_initial_states,
    teacher_tape_digest,
)
from lira.envs.harvest.env import N_AGENTS  # noqa: E402
from lira.envs.harvest.learner import (  # noqa: E402
    ARM_PAL,
    ARM_UNIFORM,
    CategoricalLearner,
    CategoricalLearnerConfig,
)
from lira.envs.harvest.meta_batch import (  # noqa: E402
    build_chart,
    cross_fitted_meta_batch_eta_gradient,
    embed_full_gradient,
    run_independent_meta_batch,
)
from lira.envs.harvest.transaction import CategoricalTape  # noqa: E402


def _clip_grad_norm(gradient: torch.Tensor, max_norm: float | None) -> torch.Tensor:
    """Cap ``gradient``'s L2 norm at ``max_norm``, preserving direction.

    A no-op (returns ``gradient`` unchanged) when ``max_norm`` is ``None`` or the
    gradient is already within the cap (the paper setting uses no clipping).
    """
    if max_norm is None:
        return gradient
    norm = float(gradient.norm())
    if norm <= max_norm or norm == 0.0:
        return gradient
    return gradient * (max_norm / norm)


def _make_learner(
    source: Path, *, seed: int, arm: str, horizon: int, rho_lr: float,
    lambda_init: float, lambda_lr: float, damage_limit_fraction: float,
    teacher_strength: float,
    shared_policies: list[Any], shared_initial_states: list[Any],
    teacher_cache: dict[str, tuple[int, Any]] | None,
) -> CategoricalLearner:
    config = CategoricalLearnerConfig(
        obs_dim=OBS_DIM,
        categories=8,
        hidden_dim=32,
        horizon=horizon,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coefficient=0.01,
        actor_lr=3e-4,
        critic_lr=1e-3,
        lambda_lr=lambda_lr,
        rho_lr=rho_lr,
        damage_limit=damage_limit_fraction * horizon,
        lambda_init=lambda_init,
        seed=seed,
        meta_du_sc_q=10,
    )
    env = CommonsHarvestLearningAdapter(
        source, max_steps=horizon, shared_policies=shared_policies,
        shared_initial_states=shared_initial_states, teacher_cache=teacher_cache,
    )
    learner = CategoricalLearner(env, config, arm=arm)
    initialize_teacher_prior_(learner, strength=teacher_strength)
    return learner


def _tensor_digest(learner: CategoricalLearner) -> str:
    digest = hashlib.sha256()
    modules = (*learner.actor, *learner.reward_critic, *learner.cost_critic)
    for module in modules:
        for value in module.state_dict().values():
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    for optimizer in (*learner.actor_optimizers, *learner.reward_optimizers, *learner.cost_optimizers):
        for state in optimizer.state_dict()["state"].values():
            for key in sorted(state):
                value = state[key]
                if isinstance(value, torch.Tensor):
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
    digest.update(learner.shared_lambda.values.detach().cpu().numpy().tobytes())
    digest.update(learner.rho.logits.detach().cpu().numpy().tobytes())
    digest.update(repr((learner.rollout_counter, learner.transaction_counter)).encode())
    digest.update(learner._torch_generator.get_state().cpu().numpy().tobytes())
    return digest.hexdigest()


def _optimization_state_digest(learner: CategoricalLearner) -> str:
    """Digest trainable/dual state while intentionally excluding rollout RNG/counters."""
    digest = hashlib.sha256()
    modules = (*learner.actor, *learner.reward_critic, *learner.cost_critic)
    for module in modules:
        for value in module.state_dict().values():
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    for optimizer in (*learner.actor_optimizers, *learner.reward_optimizers, *learner.cost_optimizers):
        digest.update(repr(optimizer.state_dict()).encode())
    digest.update(learner.shared_lambda.values.detach().cpu().numpy().tobytes())
    digest.update(learner.duplicated_lambdas.detach().cpu().numpy().tobytes())
    digest.update(learner.rho.logits.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _tape(seed: int, index: int, *, horizon: int) -> CategoricalTape:
    tape_seed = seed * 10_000_000 + 5_000_000 + index * 104_729
    generator = torch.Generator().manual_seed(tape_seed)
    uniforms = torch.rand((horizon, N_AGENTS), generator=generator, dtype=torch.float64)
    return CategoricalTape(tape_seed, uniforms)


def _loo_gradient(learner: CategoricalLearner, *, seed: int, horizon: int, q: int, replicates: int) -> tuple[torch.Tensor, list[float]]:
    chart = build_chart(learner)
    groups = []
    for replicate in range(replicates):
        base = replicate * (q + 1)
        updates = tuple(_tape(seed, base + index, horizon=horizon) for index in range(q))
        terminal = (_tape(seed, base + q, horizon=horizon),)
        groups.append((updates, terminal))
    live_env = learner.env
    meta_env = CommonsHarvestLearningAdapter(
        live_env.source_path,
        max_steps=live_env.max_steps,
        shared_policies=live_env._policies,
        shared_initial_states=live_env._shared_initial_states,
        teacher_cache=live_env._teacher_cache,
    )
    try:
        rows = run_independent_meta_batch(learner, chart, groups, env=meta_env)
    finally:
        meta_env.close()
    eta_gradient = cross_fitted_meta_batch_eta_gradient(rows)
    full = embed_full_gradient(eta_gradient, chart, N_AGENTS)
    return full, [float(row.welfare) for row in rows]


def _rollout_row(result: Any) -> dict[str, Any]:
    rollout = result.rollout if hasattr(result, "rollout") else result
    return {
        "welfare": float(rollout.rewards.sum()),
        "reward_per_agent": [float(x) for x in rollout.rewards.sum(dim=0)],
        "episode_cost": float(rollout.episode_damage),
        "teacher_action_agreement": float(
            (rollout.actions == rollout.observations[:, :, :8].argmax(dim=-1)).float().mean()
        ),
    }


def _transaction_row(learner: CategoricalLearner, result: Any) -> dict[str, Any]:
    row = _rollout_row(result)
    row.update({
        "shared_lambda": float(learner.lambda_value),
        "duplicated_lambdas": [float(value) for value in learner.duplicated_lambdas],
        "rho": [float(value) for value in learner.rho_value],
    })
    return row


def _mean_rollout_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one rollout row is required")
    return {
        "welfare": sum(row["welfare"] for row in rows) / len(rows),
        "reward_per_agent": [
            sum(row["reward_per_agent"][i] for row in rows) / len(rows)
            for i in range(N_AGENTS)
        ],
        "episode_cost": sum(row["episode_cost"] for row in rows) / len(rows),
        "teacher_action_agreement": sum(row["teacher_action_agreement"] for row in rows) / len(rows),
        "rollouts": rows,
    }


def _matched_rollout(left: Any, right: Any) -> dict[str, bool]:
    return {
        "observations_exact": bool(torch.equal(left.rollout.observations, right.rollout.observations)),
        "actions_exact": bool(torch.equal(left.rollout.actions, right.rollout.actions)),
        "old_log_probs_exact": bool(torch.equal(left.rollout.old_log_probs, right.rollout.old_log_probs)),
        "rewards_exact": bool(torch.equal(left.rollout.rewards, right.rollout.rewards)),
        "cost_exact": bool(torch.equal(left.rollout.shared_damage, right.rollout.shared_damage)),
    }


def _replay_mutable_update(learner: CategoricalLearner, rollout: Any) -> None:
    """Apply the live learner update to one already-collected shared rollout."""
    reward_adv, cost_adv, reward_returns, cost_returns = learner._advantages(rollout)
    active = (rollout.dones == 0).to(learner.dtype)
    active_count = active.sum().clamp_min(1.0)
    for agent in range(N_AGENTS):
        obs = rollout.observations[:, agent]
        learner.reward_optimizers[agent].zero_grad(set_to_none=True)
        reward_loss = (((learner.reward_critic[agent](obs) - reward_returns[:, agent]) ** 2) * active).sum() / active_count
        reward_loss.backward()
        learner.reward_optimizers[agent].step()
        learner.cost_optimizers[agent].zero_grad(set_to_none=True)
        cost_loss = (((learner.cost_critic[agent](obs) - cost_returns) ** 2) * active).sum() / active_count
        cost_loss.backward()
        learner.cost_optimizers[agent].step()
        learner.actor_optimizers[agent].zero_grad(set_to_none=True)
        actor_loss = learner._actor_loss(agent, rollout, reward_adv, cost_adv)
        actor_loss.backward()
        learner.actor_optimizers[agent].step()
    learner._update_duals(rollout.episode_damage)
    learner.transaction_counter += 1
    learner.last_rollout = rollout


def _sync_after_shared_collect(source: CategoricalLearner, target: CategoricalLearner) -> None:
    target._torch_generator.set_state(source._torch_generator.get_state())
    target.rollout_counter = source.rollout_counter


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _state_parity(left: CategoricalLearner, right: CategoricalLearner) -> dict[str, bool]:
    return {
        "actors": _nested_equal(left.actor.state_dict(), right.actor.state_dict()),
        "reward_critics": _nested_equal(left.reward_critic.state_dict(), right.reward_critic.state_dict()),
        "cost_critics": _nested_equal(left.cost_critic.state_dict(), right.cost_critic.state_dict()),
        "actor_optimizers": all(_nested_equal(a.state_dict(), b.state_dict()) for a, b in zip(left.actor_optimizers, right.actor_optimizers, strict=True)),
        "reward_optimizers": all(_nested_equal(a.state_dict(), b.state_dict()) for a, b in zip(left.reward_optimizers, right.reward_optimizers, strict=True)),
        "cost_optimizers": all(_nested_equal(a.state_dict(), b.state_dict()) for a, b in zip(left.cost_optimizers, right.cost_optimizers, strict=True)),
        "lambda": bool(torch.equal(left.shared_lambda.values, right.shared_lambda.values)),
        "rho": bool(torch.equal(left.rho.logits, right.rho.logits)),
        "counters": (left.rollout_counter, left.transaction_counter) == (right.rollout_counter, right.transaction_counter),
        "local_rng": bool(torch.equal(left._torch_generator.get_state(), right._torch_generator.get_state())),
    }


PAPER_DEFAULTS = {
    "horizon": 100,
    "q": 10,
    "replicates": 4,
    "rho_lr": 0.15,
    "lambda_init": 0.3,
    "lambda_lr": 0.05,
    "damage_limit_fraction": 0.68,
    "response_updates": 60,
    "heldout_rollouts": 5,
    "teacher_strength": 3.5,
    "min_teacher_agreement": 0.4,
    "outer_every": 5,
}


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Uniform/PAL/LiRA on Melting Pot Commons Harvest (one seed per run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config; its `args` mapping sets defaults, CLI flags override")
    parser.add_argument("--meltingpot-source", type=Path, default=os.environ.get("MELTINGPOT_SOURCE"),
                        help="patched Melting Pot source checkout (or set MELTINGPOT_SOURCE)")
    parser.add_argument("--seed", type=int, default=None,
                        help="training seed; if omitted, every seed listed in the config is run in turn")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/harvest"),
                        help="directory for seed<seed>.json results")
    parser.add_argument("--out", type=Path, default=None,
                        help="explicit output JSON path (single seed only; overrides --output-dir)")
    parser.add_argument("--horizon", type=int, help="episode length (steps)")
    parser.add_argument("--q", type=int, help="inner PPO updates per lookahead replicate")
    parser.add_argument("--replicates", type=int, help="M, independent lookahead replicates")
    parser.add_argument("--rho-lr", type=float, help="responsibility step size")
    parser.add_argument("--lambda-init", type=float, help="initial shared dual")
    parser.add_argument("--lambda-lr", type=float, help="dual step size")
    parser.add_argument("--damage-limit-fraction", type=float,
                        help="episode cost budget as a fraction of the horizon (0.68 * 100 = 68)")
    parser.add_argument("--response-updates", type=int, help="training (response) updates per arm")
    parser.add_argument("--debug-max-updates", type=int, default=None,
                        help="run only this many leading response updates (quick checks only)")
    parser.add_argument("--heldout-rollouts", type=int, help="held-out evaluation episodes per arm")
    parser.add_argument("--teacher-strength", type=float,
                        help="warm-start strength toward the pretrained bot's proposed action")
    parser.add_argument("--min-teacher-agreement", type=float,
                        help="minimum bot-action agreement required of the initial policy")
    parser.add_argument("--outer-every", type=int,
                        help="allocation update cadence in response updates (0 = only the initial update)")
    parser.add_argument(
        "--rho-grad-clip-norm", type=float, default=None,
        help="optional cap on the allocation gradient's L2 norm (unset in the paper)",
    )
    parser.set_defaults(**PAPER_DEFAULTS)
    return parser


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args()
    config = _load_config(known.config)
    parser = build_parser()
    parser.set_defaults(**{key.replace("-", "_"): value for key, value in (config.get("args") or {}).items()})
    args = parser.parse_args()
    if args.meltingpot_source is None:
        parser.error("--meltingpot-source (or MELTINGPOT_SOURCE) is required")
    seeds = [args.seed] if args.seed is not None else list(config.get("seeds") or [])
    if not seeds:
        parser.error("no seed given (use --seed or a config with `seeds`)")
    if args.out is not None and len(seeds) != 1:
        parser.error("--out requires a single --seed")
    for seed in seeds:
        out = args.out if args.out is not None else args.output_dir / f"seed{seed}.json"
        run_seed(args, seed=int(seed), out=out)
    return 0


def run_seed(args: argparse.Namespace, *, seed: int, out: Path) -> None:
    if args.q != 10 or args.replicates < 2:
        raise ValueError("the Harvest protocol uses q=10 and at least two leave-one-out replicates")
    if args.rho_grad_clip_norm is not None and args.rho_grad_clip_norm <= 0:
        raise ValueError("--rho-grad-clip-norm must be positive when provided")
    if args.response_updates < 1 or (args.debug_max_updates is not None and args.debug_max_updates < 1):
        raise ValueError("response-updates and a provided debug-max-updates must be positive")
    updates_to_run = min(args.response_updates, args.debug_max_updates or args.response_updates)
    source = args.meltingpot_source
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from meltingpot import bot
    shared_policies = [bot.build("commons_harvest__open__free_0") for _ in range(N_AGENTS)]
    # SavedModelPolicy lazily builds its initial-state graph on first use.  Do
    # that once before any matched arm so graph construction cannot perturb
    # one arm's otherwise paired Python-random seed stream.
    teacher_rng_seed = int(seed) + 7919
    # Warm the lazy initial-state graph before matched arms. This consumes a
    # separate reproducible teacher key sequence and restores caller RNG.
    seeded_teacher_initial_states(shared_policies, seed=seed + 123)
    # ``step`` is independently lazy-built in the TF1 policy wrapper.  Warm it
    # on a disposable seeded environment as well, before either matched arm.
    warm_adapter = CommonsHarvestLearningAdapter(
        source, max_steps=1, shared_policies=shared_policies
    )
    warm_adapter.reset(seed + 456)
    warm_adapter.close()
    # SavedModelPolicy samples via an explicit initial-state PRNG key drawn
    # from Python random; its recurrent state threads the key on every action.
    shared_initial_states = seeded_teacher_initial_states(
        shared_policies, seed=teacher_rng_seed
    )
    # Each arm evaluates the same pretrained policy on its own native state.
    # Materialize one immutable action/recurrent-state tape keyed by
    # (environment seed, step, player), so teacher-policy RNG consumption
    # cannot silently perturb Uniform/PAL/LiRA within a matched run. Its digest
    # is saved; cross-process parity additionally depends on the native
    # engine's reset determinism.
    teacher_cache: dict[str, tuple[int, Any]] = {}

    common = dict(
        source=source, seed=seed, horizon=args.horizon, rho_lr=args.rho_lr,
        lambda_init=args.lambda_init, lambda_lr=args.lambda_lr,
        damage_limit_fraction=args.damage_limit_fraction,
        teacher_strength=args.teacher_strength,
        shared_policies=shared_policies, shared_initial_states=shared_initial_states,
        teacher_cache=teacher_cache,
    )
    uniform = _make_learner(arm=ARM_UNIFORM, **common)
    lira = _make_learner(arm=ARM_UNIFORM, **common)
    pal = _make_learner(arm=ARM_PAL, **common)
    initial_optimization_state_sha256 = {
        "uniform": _optimization_state_digest(uniform),
        "pal": _optimization_state_digest(pal),
        "lira": _optimization_state_digest(lira),
    }
    train_seed, parity_seed, response_seed, heldout_seed = (seed * 1000 + i for i in (1, 2, 3, 900))
    try:
        initial_parity = _state_parity(uniform, lira)
        print(json.dumps({"stage": "initial_state_parity", **initial_parity}), flush=True)
        if not all(initial_parity.values()):
            raise RuntimeError(f"same-S0 learner state mismatch: {initial_parity}")
        learners = {"uniform": uniform, "lira": lira, "pal": pal}
        first_order = counterbalanced_arm_order(tuple(learners), seed=seed, phase=0)
        first_raw = {arm: learners[arm].collect(seed=train_seed) for arm in first_order}
        shared_source = next(arm for arm in first_order if arm in ("uniform", "lira"))
        shared_target = "lira" if shared_source == "uniform" else "uniform"
        _sync_after_shared_collect(learners[shared_source], learners[shared_target])
        _replay_mutable_update(uniform, first_raw[shared_source])
        _replay_mutable_update(lira, first_raw[shared_source])
        _replay_mutable_update(pal, first_raw["pal"])
        first_update_parity = _state_parity(uniform, lira)
        print(json.dumps({"stage": "first_update_parity", **first_update_parity}), flush=True)
        if not all(first_update_parity.values()):
            raise RuntimeError(f"same-raw pre-meta mutable update parity failed: {first_update_parity}")
        competence = _rollout_row(first_raw[shared_source])
        if competence["welfare"] <= 0 or competence["teacher_action_agreement"] < args.min_teacher_agreement:
            raise RuntimeError(f"teacher-prior Uniform competence gate failed: {competence}")

        before_meta = _tensor_digest(lira)
        gradient, meta_welfare = _loo_gradient(
            lira, seed=seed + 17, horizon=args.horizon, q=args.q, replicates=args.replicates
        )
        after_meta = _tensor_digest(lira)
        if before_meta != after_meta:
            raise RuntimeError("outer_lr=0 meta branch mutated the live learner")
        if gradient.shape != (1, N_AGENTS) or not torch.isfinite(gradient).all() or float(gradient.norm()) == 0.0:
            raise RuntimeError("DU+LOO-SC gradient is disconnected or invalid")

        parity_order = counterbalanced_arm_order(tuple(learners), seed=seed, phase=1)
        parity_raw = {arm: learners[arm].collect(seed=parity_seed) for arm in parity_order}
        parity_source = next(arm for arm in parity_order if arm in ("uniform", "lira"))
        parity_target = "lira" if parity_source == "uniform" else "uniform"
        _sync_after_shared_collect(learners[parity_source], learners[parity_target])
        _replay_mutable_update(uniform, parity_raw[parity_source])
        _replay_mutable_update(lira, parity_raw[parity_source])
        _replay_mutable_update(pal, parity_raw["pal"])
        parity = {"shared_raw_rollout": True, "post_state_exact": _tensor_digest(uniform) == _tensor_digest(lira)}
        if not all(parity.values()):
            raise RuntimeError(f"outer_lr=0 same-S0/RNG/update parity failed: {parity}")
        rho_before = lira.rho_value.detach().clone()
        clipped_gradient = _clip_grad_norm(gradient, args.rho_grad_clip_norm)
        # apply_tangent_gradient_ is a descent primitive (logits -= step*tangent);
        # `gradient` is a welfare-ASCENT direction, so it is negated once here.
        lira.rho.apply_tangent_gradient_(-clipped_gradient.to(lira.dtype), args.rho_lr)
        rho_after = lira.rho_value.detach().clone()
        if torch.equal(rho_before, rho_after):
            raise RuntimeError("original rho updater did not move the simplex")

        if args.response_updates < 1 or args.heldout_rollouts < 1:
            raise ValueError("response-updates and heldout-rollouts must be positive")
        if args.outer_every < 0:
            raise ValueError("outer-every must be nonnegative")
        meta_history = [{
            "after_response_update": 0,
            "gradient_norm": float(gradient.norm()),
            "gradient": gradient.tolist(),
            "meta_replicate_welfare": meta_welfare,
            "rho": [float(value) for value in rho_after],
        }]
        response_trace = {"uniform": [], "pal": [], "lira": []}
        teacher_tape_prefix_sha256: list[str] = []
        for update in range(updates_to_run):
            if update > 0 and args.outer_every and update % args.outer_every == 0:
                before_repeat_meta = _tensor_digest(lira)
                repeat_gradient, repeat_welfare = _loo_gradient(
                    lira,
                    seed=seed + 17 + 1_000_003 * (update // args.outer_every),
                    horizon=args.horizon,
                    q=args.q,
                    replicates=args.replicates,
                )
                if before_repeat_meta != _tensor_digest(lira):
                    raise RuntimeError("repeated outer branch mutated the live learner")
                repeat_norm = float(repeat_gradient.norm())
                if not torch.isfinite(repeat_gradient).all():
                    print(json.dumps({"stage": "repeated_gradient_gate", "update": update,
                                      "finite": bool(torch.isfinite(repeat_gradient).all()),
                                      "norm": repeat_norm}), flush=True)
                    raise RuntimeError("repeated DU+LOO-SC gradient is disconnected or invalid")
                if repeat_norm == 0.0:
                    print(json.dumps({"stage": "repeated_gradient_zero_update", "update": update,
                                      "finite": True, "norm": 0.0}), flush=True)
                    meta_history.append({"after_response_update": update,
                                         "gradient_norm": 0.0, "gradient": repeat_gradient.tolist(),
                                         "meta_replicate_welfare": repeat_welfare,
                                         "rho": [float(value) for value in lira.rho_value],
                                         "update_applied": False, "zero_gradient": True})
                    continue
                clipped_repeat_gradient = _clip_grad_norm(repeat_gradient, args.rho_grad_clip_norm)
                # Same ascent-direction-into-descent-primitive negation as the
                # first meta gradient application above.
                lira.rho.apply_tangent_gradient_(-clipped_repeat_gradient.to(lira.dtype), args.rho_lr)
                meta_history.append({
                    "after_response_update": update,
                    "gradient_norm": float(repeat_gradient.norm()),
                    "gradient": repeat_gradient.tolist(),
                    "meta_replicate_welfare": repeat_welfare,
                    "rho": [float(value) for value in lira.rho_value],
                })
            update_seed = response_seed + 104729 * update
            order = counterbalanced_arm_order(tuple(learners), seed=seed, phase=2 + update)
            for arm in order:
                result = learners[arm].transaction(seed=update_seed)
                response_trace[arm].append(_transaction_row(learners[arm], result))
            teacher_tape_prefix_sha256.append(teacher_tape_digest(teacher_cache))
        eval_before = {arm: _optimization_state_digest(learner) for arm, learner in (("uniform", uniform), ("pal", pal), ("lira", lira))}
        heldout_rows = {"uniform": [], "pal": [], "lira": []}
        for index in range(args.heldout_rollouts):
            eval_seed = heldout_seed + 130363 * index
            # Held-out ordering is fixed independently of the planned training
            # horizon so matched 60/90 prefixes also share eval tape assignment.
            order = heldout_arm_order(tuple(learners), seed=seed, rollout_index=index)
            for arm in order:
                heldout_rows[arm].append(_rollout_row(learners[arm].collect(seed=eval_seed)))
        heldout = {arm: _mean_rollout_rows(rows) for arm, rows in heldout_rows.items()}
        eval_after = {arm: _optimization_state_digest(learner) for arm, learner in (("uniform", uniform), ("pal", pal), ("lira", lira))}
        eval_read_only = {arm: eval_before[arm] == eval_after[arm] for arm in eval_before}
        if not all(eval_read_only.values()):
            raise RuntimeError(f"held-out evaluation mutated optimization state: {eval_read_only}")
        payload = {
            "schema": "lira-harvest-v1",
            "status": "PASS",
            "seed": seed,
            "n_agents": N_AGENTS,
            "n_constraints": 1,
            "horizon": args.horizon,
            "q": args.q,
            "replicates": args.replicates,
            "lambda_init": args.lambda_init,
            "lambda_lr": args.lambda_lr,
            "damage_limit": args.damage_limit_fraction * args.horizon,
            "response_updates": updates_to_run,
            "planned_response_updates": args.response_updates,
            "debug_max_updates": args.debug_max_updates,
            "teacher_rng": {
                "algorithm": "Python-seeded SavedModel initial PRNG key, threaded through recurrent state",
                "seed": teacher_rng_seed,
                "initial_state_sha256": teacher_tape_digest({
                    str(index): (0, state)
                    for index, state in enumerate(shared_initial_states)
                }),
                "shared_across_arms": True,
                "tape_entries": len(teacher_cache),
                "tape_sha256": teacher_tape_digest(teacher_cache),
            },
            "teacher_tape_prefix_sha256": teacher_tape_prefix_sha256,
            "outer_every": args.outer_every,
            "rho_grad_clip_norm": args.rho_grad_clip_norm,
            "meta_history": meta_history,
            "response_trace": response_trace,
            "heldout_rollouts": args.heldout_rollouts,
            "teacher_strength": args.teacher_strength,
            "min_teacher_agreement": args.min_teacher_agreement,
            "eval_read_only": eval_read_only,
            "meta_env_isolated": True,
            # Preserve each held-out rollout so high-repetition evaluation can
            # report block means from one post-training checkpoint rather than
            # hiding native-environment entropy behind one aggregate mean.
            "heldout_raw": heldout_rows,
            "config": uniform.config.canonical_dict(),
            "environment_digest": uniform.environment_digest,
            "cost": "sum_t(1-live_apples_t/64)",
            "competence": competence,
            "outer_lr_zero_parity": parity,
            "initial_optimization_state_sha256": initial_optimization_state_sha256,
            "gradient_norm": float(gradient.norm()),
            "gradient": gradient.tolist(),
            "meta_replicate_welfare": meta_welfare,
            "rho_before": rho_before.tolist(),
            "rho_after": rho_after.tolist(),
            "heldout": heldout,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"seed": seed, "out": str(out),
                          "heldout_welfare": {arm: heldout[arm]["welfare"] for arm in heldout},
                          "heldout_episode_cost": {arm: heldout[arm]["episode_cost"] for arm in heldout}},
                         sort_keys=True), flush=True)
    finally:
        uniform.close()
        lira.close()
        pal.close()
        for policy in shared_policies:
            policy.close()


if __name__ == "__main__":
    raise SystemExit(main())
