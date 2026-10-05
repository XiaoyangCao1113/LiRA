"""Live-dual training loop for the Uniform, PAL and LiRA arms on MABIM.

All arms share the dimensionless constraint ``C_k / d_k <= 1`` with budgets
``d_k`` calibrated on the validation period, and the same committed learner
schedule: every outer cycle runs ``q`` on-policy functional PPO/Adam updates,
each followed by a projected dual update.

* ``uniform``: K shared multipliers, fixed uniform responsibilities.
* ``pal``: independent per-agent multipliers (K x N), each updated from the
  shared cost, with no responsibility allocation.
* ``lira``: K shared multipliers and learned responsibilities
  ``rho = softmax(z)``; before each committed block, ``z`` takes one
  gradient-ascent step along the leave-one-out score-corrected q-step
  lookahead gradient averaged over ``m_replicates`` independent replicas.
* ``lira_frozen_rho``: LiRA's estimator is run but the allocation update is
  disabled (ablation).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from . import meta_gradient
from . import transaction as tx
from .learner import load_shared_static_checkpoint
from .live_dual import (
    effective_raw_multiplier,
    normalize_cost_advantages,
    projected_dual_update,
)

ARMS = ("uniform", "pal", "lira", "lira_frozen_rho")


def _load(checkpoint: Path, seed: int, training_mode: str, env_root=None):
    runner = tx.env_runner(env_root)
    if training_mode not in {"train", "test"}:
        raise ValueError(f"unsupported MABIM training mode {training_mode!r}")
    # The environment wrapper is built in test mode; switch the underlying
    # ReplenishmentEnv to the requested data split before its first reset.
    runner.env.env.env.mode = training_mode
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    counters = load_shared_static_checkpoint(runner, checkpoint, expected_source_digests=raw["source_digests"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return runner, counters, raw["source_digests"]


def _detach(params, states):
    return (
        {key: value.detach().clone().requires_grad_(True) for key, value in params.items()},
        {key: (m.detach().clone(), v.detach().clone(), int(step)) for key, (m, v, step) in states.items()},
    )


def _committed_block(runner, params, states, z, dual, budgets, eta, q, dual_max, *, pal: bool):
    rows = []
    for inner in range(q):
        tape, _ = tx.collect_onpolicy(runner, params)
        raw_cost = tape.costs.sum(0).detach()
        norm_adv = normalize_cost_advantages(tape.cost_adv, budgets)
        rho = torch.softmax(z, dim=-1)
        coeff = dual if pal else runner.config.n_agents * dual[:, None] * rho
        penalty_by_k = torch.stack([
            torch.linalg.vector_norm(norm_adv[:, :, k] * coeff[k].unsqueeze(0))
            for k in range(runner.config.n_constraints)
        ])
        before = dual.clone()
        objective = tx.loss(runner, params, z, tape, dual, cost_budgets=budgets)
        params, states = tx.adam(params, states, objective, runner.config.actor_lr, (.9, .999), 1e-8)
        params, states = _detach(params, states)
        if pal:
            residual = raw_cost / budgets - 1.0
            dual = torch.clamp(dual + eta * residual[:, None], min=0.0, max=dual_max)
        else:
            dual, residual = projected_dual_update(dual, raw_cost, budgets, eta, dual_max=dual_max)
        env_base = runner.env
        for _ in range(3):
            if not hasattr(env_base, "env"):
                break
            env_base = env_base.env
        rows.append({
            "inner_update": inner + 1,
            "raw_cost": raw_cost.tolist(),
            "budget": budgets.tolist(),
            "normalized_residual": residual.tolist(),
            "dual_before": before.tolist(),
            "dual_after": dual.tolist(),
            "effective_raw_multiplier_before": effective_raw_multiplier(before, budgets).tolist(),
            "effective_raw_multiplier_after": effective_raw_multiplier(dual, budgets).tolist(),
            "raw_cost_advantage_l2_by_constraint": torch.linalg.vector_norm(tape.cost_adv, dim=(0, 1)).tolist(),
            "normalized_cost_advantage_l2_by_constraint": torch.linalg.vector_norm(norm_adv, dim=(0, 1)).tolist(),
            "normalized_penalty_advantage_l2_by_constraint": penalty_by_k.tolist(),
            "rho": rho.tolist(),
            "training_data_mode": getattr(env_base, "mode", None),
            "training_window_start": str(env_base.picked_start_date.date()) if hasattr(env_base, "picked_start_date") else None,
            "training_window_end": str(env_base.picked_end_date.date()) if hasattr(env_base, "picked_end_date") else None,
        })
    return params, states, dual, rows


def _write_resume_checkpoint(path: Path, payload: dict) -> None:
    """Atomically persist a resume checkpoint (temp file + os.replace).

    Avoids leaving a truncated/corrupt file behind if the process is killed
    (e.g. by a scheduler walltime limit) mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run(
    *, checkpoint: Path, budget_file: Path, output: Path, arm: str, seed: int,
    eta: float, q: int = 5, cycles: int = 20, outer_lr: float = 0.05,
    outer_objective_scale: float = 1.0,
    m_replicates: int = 12, dual_init: float = 0.01, dual_max: float = 100.0,
    checkpoint_every: int = 1, resume_from: Path | None = None,
    resume_checkpoint_path: Path | None = None,
    training_mode: str = "test",
    env_root: str | os.PathLike | None = None,
) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)

    resume_state = None
    if resume_from is not None and Path(resume_from).exists():
        resume_state = torch.load(resume_from, map_location="cpu", weights_only=False)
        # The resume checkpoint is authoritative for everything that affects
        # reproducibility, so a resumed run does not depend on argv matching
        # the original invocation -- only --cycles and --output are allowed
        # to differ (extend the run / write a fresh output).
        arm = resume_state["arm"]
        seed = resume_state["seed"]
        rcfg = resume_state["config"]
        checkpoint = Path(rcfg["checkpoint"])
        budget_file = Path(rcfg["budget_file"])
        eta = rcfg["eta"]
        q = rcfg["q"]
        outer_lr = rcfg["outer_lr"]
        outer_objective_scale = rcfg.get("outer_objective_scale", 1.0)
        m_replicates = rcfg["m_replicates"]
        dual_init = rcfg["dual_init"]
        dual_max = rcfg["dual_max"]
        training_mode = rcfg.get("training_mode", "test")

    budget_payload = json.loads(budget_file.read_text())
    budgets = torch.as_tensor(budget_payload["budgets"], dtype=torch.float64)
    runner, counters, source_digests = _load(checkpoint, seed, training_mode, env_root)
    k, n = runner.config.n_constraints, runner.config.n_agents
    pal = arm == "pal"
    is_lira = arm in {"lira", "lira_frozen_rho"}
    freeze_rho = arm == "lira_frozen_rho"
    if outer_objective_scale <= 0:
        raise ValueError(f"outer_objective_scale must be positive, got {outer_objective_scale}")
    if outer_objective_scale != 1.0 and not is_lira:
        raise ValueError("outer objective scaling applies only to LiRA arms")

    if resume_state is not None:
        params = {
            key: value.detach().clone().requires_grad_(True)
            for key, value in resume_state["params"].items()
        }
        states = {
            key: (m.detach().clone(), v.detach().clone(), int(step))
            for key, (m, v, step) in resume_state["states"].items()
        }
        tx.load_functional_state(runner, params, states)
        steps_before = resume_state["steps_before"]
        digest_before = resume_state["digest_before"]
        source_digests = resume_state["source_digests"]
        counters = dict(resume_state["counters"])
        z = resume_state["z"].detach().clone()
        dual = resume_state["dual"].detach().clone()
        commit_schedule = list(resume_state["commit_schedule"])
        all_updates = list(resume_state["all_updates"])
        cycle_rows = list(resume_state["cycle_rows"])
        start_cycle = int(resume_state["next_cycle"])
    else:
        params, states = tx.capture(runner)
        steps_before = sorted({int(state[2]) for state in states.values()})
        digest_before = tx._optimizer_state_digest(states)
        z = torch.zeros((k, n), dtype=torch.float64)
        dual = torch.full((k, n) if pal else (k,), dual_init, dtype=torch.float64)
        commit_schedule = []
        all_updates = []
        cycle_rows = []
        start_cycle = 0

    resume_path = resume_checkpoint_path or output.with_suffix(".resume.pt")

    for cycle in range(start_cycle, cycles):
        rho_before = torch.softmax(z, dim=-1)
        meta_record = None
        if is_lira:
            # Every streaming replica starts from the exact current committed
            # learner/Adam/dual state; only its rollout RNG differs.
            replica_seeds = tuple(
                tx._phase_seed(seed, "live-dual-meta", cycle, replica)
                for replica in range(m_replicates)
            )

            def build_runner(replica_seed: int):
                replica, _, _ = _load(checkpoint, replica_seed, training_mode, env_root)
                tx.load_functional_state(replica, params, states)
                tx._bind_runner_rng(replica, replica_seed)
                return replica

            direct, correction, corrected, meta_record = meta_gradient.meta_batch_du_sc_outer_gradient(
                build_runner, replica_seeds, dual, q, z_init=z, baseline="loo",
                cost_budgets=budgets, dual_eta=eta, dual_max=dual_max,
            )
            if not all(torch.isfinite(value).all() for value in (direct, correction, corrected)):
                raise RuntimeError(f"non-finite LiRA meta-gradient at cycle {cycle}")
            direct_raw, correction_raw, corrected_raw = direct, correction, corrected
            direct = direct_raw * outer_objective_scale
            correction = correction_raw * outer_objective_scale
            corrected = corrected_raw * outer_objective_scale
            direct_norm = torch.linalg.vector_norm(direct_raw)
            correction_norm = torch.linalg.vector_norm(correction_raw)
            denominator = direct_norm * correction_norm
            meta_record.update({
                "outer_objective_scale": outer_objective_scale,
                "outer_objective_scale_semantics": (
                    "agent-mean welfare gradient converted to benchmark native total-profit units"
                    if outer_objective_scale == n else "explicit scalar applied equally to DU and SC gradients"
                ),
                "direct_gradient_l2_unscaled": float(direct_norm),
                "sampling_correction_gradient_l2_unscaled": float(correction_norm),
                "corrected_gradient_l2_unscaled": float(torch.linalg.vector_norm(corrected_raw)),
                "direct_gradient_l2_scaled": float(torch.linalg.vector_norm(direct)),
                "sampling_correction_gradient_l2_scaled": float(torch.linalg.vector_norm(correction)),
                "corrected_gradient_l2_scaled": float(torch.linalg.vector_norm(corrected)),
                "direct_gradient_l2": float(direct_norm),
                "sampling_correction_gradient_l2": float(correction_norm),
                "corrected_gradient_l2": float(torch.linalg.vector_norm(corrected)),
                "du_sc_cosine": None if float(denominator) == 0.0 else float(
                    torch.sum(direct_raw * correction_raw) / denominator
                ),
                "rho_update_applied": not freeze_rho,
            })
            if not freeze_rho:
                z = (z + outer_lr * corrected).detach()
            meta_record["max_abs_delta_z"] = float((outer_lr * corrected).abs().max())
            meta_record["max_abs_delta_rho"] = float((torch.softmax(z, dim=-1) - rho_before).abs().max())
        commit_seed = tx._phase_seed(seed, "live-dual-commit", cycle)
        commit_schedule.append(commit_seed)
        tx._bind_runner_rng(runner, commit_seed)
        dual_before = dual.clone()
        params, states, dual, update_rows = _committed_block(
            runner, params, states, z, dual, budgets, eta, q, dual_max, pal=pal,
        )
        for row in update_rows:
            row.update({"cycle": cycle, "global_update": cycle * q + row["inner_update"], "rng_seed": commit_seed})
        all_updates.extend(update_rows)
        cycle_rows.append({
            "cycle": cycle,
            "commit_rng_seed": commit_seed,
            "rho_before": rho_before.tolist(),
            "rho_after": torch.softmax(z, dim=-1).tolist(),
            "dual_before": dual_before.tolist(),
            "dual_after": dual.tolist(),
            "meta": meta_record,
            "rho_update_applied": bool(is_lira and not freeze_rho),
        })

        if checkpoint_every and checkpoint_every > 0 and (cycle + 1) % checkpoint_every == 0:
            _write_resume_checkpoint(resume_path, {
                "format": "mabim-live-dual-resume-v1",
                "next_cycle": cycle + 1,
                "arm": arm,
                "seed": seed,
                "params": {key: value.detach().clone() for key, value in params.items()},
                "states": {
                    key: (m.detach().clone(), v.detach().clone(), int(step))
                    for key, (m, v, step) in states.items()
                },
                "dual": dual.detach().clone(),
                "z": z.detach().clone(),
                "counters": dict(counters),
                "commit_schedule": list(commit_schedule),
                "all_updates": [dict(row) for row in all_updates],
                "cycle_rows": [dict(row) for row in cycle_rows],
                "steps_before": steps_before,
                "digest_before": digest_before,
                "source_digests": source_digests,
                "config": {
                    "checkpoint": str(checkpoint),
                    "budget_file": str(budget_file),
                    "eta": eta,
                    "q": q,
                    "cycles": cycles,
                    "outer_lr": outer_lr,
                    "outer_objective_scale": outer_objective_scale,
                    "m_replicates": m_replicates,
                    "dual_init": dual_init,
                    "dual_max": dual_max,
                    "training_mode": training_mode,
                },
            })

    steps_after = sorted({int(state[2]) for state in states.values()})
    counters = dict(counters)
    total_updates = q * cycles
    counters["updates"] = int(counters.get("updates", steps_before[0])) + total_updates
    counters["steps"] = int(counters.get("steps", steps_before[0] * runner.config.horizon)) + total_updates * runner.config.horizon
    counters["seed"] = seed
    topology = "KxN_independent" if pal else "K_shared"
    dual_meta = {
        "topology": topology,
        "constraint_form": "C_k/d_k <= 1",
        "actor_lagrangian": "lambda_ki * C_ki/d_k" if pal else "lambda_k * N*rho_ki*C_ki/d_k",
        "shared_global_safety_target": True,
        "dual_update": "projected relative residual C_k/d_k-1; PAL stores and updates KxN entries independently",
        "budgets": budgets.tolist(),
        "eta": eta,
        "cadence": 1,
        "dual_clip": [0.0, dual_max],
        "dual_init": dual_init,
        "dual_final": dual.tolist(),
        "effective_raw_multiplier_final": effective_raw_multiplier(dual, budgets).tolist(),
        "training_data_mode": training_mode,
        "training_data_window": (
            "random-intercepted train split (2018-08-01..2021-06-30), 60-step prefix"
            if training_mode == "train" else "test split (2021-09-01..2021-10-30)"
        ),
    }
    source_digests = dict(source_digests)
    source_digests["transaction.py"] = tx.sha(Path(tx.__file__))
    source_digests["trainer.py"] = tx.sha(Path(__file__))
    result = tx.save_arm(
        runner, params, states, torch.softmax(z, dim=-1), output, seed,
        f"{arm}_live_dual", dual_meta, counters, source_digests=source_digests,
    )
    summary = {
        "schema": "mabim-live-dual-three-arm-v1",
        **result,
        "arm": arm,
        "seed": seed,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": tx.sha(checkpoint),
        "source_digests": source_digests,
        "budget_artifact": str(budget_file),
        "budget_artifact_payload": budget_payload,
        "dual_topology": topology,
        "pal_definition_preserved": pal,
        "rho_topology": ("fixed uniform KxN simplex; LiRA gradient estimated but update disabled"
                         if freeze_rho else "learned KxN simplex" if is_lira
                         else "fixed KxN simplex (uniform)"),
        "training_data_mode": training_mode,
        "normalization_is_consistent": True,
        "q": q,
        "cycles": cycles,
        "outer_lr": outer_lr if is_lira else None,
        "outer_objective_scale": outer_objective_scale if is_lira else None,
        "m_replicates": m_replicates if is_lira else None,
        "sc_mechanism": "loo" if is_lira else None,
        "committed_updates": total_updates,
        "optimizer_steps_before": steps_before,
        "optimizer_steps_after": steps_after,
        "optimizer_digest_before": digest_before,
        "optimizer_digest_after": tx._optimizer_state_digest(states),
        "committed_rng_schedule": commit_schedule,
        "meta_rng_isolation": "fresh current-state clone per replica with explicit Python/NumPy/Torch RNG binding" if is_lira else "no meta replicas",
        "dual": dual_meta,
        "cycle_log": cycle_rows,
        "updates": all_updates,
        "rho_final": torch.softmax(z, dim=-1).tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
