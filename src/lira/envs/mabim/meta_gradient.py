"""Leave-one-out (LOO) score-corrected LiRA meta-gradient for MABIM.

Given ``M`` independent replicates of the q-step lookahead
(:func:`lira.envs.mabim.transaction.du_sc_outer_gradient`), each started from
the same learner state with its own RNG stream, replicate ``i``'s
score-correction baseline is the mean terminal welfare of the other ``M - 1``
replicates.  Because the baseline never uses replicate ``i``'s own welfare,
it is independent of replicate ``i``'s score and keeps the score-function
estimator unbiased.

The estimator is evaluated in streaming form: each replicate is
differentiated immediately, only its small rho-space direct and score
gradients plus scalar welfare are cached, and its graph is released before
the next replicate is built.  By linearity of differentiation this equals the
materialized LOO estimator while keeping a single live unroll in memory.
"""
from __future__ import annotations

import gc
import inspect
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from . import transaction


def _build_seeded_runner(build_runner: Callable[[int], object], seed: int):
    """Call a fresh-runner factory without weakening its seed contract.

    Factories may accept the seed positionally or as a keyword-only ``seed``.
    Inspect the signature before calling so either API is supported, and do
    not catch an exception raised *inside* the factory.
    """
    signature = inspect.signature(build_runner)
    try:
        signature.bind(seed=seed)
    except TypeError:
        signature.bind(seed)
        return build_runner(seed)
    return build_runner(seed=seed)


@dataclass(frozen=True)
class MabimMetaBatchReplicate:
    """One independent replicate's DU/SC inputs, decomposed for cross-fitting.

    ``objective`` is the raw (pre-gradient) DU scalar -- ``term.objective``,
    scoring the *final*, rho-dependent trained parameters -- averaged across
    replicates with no baseline correction by :func:`meta_batch_du`, since DU
    carries no score/sampling term. ``welfare`` and ``score`` are the
    separated inputs to the SC control variate: ``welfare`` is the detached
    terminal welfare and ``score`` is the live (rho-differentiable) sum of
    behavior log-probabilities across this replicate's ``q`` inner tapes.
    """

    z: torch.Tensor
    welfare: torch.Tensor
    score: torch.Tensor
    objective: torch.Tensor


def replicate_welfare_and_score(
    runner, dual: torch.Tensor, q: int, *, z_init: torch.Tensor | None = None,
    cost_budgets: torch.Tensor | None = None,
    dual_eta: float | None = None,
    dual_max: float = 100.0,
) -> MabimMetaBatchReplicate:
    """Run one q-step DU+SC block and return its decomposed SC inputs.

    ``du_sc_outer_gradient`` only returns the already-differentiated
    gradients, not the raw ``(welfare, score_sum)`` pair a leave-one-out
    baseline needs to recombine across replicates. This reads both off the
    *same* call's internal graph (via ``return_score_sum=True``) rather than
    reimplementing or re-running the q-step loop, so there is no separate
    code path that could silently drift from ``du_sc_outer_gradient``'s own
    formula. The reconstructed ``(welfare.detach() - baseline) * score_sum``
    gradient is checked against ``du_sc_outer_gradient``'s own
    ``correction_grad`` before returning; a mismatch raises rather than
    silently mis-scoring downstream replicates.
    """
    z, direct_grad, correction_grad, corrected_grad, cur, term, score_sum = transaction.du_sc_outer_gradient(
        runner, dual, q, z_init=z_init, return_score_sum=True,
        cost_budgets=cost_budgets, dual_eta=dual_eta, dual_max=dual_max,
    )
    baseline = torch.as_tensor(transaction.SC_SPEC.baseline, dtype=term.welfare.dtype)
    reconstructed_correction = (term.welfare.detach() - baseline) * score_sum
    if reconstructed_correction.requires_grad:
        # retain_graph=True: callers (e.g. a leave-one-out cross-fitted mean
        # over M replicates) still need to backward through this replicate's
        # ``score`` again later. This self-check must not be the one to
        # consume/free that shared graph.
        (reconstructed_grad,) = torch.autograd.grad(
            reconstructed_correction, z, retain_graph=True, allow_unused=True,
        )
        reconstructed_grad = torch.zeros_like(z) if reconstructed_grad is None else reconstructed_grad
    else:
        reconstructed_grad = torch.zeros_like(z)
    if not torch.allclose(reconstructed_grad, correction_grad, atol=1e-9):
        raise RuntimeError(
            "reconstructed welfare*score_sum gradient does not match "
            "du_sc_outer_gradient's own correction_grad; MABIM SC formula drifted"
        )
    return MabimMetaBatchReplicate(z=z, welfare=term.welfare.detach(), score=score_sum, objective=term.objective)


def leave_one_out_baselines(welfare_values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Return each replicate's cross-fitted baseline: mean welfare of the OTHERS.

    Every returned baseline is detached and depends only on the other
    ``M - 1`` replicates' welfare, never on replicate ``i``'s own -- the
    property that keeps this an unbiased (not merely lower-variance) control
    variate. Baseline ``i`` is statistically independent of score ``i`` by
    construction (different replicates, independent RNG), so subtracting it
    does not introduce the bias a same-batch/self baseline would.
    """
    m = len(welfare_values)
    if m < 2:
        raise ValueError("a leave-one-out baseline requires at least two independent replicates")
    detached = [value.detach() for value in welfare_values]
    if any(value.ndim != 0 for value in detached):
        raise ValueError("welfare values must be scalar tensors")
    total = torch.stack(detached).sum()
    return tuple(((total - value) / (m - 1)).detach() for value in detached)


def meta_batch_du_sc_outer_gradient(
    build_runner: Callable[[int], object],
    seeds: Sequence[int],
    dual: torch.Tensor,
    q: int,
    *,
    z_init: torch.Tensor | None = None,
    baseline: str = "loo",
    runners: Sequence[object] | None = None,
    cost_budgets: torch.Tensor | None = None,
    dual_eta: float | None = None,
    dual_max: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """LiRA outer gradient (mean DU + LOO score correction) over ``M`` replicates.

    ``baseline`` is ``"loo"`` for the paper estimator; ``"raw"`` (zero
    baseline) and ``"du_only"`` (no score correction) are ablations.  LOO needs only each
    replicate's detached welfare and score *gradient*, however.  We therefore
    differentiate one replicate immediately, cache only its small rho-space
    ``direct_grad``/``score_grad`` tensors plus scalar welfare/objective, and
    release that graph before constructing the next replicate.  The final
    LOO/raw formulas are algebraically identical by linearity of
    differentiation, while peak live graphs fall from ``M`` to one.
    """
    if baseline not in ("loo", "raw", "du_only"):
        raise ValueError(f"unknown baseline {baseline!r}; expected 'loo', 'raw', or 'du_only'")
    min_seeds = 1 if baseline == "du_only" else 2
    if len(seeds) < min_seeds:
        raise ValueError(
            f"baseline={baseline!r} requires at least {min_seeds} replicate seed(s), got {len(seeds)}"
        )
    if runners is not None:
        if len(runners) != len(seeds):
            raise ValueError(f"runners/seeds length mismatch: {len(runners)} versus {len(seeds)}")
        if len(runners) < min_seeds:
            raise ValueError(
                f"baseline={baseline!r} requires at least {min_seeds} runners, got {len(runners)}"
            )
        runner_iterable = runners
    else:
        # Generator, not a tuple: construct/process/release one independent
        # runner before building the next, preserving both RNG isolation and
        # the one-live-unroll memory bound.
        runner_iterable = (_build_seeded_runner(build_runner, seed) for seed in seeds)

    direct_grads: list[torch.Tensor] = []
    score_grads: list[torch.Tensor] = []
    welfare_values: list[torch.Tensor] = []
    objective_values: list[float] = []
    need_score = baseline != "du_only"
    live_kwargs = {}
    if cost_budgets is not None:
        live_kwargs["cost_budgets"] = cost_budgets
    if dual_eta is not None:
        live_kwargs.update({"dual_eta": dual_eta, "dual_max": dual_max})
    for runner in runner_iterable:
        replicate = replicate_welfare_and_score(
            runner, dual, q, z_init=z_init, **live_kwargs,
        )
        (direct_grad,) = torch.autograd.grad(
            replicate.objective, replicate.z, retain_graph=need_score, allow_unused=True,
        )
        direct_grad = torch.zeros_like(replicate.z) if direct_grad is None else direct_grad
        if need_score:
            (score_grad,) = torch.autograd.grad(
                replicate.score, replicate.z, retain_graph=False, allow_unused=True,
            )
            score_grad = torch.zeros_like(replicate.z) if score_grad is None else score_grad
        else:
            score_grad = torch.zeros_like(replicate.z)
        direct_grads.append(direct_grad.detach().clone())
        score_grads.append(score_grad.detach().clone())
        welfare_values.append(replicate.welfare.detach().clone())
        objective_values.append(float(replicate.objective.detach()))
        del replicate, direct_grad, score_grad
        gc.collect()

    direct_grad = torch.stack(direct_grads).mean(0)
    if baseline == "du_only":
        correction_grad = torch.zeros_like(direct_grad)
    elif baseline == "raw":
        correction_grad = torch.stack([
            welfare * score_grad
            for welfare, score_grad in zip(welfare_values, score_grads, strict=True)
        ]).mean(0)
    else:
        loo = leave_one_out_baselines(welfare_values)
        correction_grad = torch.stack([
            (welfare - baseline_value) * score_grad
            for welfare, baseline_value, score_grad in zip(welfare_values, loo, score_grads, strict=True)
        ]).mean(0)
    corrected_grad = direct_grad + correction_grad
    summary = {
        "sc_mechanism": baseline,
        "m_replicates": len(seeds),
        "replicate_seeds": list(seeds),
        "baseline_denominator": (len(seeds) - 1) if baseline == "loo" else None,
        "terminal_score_computed_once_per_replicate": True,
        "training_sc_uses_steps_before_terminal_tape": True,
        "training_sc_step_count_per_replicate": q,
        "per_replicate_welfare": [float(value) for value in welfare_values],
        "per_replicate_objective": objective_values,
        "aggregation_mode": "streaming_rho_gradients",
        "peak_live_unroll_graphs": 1,
        "live_dual": dual_eta is not None,
        "dual_eta": dual_eta,
        "cost_budgets": None if cost_budgets is None else torch.as_tensor(cost_budgets).tolist(),
    }
    del direct_grads, score_grads, welfare_values
    gc.collect()
    return direct_grad, correction_grad, corrected_grad, summary
