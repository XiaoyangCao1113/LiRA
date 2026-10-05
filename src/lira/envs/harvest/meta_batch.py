"""Leave-one-out (LOO) score-correction lookahead gradient for Harvest.

Given ``M`` independent replicates drawn from the same live learner state,
each an independent (update_tapes, terminal_tapes) draw through
:func:`lira.envs.harvest.transaction.execute_direct_unroll_sampling_correction`,
replicate ``i``'s welfare baseline is the mean welfare of the other ``M-1``
replicates.  Because that baseline never depends on replicate ``i``'s own
score, subtracting it keeps the estimator unbiased while reducing variance.

Sign convention: the returned gradient is a welfare-ascent direction.  The
training script negates it once before passing it to the descent primitive
``StaticSimplex.apply_tangent_gradient_`` (``logits -= step * tangent``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from lira.responsibility import simplex_from_logits

from .meta_unroll import ProbabilityTangentChart
from .transaction import (
    CategoricalTape,
    DirectUnrollResult,
    execute_direct_unroll_sampling_correction,
)

_CHART_DTYPE = torch.float64


@dataclass(frozen=True)
class MetaBatchReplicate:
    """One independent replicate's transaction plus its decomposed SC inputs."""

    result: DirectUnrollResult
    eta: torch.Tensor      # this replicate's own independent leaf, shape (1, N-1)
    welfare: torch.Tensor  # detached scalar; same value multiplied inside result.correction_objective
    score: torch.Tensor    # live joint training score; same value multiplied inside result.correction_objective


def build_chart(learner: Any) -> ProbabilityTangentChart:
    """Build a chart centered on ``learner``'s CURRENT rho, at eta == 0.

    The chart is built in float64 (see ``transaction._chart_logits_at_dtype``).
    """
    logits64 = learner.rho.logits.detach().to(_CHART_DTYPE)
    floor = float(learner.rho.floor)
    center = simplex_from_logits(logits64, floor)
    return ProbabilityTangentChart(center, floor)


def replicate_welfare_and_score(result: DirectUnrollResult) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the ``(welfare, score)`` pair whose product is ``result.correction_objective``.

    ``run_direct_unroll`` only returns the already-composed
    ``correction_objective = (welfare.detach() - baseline) * reduced_score``
    (``meta_unroll.compose_score_terms``).  A leave-one-out baseline needs
    welfare and score *separately* -- it replaces the baseline, not the
    already-multiplied product -- so both are recomputed here from the same
    public, already-tested dataclass fields ``DirectUnrollResult``
    exports.  The reconstruction is checked against the result's own
    ``correction_objective`` (formed with the entry point's default
    zero baseline) so a silent formula drift fails closed instead of
    silently mis-scoring.
    """
    welfare = torch.stack([item.welfare for item in result.terminal_scores]).mean().detach()
    score = torch.stack(result.behavior_joint_log_probs).sum()
    if not torch.allclose(welfare * score, result.correction_objective.detach(), atol=1e-5, rtol=1e-4):
        raise RuntimeError(
            "reconstructed welfare*score does not match result.correction_objective; SC formula drifted"
        )
    return welfare, score


def run_independent_meta_batch(
    learner: Any,
    chart: ProbabilityTangentChart,
    replicate_tapes: Sequence[tuple[Sequence[CategoricalTape], Sequence[CategoricalTape]]],
    *,
    env: Any | None = None,
) -> tuple[MetaBatchReplicate, ...]:
    """Run one independent DU+SC transaction per (update_tapes, terminal_tapes) pair.

    Every replicate calls ``execute_direct_unroll_sampling_correction``
    with its own fresh, independent ``eta`` leaf and its own independent
    tapes, against the SAME ``learner``/``chart`` (the fixed S0).  Each
    ``run_direct_unroll`` call internally builds a fresh
    ``FunctionalTransactionState`` snapshot from the current learner
    (``state=None`` default), so no replicate mutates the live learner
    modules or shares functional parameter state with any other replicate:
    this is already an independent-clone construction. ``replicate_tapes``
    must themselves carry independent seeds per replicate (this function
    does not reseed or generate tapes itself).
    """
    if len(replicate_tapes) < 2:
        raise ValueError("an independent meta batch requires at least two replicates")
    replicates: list[MetaBatchReplicate] = []
    for update_tapes, terminal_tapes in replicate_tapes:
        eta = torch.zeros((1, chart.tangent_dim), dtype=_CHART_DTYPE, requires_grad=True)
        _eta_gradient, result = execute_direct_unroll_sampling_correction(
            learner, chart, eta, update_tapes=update_tapes, terminal_tapes=terminal_tapes,
            env=env,
        )
        welfare, score = replicate_welfare_and_score(result)
        replicates.append(MetaBatchReplicate(result, eta, welfare, score))
    return tuple(replicates)


def leave_one_out_baselines(welfare_values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Return each replicate's cross-fitted baseline: mean welfare of the OTHERS.

    Every returned baseline is detached and depends only on the other
    ``M - 1`` replicates' welfare, never on replicate ``i``'s own -- the
    property that keeps this an unbiased (not merely lower-variance) control
    variate.
    """
    m = len(welfare_values)
    if m < 2:
        raise ValueError("a leave-one-out baseline requires at least two independent replicates")
    detached = [value.detach() for value in welfare_values]
    if any(value.ndim != 0 for value in detached):
        raise ValueError("welfare values must be scalar tensors")
    total = torch.stack(detached).sum()
    return tuple(((total - value) / (m - 1)).detach() for value in detached)


def cross_fitted_meta_batch_eta_gradient(replicates: Sequence[MetaBatchReplicate]) -> torch.Tensor:
    """Return the M-replicate-averaged DU+cross-fitted-SC eta-space gradient.

    Per replicate: ``corrected_i = direct_i + (welfare_i.detach() - baseline_i) * score_i``,
    differentiated once w.r.t. that replicate's own independent ``eta`` leaf
    (both ``welfare_i`` and ``baseline_i`` are detached scalars, so this is
    exactly ``grad(direct_i) + (welfare_i - baseline_i) * grad(score_i)`` by
    linearity). The M per-replicate gradients are then averaged.  Same
    additive structure as ``meta_unroll.compose_score_terms`` with only the
    baseline changed; see the module docstring for the sign convention.
    """
    welfare = [item.welfare for item in replicates]
    baselines = leave_one_out_baselines(welfare)
    per_replicate_grads: list[torch.Tensor] = []
    for item, baseline in zip(replicates, baselines, strict=True):
        corrected = item.result.direct_objective + (item.welfare.detach() - baseline) * item.score
        if not torch.isfinite(corrected):
            raise RuntimeError("meta-batch corrected objective is non-finite")
        grad = torch.autograd.grad(corrected, item.eta, retain_graph=False, allow_unused=True)[0]
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError("meta-batch replicate produced a disconnected or non-finite eta gradient")
        per_replicate_grads.append(grad)
    aggregate = torch.stack(per_replicate_grads).mean(dim=0)
    if not torch.isfinite(aggregate).all():
        raise RuntimeError("meta-batch aggregate eta gradient is non-finite")
    return aggregate


def embed_full_gradient(eta_gradient: torch.Tensor, chart: ProbabilityTangentChart, n_agents: int) -> torch.Tensor:
    """Embed a ``(1, N-1)`` eta-space gradient into the full ``(1, N)`` space.

    The chart rows form an orthonormal basis of the zero-sum tangent space,
    so ``eta_grad @ chart.basis`` is the corresponding tangent vector in
    logit/probability coordinates.
    """
    full_gradient = eta_gradient.detach() @ chart.basis
    if full_gradient.shape != (1, n_agents) or not torch.isfinite(full_gradient).all():
        raise ValueError("embedded meta-batch responsibility gradient has an invalid shape or is non-finite")
    return full_gradient


__all__ = [
    "MetaBatchReplicate",
    "build_chart",
    "replicate_welfare_and_score",
    "run_independent_meta_batch",
    "leave_one_out_baselines",
    "cross_fitted_meta_batch_eta_gradient",
    "embed_full_gradient",
]
