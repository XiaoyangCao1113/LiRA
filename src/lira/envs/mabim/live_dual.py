"""Shared, dimensionless live-dual primitives for MABIM.

Each raw shared constraint ``C_k <= d_k`` is represented as the equivalent
dimensionless constraint ``C_k / d_k <= 1``.  The same normalization must be
used in both the actor Lagrangian and the projected dual update; normalizing
only the residual would change the effective optimization problem.
"""
from __future__ import annotations

import torch


def validate_budgets(budgets: torch.Tensor, n_constraints: int | None = None) -> torch.Tensor:
    budgets = torch.as_tensor(budgets, dtype=torch.float64)
    if budgets.ndim != 1:
        raise ValueError(f"budgets must be rank one, got shape {tuple(budgets.shape)}")
    if n_constraints is not None and budgets.numel() != n_constraints:
        raise ValueError(f"expected {n_constraints} budgets, got {budgets.numel()}")
    if not torch.isfinite(budgets).all() or not torch.all(budgets > 0):
        raise ValueError(f"budgets must be finite and strictly positive, got {budgets.tolist()}")
    return budgets


def normalize_cost_advantages(cost_adv: torch.Tensor, budgets: torch.Tensor) -> torch.Tensor:
    budgets = validate_budgets(budgets, cost_adv.shape[-1]).to(cost_adv)
    return cost_adv / budgets


def relative_residual(costs: torch.Tensor, budgets: torch.Tensor) -> torch.Tensor:
    budgets = validate_budgets(budgets, costs.numel()).to(costs)
    return costs / budgets - 1.0


def projected_dual_update(
    dual: torch.Tensor,
    costs: torch.Tensor,
    budgets: torch.Tensor,
    eta: float,
    *,
    dual_max: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    if dual_max <= 0:
        raise ValueError(f"dual_max must be positive, got {dual_max}")
    residual = relative_residual(costs, budgets)
    updated = torch.clamp(dual + float(eta) * residual, min=0.0, max=float(dual_max))
    return updated, residual


def effective_raw_multiplier(dual: torch.Tensor, budgets: torch.Tensor) -> torch.Tensor:
    """Multiplier on raw ``C_k`` induced by ``dual_k * C_k / d_k``."""
    budgets = validate_budgets(budgets, dual.shape[0]).to(dual)
    if dual.ndim == 1:
        return dual / budgets
    return dual / budgets[:, None]
