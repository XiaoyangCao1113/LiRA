"""Building blocks of the q-step lookahead estimator (DU + score correction).

* ``ProbabilityTangentChart`` parameterizes each responsibility column on the
  floored simplex through orthonormal tangent coordinates ``eta``.
* ``terminal_joint_welfare_score`` turns a terminal rollout into the welfare
  value and its joint-policy likelihood-ratio objective (the direct term).
* ``compose_score_terms`` adds the score-function correction for the inner
  training batches' dependence on the allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ProbabilityTangentChart:
    """K independent orthonormal probability charts for a general N-agent simplex.

    Each row k parameterizes ``rho[k, :]`` on the closed floor-simplex
    ``{p in R^N : p_i >= floor, sum_i p_i = 1}`` through an ``(N-1)``-dim
    tangent coordinate ``eta[k, :]``, using the orthonormal Helmert basis of
    the zero-sum hyperplane (``basis @ ones(N) == 0``, ``basis @ basis.T ==
    I``).  For ``N=2`` this basis is exactly the single direction
    ``(1/sqrt(2), -1/sqrt(2))`` used by the original 2-agent chart, and
    ``eta`` may still be passed as a bare ``(K,)`` vector (the legacy
    2-agent convention) in addition to the general ``(K, N-1)`` shape.
    """

    center: torch.Tensor
    floor: float

    def __post_init__(self) -> None:
        if self.center.ndim != 2 or self.center.shape[1] < 2:
            raise ValueError("chart requires K x N probabilities with N >= 2")
        n_agents = self.center.shape[1]
        if not torch.isfinite(self.center).all() or not np.isfinite(self.floor) or not 0.0 <= self.floor < 1.0 / n_agents:
            raise ValueError("invalid probability chart")
        if torch.any(self.center < self.floor) or not torch.allclose(
            self.center.sum(dim=-1), torch.ones(self.center.shape[0], dtype=self.center.dtype, device=self.center.device),
            atol=1e-12, rtol=0.0,
        ):
            raise ValueError("chart center is outside the true-floor simplex")

    @property
    def dimension(self) -> int:
        return int(self.center.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.center.shape[1])

    @property
    def tangent_dim(self) -> int:
        return self.n_agents - 1

    @property
    def expected_eta_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Valid ``eta`` shapes: the general one, plus the legacy N=2 vector."""
        general = (self.dimension, self.tangent_dim)
        if self.n_agents == 2:
            return ((self.dimension,), general)
        return (general,)

    @property
    def basis(self) -> torch.Tensor:
        """Orthonormal Helmert basis of the zero-sum hyperplane, shape (N-1, N).

        Row ``i`` (1-indexed) is ``(i ones, -i, zeros) / sqrt(i * (i + 1))``.
        Every row sums to zero (tangent to the simplex) and the rows are
        mutually orthonormal.  At ``N=2`` the single row is exactly
        ``(1/sqrt(2), -1/sqrt(2))``.
        """
        n_agents = self.n_agents
        rows = []
        for i in range(1, n_agents):
            scale = 1.0 / np.sqrt(i * (i + 1))
            rows.append([scale] * i + [-i * scale] + [0.0] * (n_agents - i - 1))
        return torch.tensor(rows, dtype=self.center.dtype, device=self.center.device)

    def _is_legacy_scalar_eta(self, eta: torch.Tensor) -> bool:
        return self.n_agents == 2 and eta.ndim == 1

    def rho(self, eta: torch.Tensor) -> torch.Tensor:
        if eta.device != self.center.device or eta.dtype != self.center.dtype or not torch.isfinite(eta).all() or eta.shape not in self.expected_eta_shapes:
            raise ValueError("tangent coordinate is invalid")
        if self._is_legacy_scalar_eta(eta):
            # Bit-identical to the original 2-agent-only formula: same basis
            # row, same elementwise broadcast, same multiplication order.
            rho = self.center + eta[:, None] * self.basis[0]
        else:
            rho = self.center + eta @ self.basis
        if torch.any(rho < self.floor - 1e-12):
            raise ValueError("tangent perturbation leaves the true-floor simplex")
        return rho

    def logits(self, eta: torch.Tensor) -> torch.Tensor:
        rho = self.rho(eta)
        scaled = (rho - self.floor) / (1.0 - self.n_agents * self.floor)
        logits = torch.log(scaled)
        return logits - logits.mean(dim=-1, keepdim=True)


@dataclass(frozen=True)
class TerminalScore:
    """Raw terminal welfare and its joint-policy likelihood-ratio objective."""

    welfare: torch.Tensor
    objective: torch.Tensor


@dataclass(frozen=True)
class SamplingCorrectionSpec:
    """Explicit target-preserving likelihood-ratio correction contract."""

    enabled: bool = True
    baseline: float = 0.0
    baseline_kind: str = "zero"
    score_reduction: str = "sum"
    normalization: str = "none"
    target: str = "q10_sum_t_mean_agent_reward_until_done_or_eval_horizon"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not np.isfinite(self.baseline):
            raise ValueError("sampling correction spec must have a finite boolean/scalar")
        if self.baseline_kind not in {"zero", "fixed_external"}:
            raise ValueError("baseline must be zero or a fixed external scalar")
        if self.baseline_kind == "zero" and self.baseline != 0.0:
            raise ValueError("zero baseline kind requires baseline=0")
        if self.score_reduction != "sum":
            raise ValueError("only unnormalised score-sum preserves the finite-training target")
        if self.normalization != "none":
            raise ValueError("sampling correction normalization is fixed to none")
        if self.target != "q10_sum_t_mean_agent_reward_until_done_or_eval_horizon":
            raise ValueError("sampling correction target is not the frozen q=10 welfare target")

    @classmethod
    def disabled(cls) -> "SamplingCorrectionSpec":
        return cls(enabled=False)


def compose_score_terms(
    direct: torch.Tensor, welfare: torch.Tensor, inner_scores: Sequence[torch.Tensor], *, reduction: str = "sum",
    detach_inner_scores: bool = False, correction_spec: SamplingCorrectionSpec | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One shared Direct/SC assembly with an explicit correction contract."""
    if not inner_scores or reduction not in {"sum", "mean"}:
        raise ValueError("score composition is invalid")
    spec = SamplingCorrectionSpec() if correction_spec is None else correction_spec
    if correction_spec is not None and spec.enabled and reduction != spec.score_reduction:
        raise ValueError("explicit sampling correction requires its declared score reduction")
    scores = torch.stack(tuple(inner_scores))
    if detach_inner_scores:
        scores = scores.detach()
    reduced = scores.sum() if reduction == "sum" else scores.mean()
    if not spec.enabled:
        correction = torch.zeros_like(direct)
    else:
        baseline = torch.as_tensor(spec.baseline, dtype=welfare.dtype, device=welfare.device)
        correction = (welfare.detach() - baseline) * reduced
    return direct, correction, direct + correction


def terminal_joint_welfare_score(batch: Any, *, eval_horizon: int | None = None) -> TerminalScore:
    """Return a joint-welfare policy-score objective for one fresh episode.

    Rewards are raw samples and are therefore detached.  Every agent's policy
    score receives the same social return-to-go, which retains cross-agent
    welfare effects in general-sum environments.  The terminal behavior
    log-probability itself remains differentiable through the final learner
    state.  This function deliberately has no environment/pathwise derivative.
    """
    if batch.behavior != "on_policy" or not batch.training_eligible:
        raise ValueError("terminal score requires on-policy provenance")
    if batch.old_log_probs.ndim != 2 or batch.rewards.shape != batch.old_log_probs.shape:
        raise ValueError("terminal batch shape is invalid")
    # Welfare target: sum_t mean_i reward[t,i], not a
    # discounted return.  Scores still sum across agents below.
    if eval_horizon is not None and (not isinstance(eval_horizon, int) or eval_horizon <= 0):
        raise ValueError("terminal eval_horizon must be a positive integer")
    limit = batch.rewards.shape[0] if eval_horizon is None else min(eval_horizon, batch.rewards.shape[0])
    # Retain only the valid prefix: an episode ends at the first done and any
    # fixed-shape collector padding has neither welfare nor likelihood mass.
    valid = torch.zeros(batch.rewards.shape[0], dtype=batch.rewards.dtype, device=batch.rewards.device)
    alive = True
    for index in range(limit):
        if not alive:
            break
        valid[index] = 1.0
        alive = not bool(batch.dones[index].detach().item())
    rewards = batch.rewards.detach().mean(dim=-1) * valid
    returns = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for index in range(rewards.numel() - 1, -1, -1):
        running = rewards[index] + (1.0 - batch.dones[index]) * running
        returns[index] = running
    joint_log_prob = batch.old_log_probs.sum(dim=-1) * valid
    welfare = rewards.sum()
    return TerminalScore(welfare, (returns * joint_log_prob).sum())


__all__ = [
    "ProbabilityTangentChart",
    "SamplingCorrectionSpec",
    "TerminalScore",
    "compose_score_terms",
    "terminal_joint_welfare_score",
]
