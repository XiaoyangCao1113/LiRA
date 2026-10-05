"""The single canonical frozen-batch clipped PPO surrogate.

The mixed derivative below is exact *for this fixed batch and graph*.  It is
not named a full game Jacobian because rollout distributions, other policy
responses, environment dynamics, and potentially shared critics are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import torch

from .responsibility import simplex_from_logits


class _SecondOrderZero(torch.autograd.Function):
    """A graph-carrying mathematical zero without any floating multiplication."""

    @staticmethod
    def forward(ctx, grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(input)

    @staticmethod
    def backward(ctx, grad_gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(grad_gradient), torch.zeros_like(grad_gradient)


class _ZeroTangent(torch.autograd.Function):
    """A finite zero with a differentiable zero tangent, even for ``input=inf``.

    ``input * 0`` would retain a higher-order edge, but recreates ``inf * 0``.
    The nested primitive instead represents the same mathematical zero with
    custom zero first and second tangents.  Its output is attached to ``input``
    when ``create_graph=True``; HVP/meta differentiation can therefore consume
    the finite second-order zero rather than failing on a detached gradient.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return torch.zeros_like(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        return _SecondOrderZero.apply(grad_output, input)


@dataclass(frozen=True)
class PPOConfig:
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.0
    rho_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.clip_ratio < 0 or self.entropy_coefficient < 0 or self.rho_floor < 0:
            raise ValueError("PPO config coefficients must be nonnegative")


@dataclass(frozen=True)
class PPOBatch:
    """Frozen tensors with B samples and K cost-advantage rows."""

    old_log_probs: torch.Tensor
    reward_advantages: torch.Tensor
    cost_advantages: torch.Tensor  # K x B
    active_masks: torch.Tensor
    factor_weights: torch.Tensor
    entropies: torch.Tensor

    def validate(self, num_constraints: int) -> int:
        vectors = (
            self.old_log_probs,
            self.reward_advantages,
            self.active_masks,
            self.factor_weights,
            self.entropies,
        )
        if any(t.ndim != 1 for t in vectors):
            raise ValueError("all PPO per-sample tensors must be vectors")
        batch_size = self.old_log_probs.numel()
        if any(t.numel() != batch_size for t in vectors):
            raise ValueError("PPO per-sample tensor lengths disagree")
        if self.cost_advantages.shape != (num_constraints, batch_size):
            raise ValueError("cost advantages must have shape K x B")
        if torch.any(self.active_masks < 0) or torch.any(self.factor_weights < 0):
            raise ValueError("masks and factor weights must be nonnegative")
        if self.active_masks.sum() <= 0:
            raise ValueError("at least one PPO sample must be active")
        return int(batch_size)


@dataclass(frozen=True)
class PPOBreakdown:
    total: torch.Tensor
    actor: torch.Tensor
    entropy: torch.Tensor
    combined_advantage: torch.Tensor


class PPOObjective:
    """One loss definition used for actor updates and all compact derivatives."""

    def __init__(self, config: PPOConfig = PPOConfig()) -> None:
        self.config = config

    @staticmethod
    def _weighted_mean(values: torch.Tensor, batch: PPOBatch) -> torch.Tensor:
        return (values * batch.factor_weights * batch.active_masks).sum() / batch.active_masks.sum()

    def evaluate(
        self,
        log_probs: torch.Tensor,
        batch: PPOBatch,
        rho: torch.Tensor,
        shared_lambda: torch.Tensor,
        agent_index: int,
    ) -> PPOBreakdown:
        if rho.ndim != 2 or shared_lambda.ndim != 1 or rho.shape[0] != shared_lambda.numel():
            raise ValueError("rho must be K x N and lambda must be K")
        k, n_agents = rho.shape
        if not 0 <= agent_index < n_agents or torch.any(shared_lambda < 0):
            raise ValueError("invalid agent index or negative shared lambda")
        batch.validate(k)
        if log_probs.shape != batch.old_log_probs.shape:
            raise ValueError("new log probabilities must be a B-vector")

        penalties = n_agents * shared_lambda * rho[:, agent_index]
        combined_advantage = batch.reward_advantages - torch.einsum("k,kb->b", penalties, batch.cost_advantages)
        # Preserve the literal historical PPO graph everywhere that its
        # exponential is representable.  This detail matters: at a clip tie,
        # ``torch.minimum`` splits its incoming derivative equally, while the
        # clipped side still differentiates through ``torch.clamp``.  Likewise
        # when A == 0, both products tie and their half-weighted A derivatives
        # induce the historical rho/lambda meta-gradient.  Rewriting the
        # expression in log space changes both conventions even though the
        # values agree.
        #
        # There is one exceptional domain, observed in practice: A >= 0 and
        # log(r) reaches the largest representable log(r).  The mathematical
        # PPO choice there is already the constant upper-clipped branch,
        # (1 + eps) A.  Materializing exp(log(r)) would create an *unselected*
        # infinity, followed by the undefined ExpBackward product 0 * inf.
        # Partition before exp only for that exceptional domain.  It is a safe
        # limiting extension, not ratio clipping: negative advantages retain
        # their unbounded selected branch, and NaNs stay on the historical
        # path so they propagate rather than being replaced.
        log_ratio = log_probs - batch.old_log_probs
        upper_ratio = torch.as_tensor(
            1.0 + self.config.clip_ratio, device=log_ratio.device, dtype=log_ratio.dtype,
        )
        max_log_ratio = torch.as_tensor(
            torch.finfo(log_ratio.dtype).max, device=log_ratio.device, dtype=log_ratio.dtype,
        ).log()
        overflowed_upper_limit = (combined_advantage >= 0.0) & (log_ratio >= max_log_ratio)
        ordinary = ~overflowed_upper_limit
        surrogate = torch.empty_like(combined_advantage)
        if bool(ordinary.any()):
            ordinary_ratio = torch.exp(log_ratio[ordinary])
            ordinary_clipped_ratio = torch.clamp(
                ordinary_ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio,
            )
            surrogate[ordinary] = torch.minimum(
                ordinary_ratio * combined_advantage[ordinary],
                ordinary_clipped_ratio * combined_advantage[ordinary],
            )
        if bool(overflowed_upper_limit.any()):
            limit_log_ratio = log_ratio[overflowed_upper_limit]
            surrogate[overflowed_upper_limit] = (
                upper_ratio * combined_advantage[overflowed_upper_limit]
                + _ZeroTangent.apply(limit_log_ratio)
            )
        actor = -self._weighted_mean(surrogate, batch)
        entropy = -self.config.entropy_coefficient * self._weighted_mean(batch.entropies, batch)
        return PPOBreakdown(total=actor + entropy, actor=actor, entropy=entropy, combined_advantage=combined_advantage)

    def loss_from_parameters(
        self,
        parameters: torch.Tensor,
        responsibility_logits: torch.Tensor,
        batch: PPOBatch,
        shared_lambda: torch.Tensor,
        agent_index: int,
        logprob_fn: Callable[[torch.Tensor], torch.Tensor],
        entropy_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Build the canonical loss and responsibility dependence in one graph."""
        rho = simplex_from_logits(responsibility_logits, self.config.rho_floor)
        # Supplying entropy_fn keeps entropy on the exact graph differentiated
        # below.  Frozen entropy values remain useful for controlled probes.
        graph_batch = batch if entropy_fn is None else replace(batch, entropies=entropy_fn(parameters))
        return self.evaluate(logprob_fn(parameters), graph_batch, rho, shared_lambda, agent_index).total

    def first_derivative(
        self,
        parameters: torch.Tensor,
        responsibility_logits: torch.Tensor,
        batch: PPOBatch,
        shared_lambda: torch.Tensor,
        agent_index: int,
        logprob_fn: Callable[[torch.Tensor], torch.Tensor],
        entropy_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        loss = self.loss_from_parameters(
            parameters, responsibility_logits, batch, shared_lambda, agent_index, logprob_fn, entropy_fn
        )
        # A frozen log-probability (and zero entropy coefficient) is a valid
        # probe whose loss need not depend on the policy parameters.  Treat
        # that inactive path as the mathematically correct zero derivative,
        # rather than asking autograd to differentiate a constant.
        if not parameters.requires_grad or not loss.requires_grad:
            return torch.zeros_like(parameters)
        gradient = torch.autograd.grad(
            loss, parameters, create_graph=True, allow_unused=True
        )[0]
        return torch.zeros_like(parameters) if gradient is None else gradient

    def mixed_derivative(
        self,
        parameters: torch.Tensor,
        responsibility_logits: torch.Tensor,
        batch: PPOBatch,
        shared_lambda: torch.Tensor,
        agent_index: int,
        logprob_fn: Callable[[torch.Tensor], torch.Tensor],
        entropy_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """d/d(logits) [d canonical loss / d parameters], retaining one graph."""
        first = self.first_derivative(
            parameters, responsibility_logits, batch, shared_lambda, agent_index, logprob_fn, entropy_fn
        )
        if not responsibility_logits.requires_grad:
            return torch.zeros(
                parameters.shape + responsibility_logits.shape,
                device=responsibility_logits.device,
                dtype=responsibility_logits.dtype,
            )
        rows = []
        for component in first.reshape(-1):
            if not component.requires_grad:
                rows.append(torch.zeros_like(responsibility_logits).reshape(-1))
                continue
            gradient = torch.autograd.grad(
                component,
                responsibility_logits,
                retain_graph=True,
                allow_unused=True,
            )[0]
            rows.append(
                torch.zeros_like(responsibility_logits).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
            )
        return torch.stack(rows).reshape(*parameters.shape, *responsibility_logits.shape)
