"""Functional (differentiable) Adam state for the categorical lookahead.

The lookahead differentiates the responsibility logits through several inner
PPO/critic/Adam steps, so each inner optimizer step is written functionally
over explicit parameter/moment tensors.

``d(sqrt(x))/dx`` is singular at ``x == 0``: an action category whose
gradient is exactly zero for every sample keeps an exactly-zero Adam second
moment, and differentiating ``exp_avg_sq.sqrt()`` there produces NaN.
``SafeSqrt`` leaves the forward Adam update unchanged (forward is exactly
``torch.sqrt``) and only defines the derivative of ``sqrt`` at ``v == 0`` to
be zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FunctionalAdamState:
    """Per-parameter Adam moments carried through a differentiable lookahead."""

    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: int
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    maximize: bool

    @classmethod
    def from_optimizer(cls, optimizer: torch.optim.Optimizer, parameter: torch.Tensor) -> "FunctionalAdamState":
        if len(optimizer.param_groups) != 1 or len(optimizer.param_groups[0]["params"]) != 1:
            raise ValueError("functional transaction requires one-parameter Adam groups")
        group = optimizer.param_groups[0]
        return cls.from_parameter_group(group, optimizer.state.get(parameter, {}), parameter)

    @classmethod
    def from_parameter_group(
        cls, group: Mapping[str, Any], state: Mapping[str, Any], parameter: torch.Tensor,
    ) -> "FunctionalAdamState":
        if group.get("amsgrad", False) or group.get("capturable", False) or group.get("differentiable", False):
            raise ValueError("functional transaction does not support this Adam variant")
        step_value = state.get("step", 0)
        step = int(step_value.item()) if isinstance(step_value, torch.Tensor) else int(step_value)
        return cls(
            state.get("exp_avg", torch.zeros_like(parameter)).detach().clone(),
            state.get("exp_avg_sq", torch.zeros_like(parameter)).detach().clone(), step,
            float(group["lr"]), float(group["betas"][0]), float(group["betas"][1]), float(group["eps"]),
            float(group["weight_decay"]), bool(group.get("maximize", False)),
        )


@dataclass(frozen=True)
class FunctionalParameterCollection:
    names: tuple[str, ...]
    values: tuple[torch.Tensor, ...]
    optimizers: tuple[FunctionalAdamState, ...]

    @classmethod
    def from_module(cls, module: nn.Module, optimizer: torch.optim.Optimizer) -> "FunctionalParameterCollection":
        named = tuple(module.named_parameters())
        groups = tuple(optimizer.param_groups)
        if len(groups) != 1 or len(groups[0]["params"]) != len(named) or any(
            left is not right for left, (_, right) in zip(groups[0]["params"], named, strict=True)
        ):
            raise ValueError("functional transaction requires one ordered Adam group per module")
        return cls(
            tuple(name for name, _ in named),
            tuple(value.detach().clone().requires_grad_(True) for _, value in named),
            tuple(FunctionalAdamState.from_parameter_group(groups[0], optimizer.state.get(value, {}), value) for _, value in named),
        )

    def mapping(self) -> dict[str, torch.Tensor]:
        return dict(zip(self.names, self.values, strict=True))


class SafeSqrt(torch.autograd.Function):
    """``sqrt`` whose forward is bit-for-bit ``torch.sqrt`` and whose backward
    is zero (instead of singular) exactly at ``v == 0``.

    The division ``0.5 / sqrt(v)`` is only ever evaluated at a masked-safe
    stand-in for ``v`` (``1`` wherever ``v == 0``, ``v`` otherwise), so the
    backward pass never actually divides by zero; the ``v == 0`` positions of
    the resulting gradient are then explicitly zeroed. Because ``backward``
    is written entirely in ordinary differentiable ``torch`` ops over the
    saved input, it composes correctly under a second ``torch.autograd.grad``
    call with ``create_graph=True`` (needed for the DU/SC eta-space
    gradient, which differentiates through this Adam step twice).
    """

    @staticmethod
    def forward(ctx: Any, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v)
        return torch.sqrt(v)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (v,) = ctx.saved_tensors
        is_zero = v == 0
        safe_v = torch.where(is_zero, torch.ones_like(v), v)
        grad_v = 0.5 * grad_output / torch.sqrt(safe_v)
        return torch.where(is_zero, torch.zeros_like(grad_v), grad_v)


def _safe_sqrt_adam_update(
    optimizer: FunctionalAdamState, parameter: torch.Tensor, gradient: torch.Tensor,
) -> tuple[torch.Tensor, FunctionalAdamState]:
    """Differentiable Adam step (``torch.optim.Adam`` arithmetic, no amsgrad).

    Same recurrence, bias correction, and parameter update as PyTorch's Adam,
    except the second moment's square root goes through ``SafeSqrt``.
    """
    if parameter.shape != gradient.shape or parameter.shape != optimizer.exp_avg.shape:
        raise ValueError("functional Adam parameter/gradient shape mismatch")
    gradient = -gradient if optimizer.maximize else gradient
    if optimizer.weight_decay != 0.0:
        gradient = gradient + optimizer.weight_decay * parameter
    step = optimizer.step + 1
    exp_avg = optimizer.exp_avg * optimizer.beta1 + gradient * (1.0 - optimizer.beta1)
    exp_avg_sq = optimizer.exp_avg_sq * optimizer.beta2 + gradient.square() * (1.0 - optimizer.beta2)
    bias_correction1 = 1.0 - optimizer.beta1**step
    bias_correction2 = 1.0 - optimizer.beta2**step
    denom = SafeSqrt.apply(exp_avg_sq) / np.sqrt(bias_correction2) + optimizer.eps
    update = (exp_avg / bias_correction1) / denom
    parameter_after = parameter - optimizer.lr * update
    return parameter_after, FunctionalAdamState(
        exp_avg, exp_avg_sq, step, optimizer.lr, optimizer.beta1, optimizer.beta2,
        optimizer.eps, optimizer.weight_decay, optimizer.maximize,
    )


def _safe_sqrt_collection_update(
    collection: FunctionalParameterCollection, gradients: tuple[torch.Tensor, ...],
) -> FunctionalParameterCollection:
    """Apply ``_safe_sqrt_adam_update`` to every parameter of a collection."""
    if len(gradients) != len(collection.values):
        raise ValueError("functional optimizer gradient count mismatch")
    updates = tuple(
        _safe_sqrt_adam_update(optimizer, parameter, gradient)
        for parameter, gradient, optimizer in zip(collection.values, gradients, collection.optimizers, strict=True)
    )
    return FunctionalParameterCollection(
        collection.names, tuple(value for value, _ in updates), tuple(state for _, state in updates),
    )


__all__ = [
    "FunctionalAdamState",
    "FunctionalParameterCollection",
    "SafeSqrt",
    "_safe_sqrt_adam_update",
    "_safe_sqrt_collection_update",
]
