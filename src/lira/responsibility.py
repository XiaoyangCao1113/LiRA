"""Static responsibility and shared-dual state.

The floor construction is exact: ``floor + (1 - N*floor) * softmax(logits)``.
Consequently every component is at least ``floor`` and every column sums to
one, rather than merely being clamped and renormalised approximately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_SCOPE = "allocator_only"
_REQUIRED_CHECKPOINT_KEYS = frozenset(
    {"schema_version", "scope", "responsibility", "shared_lambda", "torch_rng_state"}
)


def _as_tensor(
    value: torch.Tensor | list[float], *, ndim: int, name: str, preserve_dtype: bool = False
) -> torch.Tensor:
    # Do not coerce restored state to the process default dtype.  Tensor inputs
    # carry the checkpoint's intentional dtype/device; Python lists naturally
    # use PyTorch's current default dtype.
    tensor = value if isinstance(value, torch.Tensor) and (preserve_dtype or value.is_floating_point()) else torch.as_tensor(
        value, dtype=None if isinstance(value, torch.Tensor) and value.is_floating_point() else torch.get_default_dtype()
    )
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tuple(tensor.shape)}")
    return tensor


def _validate_target_dtype(target_dtype: torch.dtype | None) -> None:
    if target_dtype is not None and (
        not isinstance(target_dtype, torch.dtype)
        or not torch.empty((), dtype=target_dtype).is_floating_point()
    ):
        raise ValueError("target_dtype must be a floating-point torch dtype")


def simplex_from_logits(logits: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    """Return a true-floor simplex along the final (agent) axis."""
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits need a nonempty final agent axis")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("responsibility logits must be finite floating-point values")
    n_agents = logits.shape[-1]
    if not torch.isfinite(torch.as_tensor(floor)) or floor < 0 or n_agents * floor > 1:
        raise ValueError("floor must satisfy 0 <= N * floor <= 1")
    scale = 1.0 - n_agents * floor
    return torch.softmax(logits, dim=-1) * scale + floor


@dataclass
class SharedLambda:
    """One nonnegative shared tightness variable per constraint, never per agent."""

    values: torch.Tensor

    def __post_init__(self) -> None:
        self.values = _as_tensor(self.values, ndim=1, name="lambda")
        if not self.values.is_floating_point() or not torch.isfinite(self.values).all():
            raise ValueError("shared lambda must be finite floating-point values")
        if torch.any(self.values < 0):
            raise ValueError("shared lambda must be nonnegative")

    @property
    def num_constraints(self) -> int:
        return int(self.values.numel())

    def projected_update_(self, gradient: torch.Tensor, step_size: float) -> None:
        gradient = _as_tensor(gradient, ndim=1, name="lambda gradient").to(self.values)
        if gradient.shape != self.values.shape or step_size < 0:
            raise ValueError("lambda update has invalid shape or step size")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite shared-lambda gradient")
        if step_size == 0:
            return
        self.values = torch.clamp(self.values + step_size * gradient, min=0.0)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"values": self.values.detach().clone()}

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        target_device: torch.device | str | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(state, Mapping) or "values" not in state:
            raise ValueError("invalid shared-lambda state")
        _validate_target_dtype(target_dtype)
        restored = _as_tensor(state["values"], ndim=1, name="lambda values", preserve_dtype=True)
        restore_device = self.values.device if target_device is None else torch.device(target_device)
        restored = restored.to(device=restore_device)
        if target_dtype is not None:
            restored = restored.to(dtype=target_dtype)
        if (
            restored.shape != self.values.shape
            or not restored.is_floating_point()
            or not torch.isfinite(restored).all()
            or torch.any(restored < 0)
        ):
            raise ValueError("invalid shared-lambda restore")
        self.values = restored.clone()


@dataclass
class StaticSimplex:
    """K independent static agent-simplex columns, represented as K x N rows."""

    logits: torch.Tensor
    floor: float = 0.0

    def __post_init__(self) -> None:
        self.logits = _as_tensor(self.logits, ndim=2, name="responsibility logits")
        # Validate the floor eagerly, including the N*floor boundary.
        simplex_from_logits(self.logits, self.floor)

    @classmethod
    def uniform(cls, num_constraints: int, num_agents: int, floor: float = 0.0) -> "StaticSimplex":
        if num_constraints < 1 or num_agents < 1:
            raise ValueError("K and N must both be positive")
        return cls(torch.zeros((num_constraints, num_agents)), floor=floor)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.logits.shape)  # type: ignore[return-value]

    @property
    def rho(self) -> torch.Tensor:
        return simplex_from_logits(self.logits, self.floor)

    def effective_penalties(self, shared_lambda: SharedLambda) -> torch.Tensor:
        """Return ``mu[k, i] = N lambda[k] rho[k, i]``."""
        if shared_lambda.num_constraints != self.shape[0]:
            raise ValueError("lambda K does not match responsibility K")
        return self.shape[1] * shared_lambda.values[:, None] * self.rho

    def coefficient_totals(self, shared_lambda: SharedLambda) -> torch.Tensor:
        """The fixed-policy coefficient-conservation totals, equal to N*lambda."""
        return self.effective_penalties(shared_lambda).sum(dim=-1)

    def apply_tangent_gradient_(self, gradient: torch.Tensor, step_size: float) -> None:
        """Update logits using a per-constraint tangent direction; fail loudly."""
        gradient = _as_tensor(gradient, ndim=2, name="rho-logit gradient").to(self.logits)
        if gradient.shape != self.logits.shape or step_size <= 0:
            raise ValueError("responsibility update has invalid shape or step size")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite responsibility gradient")
        tangent = gradient - gradient.mean(dim=-1, keepdim=True)
        self.logits = self.logits - step_size * tangent

    def state_dict(self) -> dict[str, Any]:
        return {"logits": self.logits.detach().clone(), "floor": self.floor}

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        target_device: torch.device | str | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(state, Mapping) or "logits" not in state or "floor" not in state:
            raise ValueError("invalid responsibility state")
        _validate_target_dtype(target_dtype)
        restored = _as_tensor(state["logits"], ndim=2, name="responsibility logits", preserve_dtype=True)
        floor = float(state["floor"])
        restore_device = self.logits.device if target_device is None else torch.device(target_device)
        restored = restored.to(device=restore_device)
        if target_dtype is not None:
            restored = restored.to(dtype=target_dtype)
        if restored.shape != self.logits.shape:
            raise ValueError("responsibility restore shape mismatch")
        if not torch.isfinite(torch.as_tensor(floor)):
            raise ValueError("responsibility floor must be finite")
        simplex_from_logits(restored, floor)
        self.logits, self.floor = restored.clone(), floor


def checkpoint_save(
    path: str | Path,
    responsibility: StaticSimplex,
    shared_lambda: SharedLambda,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save the allocator-only state needed by compact CPU probes.

    This is intentionally a partial checkpoint: it contains the responsibility
    simplex, shared lambda, optional allocator optimizer, and CPU Torch RNG. It
    does not claim to resume policy/critic parameters or their optimizers, nor
    Python/NumPy/CUDA RNG streams.
    """
    payload: dict[str, Any] = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "scope": _CHECKPOINT_SCOPE,
        "responsibility": responsibility.state_dict(),
        "shared_lambda": shared_lambda.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, Path(path))


def checkpoint_load(
    path: str | Path,
    responsibility: StaticSimplex,
    shared_lambda: SharedLambda,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    target_device: torch.device | str | None = None,
    target_dtype: torch.dtype | None = None,
) -> None:
    """Restore allocator state onto an explicit, portable target.

    ``target_device`` defaults to the current responsibility-logit device, so
    a checkpoint saved on another device is remapped during deserialization.
    ``target_dtype`` defaults to each saved tensor's dtype and, when supplied,
    casts both allocator tensors explicitly.  No process-wide dtype setting is
    changed by restore.
    """
    if target_device is None:
        target_device = responsibility.logits.device
    else:
        target_device = torch.device(target_device)
    if responsibility.logits.device != shared_lambda.values.device:
        raise ValueError("responsibility and shared lambda must share a device before restore")
    _validate_target_dtype(target_dtype)
    try:
        payload = torch.load(Path(path), map_location=target_device, weights_only=False)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint cannot be loaded onto target device {target_device}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    missing = _REQUIRED_CHECKPOINT_KEYS.difference(payload.keys())
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")
    if payload["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if payload["scope"] != _CHECKPOINT_SCOPE:
        raise ValueError("checkpoint scope is not allocator_only")
    if not isinstance(payload["torch_rng_state"], torch.Tensor) or payload["torch_rng_state"].dtype != torch.uint8:
        raise ValueError("checkpoint torch RNG state must be a uint8 tensor")
    responsibility.load_state_dict(
        payload["responsibility"], target_device=target_device, target_dtype=target_dtype
    )
    shared_lambda.load_state_dict(
        payload["shared_lambda"], target_device=target_device, target_dtype=target_dtype
    )
    try:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    except (RuntimeError, TypeError) as exc:
        raise ValueError("checkpoint has invalid Torch RNG state") from exc
    if optimizer is not None:
        if "optimizer" not in payload:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
