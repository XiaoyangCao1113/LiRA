"""Strictly bounded tanh-squashed Normal policy utilities."""
from __future__ import annotations
import math
import torch
from torch.distributions import Normal


class TanhNormal:
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, scale: float = 0.8):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.mean = mean
        self.log_std = log_std.clamp(-5.0, 2.0)
        self.scale = float(scale)
        self.base = Normal(self.mean, self.log_std.exp())

    def sample(self, deterministic: bool = False):
        pre = self.mean if deterministic else self.base.rsample()
        return self.scale * torch.tanh(pre)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        y = (action / self.scale).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre = torch.atanh(y)
        correction = math.log(self.scale) + torch.log(1.0 - y.square() + 1e-6)
        return self.base.log_prob(pre) - correction

    def entropy(self) -> torch.Tensor:
        # Exact transformed entropy has no closed form. This explicit base
        # Normal entropy is diagnostic-only and is not used in the loss.
        return self.base.entropy()

