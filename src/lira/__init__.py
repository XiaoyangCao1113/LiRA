"""LiRA: Lagrangian Responsibility Allocation for shared constraints in MARL.

The package root only imports the domain-agnostic core (the floored
responsibility simplex, the shared dual variable, and the PPO surrogate).
Environment adapters live in :mod:`lira.envs` and are imported explicitly by
their entry points, so the core stays importable without any simulator.
"""

from .ppo import PPOBatch, PPOConfig, PPOObjective
from .responsibility import (
    SharedLambda,
    StaticSimplex,
    checkpoint_load,
    checkpoint_save,
    simplex_from_logits,
)

__version__ = "1.0.0"

__all__ = [
    "PPOBatch",
    "PPOConfig",
    "PPOObjective",
    "SharedLambda",
    "StaticSimplex",
    "checkpoint_load",
    "checkpoint_save",
    "simplex_from_logits",
]
