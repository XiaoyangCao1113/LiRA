"""MetaDrive multi-agent Intersection (N=4, K=1): adapter, PPO learner, and LiRA arms.

Requires ``metadrive-simulator`` (imported lazily when the environment is built).
"""

from .direct_unroll import (
    MetaDriveDirectUnroll,
    MetaDriveDUConfig,
    MetaDriveDUResult,
    MetaDriveRolloutTape,
    capture_tapes,
    run_independent_du_sc,
)
from .env import MetaDriveIntersectionConfig, MetaDriveIntersectionEnv
from .learner import MetaDriveLearnerConfig, MetaDriveStaticLearner
from .online import ARMS, MetaDriveOnlineConfig, MetaDriveOnlineLearner, heldout_evaluate, run_heldout_episode

__all__ = [
    "ARMS",
    "MetaDriveDUConfig",
    "MetaDriveDUResult",
    "MetaDriveDirectUnroll",
    "MetaDriveIntersectionConfig",
    "MetaDriveIntersectionEnv",
    "MetaDriveLearnerConfig",
    "MetaDriveOnlineConfig",
    "MetaDriveOnlineLearner",
    "MetaDriveRolloutTape",
    "MetaDriveStaticLearner",
    "capture_tapes",
    "heldout_evaluate",
    "run_heldout_episode",
]
