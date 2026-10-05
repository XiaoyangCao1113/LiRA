"""Typed observation/step records shared by the Harvest adapter and learner.

The learner consumes per-agent observation rows and categorical action masks;
the shared cost of a transition is carried in ``CostTransition.team_damage``
(``damage`` is the generic name of the shared constraint signal here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class StepObservation:
    agent_ids: tuple[str, ...]
    observations: Mapping[str, np.ndarray]
    world_state: np.ndarray
    action_masks: Mapping[str, np.ndarray]
    unit_types: np.ndarray
    unit_health: np.ndarray
    unit_alive: np.ndarray
    step: int

    @property
    def observation_matrix(self) -> np.ndarray:
        return np.stack(tuple(self.observations[a] for a in self.agent_ids), axis=0)

    @property
    def action_mask_matrix(self) -> np.ndarray:
        return np.stack(tuple(self.action_masks[a] for a in self.agent_ids), axis=0)


@dataclass(frozen=True)
class CostTransition:
    per_agent_damage: np.ndarray
    team_damage: float
    cumulative_team_damage: float
    death_events: np.ndarray
    unit_types: np.ndarray
    unit_health_before: np.ndarray
    unit_health_after: np.ndarray
    unit_alive_before: np.ndarray
    unit_alive_after: np.ndarray
    step_before: int
    step_after: int
    done: bool
    timeout: bool
    death_termination: bool
    battle_won: bool | None
    provenance: Any


@dataclass(frozen=True)
class AdapterStep:
    observation: StepObservation
    team_reward: float
    rewards: np.ndarray
    transition: CostTransition
    terminated: bool
    info: Mapping[str, Any]


__all__ = ["AdapterStep", "CostTransition", "StepObservation"]
