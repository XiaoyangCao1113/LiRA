"""Environment-free checks of the MABIM dual and LOO primitives."""
import pytest
import torch

from lira.envs.mabim.live_dual import (
    effective_raw_multiplier,
    normalize_cost_advantages,
    projected_dual_update,
)
from lira.envs.mabim.meta_gradient import leave_one_out_baselines
from lira.envs.mabim.transaction import _phase_seed


def test_projected_dual_update_uses_relative_residual_and_clips():
    dual = torch.tensor([0.5, 0.0], dtype=torch.float64)
    costs = torch.tensor([3.0, 1.0], dtype=torch.float64)
    budgets = torch.tensor([2.0, 4.0], dtype=torch.float64)
    updated, residual = projected_dual_update(dual, costs, budgets, 0.1, dual_max=0.52)
    assert torch.allclose(residual, torch.tensor([0.5, -0.75], dtype=torch.float64))
    # 0.5 + 0.1 * 0.5 = 0.55 is clipped to dual_max; 0 - 0.075 is projected to 0.
    assert torch.allclose(updated, torch.tensor([0.52, 0.0], dtype=torch.float64))


def test_cost_normalization_matches_effective_multiplier():
    budgets = torch.tensor([2.0, 4.0], dtype=torch.float64)
    adv = torch.ones(3, 5, 2, dtype=torch.float64)
    normalized = normalize_cost_advantages(adv, budgets)
    assert torch.allclose(normalized[..., 0], torch.full((3, 5), 0.5, dtype=torch.float64))
    dual = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert torch.allclose(effective_raw_multiplier(dual, budgets), torch.tensor([0.5, 0.5], dtype=torch.float64))
    with pytest.raises(ValueError):
        normalize_cost_advantages(adv, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_leave_one_out_baselines_exclude_own_welfare():
    welfare = [torch.tensor(v, dtype=torch.float64) for v in (1.0, 2.0, 6.0)]
    baselines = leave_one_out_baselines(welfare)
    assert [float(b) for b in baselines] == [4.0, 3.5, 1.5]
    with pytest.raises(ValueError):
        leave_one_out_baselines(welfare[:1])


def test_phase_seed_is_deterministic_and_phase_specific():
    assert _phase_seed(1101, "live-dual-commit", 0) == _phase_seed(1101, "live-dual-commit", 0)
    assert _phase_seed(1101, "live-dual-commit", 0) != _phase_seed(1101, "live-dual-meta", 0)
    assert 0 <= _phase_seed(1103, "live-dual-meta", 9, 5) < 2**31 - 1
