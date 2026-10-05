"""Unit tests for the floored responsibility simplex and the shared dual."""
import pytest
import torch

from lira.responsibility import SharedLambda, StaticSimplex, simplex_from_logits


def test_floored_simplex_sums_to_one_and_respects_floor():
    torch.manual_seed(0)
    logits = torch.randn(2, 5) * 4.0
    floor = 0.05
    rho = simplex_from_logits(logits, floor)
    assert torch.allclose(rho.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(rho >= floor - 1e-7)
    expected = floor + (1 - 5 * floor) * torch.softmax(logits, dim=-1)
    assert torch.allclose(rho, expected)


def test_zero_floor_recovers_softmax_and_is_shift_invariant():
    logits = torch.tensor([[0.3, -1.2, 2.0]])
    assert torch.allclose(simplex_from_logits(logits, 0.0), torch.softmax(logits, dim=-1))
    shifted = logits + 7.5
    assert torch.allclose(simplex_from_logits(shifted, 0.1), simplex_from_logits(logits, 0.1))


def test_invalid_floor_is_rejected():
    with pytest.raises(ValueError):
        simplex_from_logits(torch.zeros(1, 4), 0.3)  # N * floor > 1


def test_uniform_simplex_and_penalty_conservation():
    simplex = StaticSimplex.uniform(num_constraints=2, num_agents=4)
    assert torch.allclose(simplex.rho, torch.full((2, 4), 0.25))
    simplex.logits = torch.tensor([[1.0, 0.0, -1.0, 0.5], [0.0, 2.0, 0.0, 0.0]])
    lam = SharedLambda(torch.tensor([0.7, 1.3]))
    mu = simplex.effective_penalties(lam)
    # mu[k, i] = N * lambda_k * rho_ik and sum_i mu[k, i] = N * lambda_k.
    assert torch.allclose(mu, 4 * lam.values[:, None] * simplex.rho)
    assert torch.allclose(simplex.coefficient_totals(lam), 4 * lam.values)


def test_shared_lambda_projected_ascent():
    lam = SharedLambda(torch.tensor([0.5, 0.1]))
    lam.projected_update_(torch.tensor([1.0, -1.0]), step_size=0.2)
    assert torch.allclose(lam.values, torch.tensor([0.7, 0.0]))  # projected onto lambda >= 0
    lam.projected_update_(torch.tensor([3.0, 3.0]), step_size=0.0)  # zero step is a no-op
    assert torch.allclose(lam.values, torch.tensor([0.7, 0.0]))
    with pytest.raises(ValueError):
        lam.projected_update_(torch.tensor([1.0, 1.0]), step_size=-0.1)
    with pytest.raises(ValueError):
        SharedLambda(torch.tensor([-0.1]))


def test_tangent_update_keeps_column_and_moves_mass():
    simplex = StaticSimplex.uniform(num_constraints=1, num_agents=3)
    gradient = torch.tensor([[1.0, 0.0, -1.0]])
    simplex.apply_tangent_gradient_(gradient, step_size=0.5)
    rho = simplex.rho[0]
    assert torch.isclose(rho.sum(), torch.tensor(1.0))
    # The step descends the gradient: mass moves from agent 0 to agent 2.
    assert rho[0] < rho[1] < rho[2]
