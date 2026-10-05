"""Environment-free checks of the Harvest lookahead building blocks.

Melting Pot is imported lazily by the Harvest package, so these tests only
need PyTorch.
"""
import torch

from lira.envs.harvest.functional_adam import (
    FunctionalAdamState,
    SafeSqrt,
    _safe_sqrt_adam_update,
)
from lira.envs.harvest.meta_batch import embed_full_gradient, leave_one_out_baselines
from lira.envs.harvest.meta_unroll import ProbabilityTangentChart
from lira.responsibility import simplex_from_logits


def test_tangent_chart_basis_is_orthonormal_and_zero_sum():
    n_agents = 7
    center = simplex_from_logits(torch.zeros((1, n_agents), dtype=torch.float64))
    chart = ProbabilityTangentChart(center, 0.0)
    basis = chart.basis
    assert basis.shape == (n_agents - 1, n_agents)
    torch.testing.assert_close(basis @ basis.T, torch.eye(n_agents - 1, dtype=torch.float64))
    torch.testing.assert_close(basis.sum(dim=-1), torch.zeros(n_agents - 1, dtype=torch.float64))
    # eta = 0 maps back to the center allocation.
    eta = torch.zeros((1, n_agents - 1), dtype=torch.float64)
    torch.testing.assert_close(chart.rho(eta), center)
    full = embed_full_gradient(torch.ones((1, n_agents - 1), dtype=torch.float64), chart, n_agents)
    assert abs(float(full.sum())) < 1e-12


def test_leave_one_out_baselines_exclude_own_value():
    values = [torch.tensor(v) for v in (1.0, 2.0, 6.0)]
    baselines = leave_one_out_baselines(values)
    torch.testing.assert_close(torch.stack(baselines), torch.tensor([4.0, 3.5, 1.5]))


def test_safe_sqrt_forward_exact_and_zero_gradient_at_zero():
    v = torch.tensor([0.0, 4.0], requires_grad=True)
    out = SafeSqrt.apply(v)
    torch.testing.assert_close(out, torch.tensor([0.0, 2.0]))
    (grad,) = torch.autograd.grad(out.sum(), v)
    torch.testing.assert_close(grad, torch.tensor([0.0, 0.25]))


def test_functional_adam_step_matches_torch_adam():
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(5))
    optimizer = torch.optim.Adam([parameter], lr=3e-4)
    gradient = torch.randn(5)
    state = FunctionalAdamState.from_optimizer(optimizer, parameter)
    expected_parameter = parameter.detach().clone()
    updated, _ = _safe_sqrt_adam_update(state, expected_parameter, gradient)
    parameter.grad = gradient.clone()
    optimizer.step()
    torch.testing.assert_close(updated, parameter.detach())
