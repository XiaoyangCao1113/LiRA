"""Unit tests for the leave-one-out baseline and the DU + LOO-SC outer gradient."""
from types import SimpleNamespace

import pytest
import torch

from lira.estimators import apply_outer_ascent_step, leave_one_out_baselines, lira_gradient


def test_leave_one_out_baselines_exclude_own_welfare():
    baselines = leave_one_out_baselines([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(6.0)])
    assert [float(b) for b in baselines] == [4.0, 3.5, 1.5]
    with pytest.raises(ValueError):
        leave_one_out_baselines([torch.tensor(1.0)])


def _replicate(direct_weight, score_weight, welfare):
    logits = torch.zeros(1, 3, requires_grad=True)
    return SimpleNamespace(
        du=(torch.as_tensor(direct_weight) * logits).sum(),
        score=(torch.as_tensor(score_weight) * logits).sum(),
        welfare=torch.tensor(float(welfare)),
        rho_logits=logits,
    )


def test_lira_gradient_is_mean_of_direct_plus_centered_score_terms():
    a = [[1.0, 0.0, -1.0], [3.0, 0.0, -3.0]]
    s = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    welfare = [10.0, 4.0]  # LOO baselines: 4 and 10 -> centered welfare +6, -6
    grad, info = lira_gradient([_replicate(a[i], s[i], welfare[i]) for i in range(2)])
    expected = 0.5 * (torch.tensor([a[0]]) + 6.0 * torch.tensor([s[0]])
                      + torch.tensor([a[1]]) - 6.0 * torch.tensor([s[1]]))
    assert torch.allclose(grad, expected)
    assert info["welfare_mean"] == pytest.approx(7.0)


def test_equal_welfare_cancels_the_score_correction():
    reps = [_replicate([0.0, 0.0, 0.0], [5.0, -5.0, 0.0], 3.0) for _ in range(4)]
    grad, _ = lira_gradient(reps)
    assert torch.allclose(grad, torch.zeros(1, 3))


def test_outer_step_ascends_the_gradient():
    logits = torch.zeros(1, 3, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.1)
    apply_outer_ascent_step(logits, optimizer, torch.tensor([[1.0, 0.0, -2.0]]), grad_clip_norm=1.0)
    # Adam's first step moves each coordinate by lr * sign(gradient).
    assert torch.allclose(logits.detach(), torch.tensor([[0.1, 0.0, -0.1]]), atol=1e-6)
