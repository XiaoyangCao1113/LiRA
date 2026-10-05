"""Unit tests for the responsibility-weighted clipped PPO surrogate."""
import torch

from lira.ppo import PPOBatch, PPOConfig, PPOObjective


def _batch(reward, cost, old_log_probs=None):
    b = reward.numel()
    return PPOBatch(
        old_log_probs=torch.zeros(b) if old_log_probs is None else old_log_probs,
        reward_advantages=reward,
        cost_advantages=cost,
        active_masks=torch.ones(b),
        factor_weights=torch.ones(b),
        entropies=torch.zeros(b),
    )


def test_effective_advantage_uses_n_lambda_rho():
    reward = torch.tensor([1.0, -0.5, 2.0])
    cost = torch.tensor([[0.5, 1.0, -1.0], [2.0, 0.0, 1.0]])  # K=2 x B=3
    rho = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])  # K x N
    lam = torch.tensor([0.4, 1.5])
    out = PPOObjective(PPOConfig(clip_ratio=0.2)).evaluate(torch.zeros(3), _batch(reward, cost), rho, lam, agent_index=1)
    expected = reward - (3 * lam * rho[:, 1]) @ cost
    assert torch.allclose(out.combined_advantage, expected)
    # At ratio 1 the clipped surrogate equals the advantage itself.
    assert torch.allclose(out.actor, -expected.mean())


def test_ratio_is_clipped_for_positive_advantage():
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    batch = _batch(torch.tensor([1.0]), torch.zeros(1, 1))
    out = objective.evaluate(torch.tensor([1.0]), batch, torch.ones(1, 1), torch.zeros(1), 0)
    assert torch.allclose(out.actor, torch.tensor(-1.2))  # min(e * 1, 1.2 * 1)


def test_more_responsibility_means_more_cost_aversion():
    # With a positive cost advantage, raising agent 0's share lowers its
    # effective advantage, i.e. d(loss)/d(rho_0) > 0 at ratio 1.
    objective = PPOObjective()
    logits = torch.zeros(1, 2, requires_grad=True)
    batch = _batch(torch.zeros(4), torch.ones(1, 4))
    params = torch.zeros(1, requires_grad=True)
    loss = objective.loss_from_parameters(params, logits, batch, torch.tensor([1.0]), 0, lambda p: p.expand(4) * 0.0)
    (grad,) = torch.autograd.grad(loss, logits)
    assert grad[0, 0] > 0 and grad[0, 1] < 0
