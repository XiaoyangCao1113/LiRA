"""Simulator-free checks of the MetaDrive direct-unroll + LOO-SC estimator on synthetic tapes."""
import torch

from lira.envs.metadrive.direct_unroll import (
    MetaDriveDirectUnroll,
    MetaDriveDUConfig,
    MetaDriveRolloutTape,
    run_independent_du_sc,
)
from lira.envs.metadrive.learner import MetaDriveLearnerConfig, MetaDriveStaticLearner

N_AGENTS, OBS_DIM, ACTION_DIM, STEPS = 3, 4, 2, 5


def _learner() -> MetaDriveStaticLearner:
    torch.manual_seed(0)
    config = MetaDriveLearnerConfig(
        n_agents=N_AGENTS, obs_dim=OBS_DIM, action_dim=ACTION_DIM, horizon=STEPS, hidden=8, lambda_init=1.0,
    )
    learner = MetaDriveStaticLearner(env=None, config=config)
    # A non-uniform allocation so the lookahead gradient is not trivially symmetric.
    learner.simplex.logits = torch.tensor([[0.3, -0.2, 0.1]], dtype=learner.dtype)
    return learner


def _tape(generator: torch.Generator, dtype: torch.dtype) -> MetaDriveRolloutTape:
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype)

    return MetaDriveRolloutTape(
        obs=rand(STEPS, N_AGENTS, OBS_DIM),
        final_obs=rand(N_AGENTS, OBS_DIM),
        actions=rand(STEPS, N_AGENTS, ACTION_DIM),
        old_log_probs=rand(STEPS, N_AGENTS) - 2.0,
        rewards=rand(STEPS, N_AGENTS),
        costs=rand(STEPS, 1).abs(),
        dones=torch.zeros(STEPS, dtype=torch.bool),
    )


def test_loo_sc_gradient_matches_its_definition_and_is_tangent():
    learner = _learner()
    generator = torch.Generator().manual_seed(1)
    q, replicates = 2, 3
    updates = [tuple(_tape(generator, learner.dtype) for _ in range(q)) for _ in range(replicates)]
    evaluations = [_tape(generator, learner.dtype) for _ in range(replicates)]

    combined = run_independent_du_sc(learner, updates, evaluations, q=q, baseline="loo")

    transaction = MetaDriveDirectUnroll(learner, MetaDriveDUConfig(q=q, sampling_correction=True))
    singles = [transaction.run(u, e) for u, e in zip(updates, evaluations)]
    welfare = torch.tensor([item.terminal_welfare for item in singles], dtype=learner.dtype)
    loo = (welfare.sum() - welfare) / (replicates - 1)
    expected = torch.stack([item.direct_gradient for item in singles]).mean(0) + (
        (welfare - loo).reshape(-1, 1, 1) * torch.stack([item.score_gradient for item in singles])
    ).mean(0)

    assert combined.gradient.shape == (1, N_AGENTS)
    assert torch.isfinite(combined.gradient).all()
    torch.testing.assert_close(combined.gradient, expected)
    # Softmax logits are shift-invariant, so every per-constraint gradient row sums to zero.
    torch.testing.assert_close(combined.gradient.sum(-1), torch.zeros(1, dtype=learner.dtype), atol=1e-10, rtol=0)


def test_lookahead_does_not_mutate_live_learner():
    learner = _learner()
    generator = torch.Generator().manual_seed(2)
    before = [p.detach().clone() for p in learner.actors.parameters()]
    logits_before = learner.simplex.logits.clone()
    updates = tuple(_tape(generator, learner.dtype) for _ in range(2))
    MetaDriveDirectUnroll(learner, MetaDriveDUConfig(q=2)).run(updates, _tape(generator, learner.dtype))
    for old, new in zip(before, learner.actors.parameters()):
        assert torch.equal(old, new)
    assert torch.equal(logits_before, learner.simplex.logits)
