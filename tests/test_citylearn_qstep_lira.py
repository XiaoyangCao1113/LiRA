"""Synthetic-tensor tests for the CityLearn q-step DU+LOO-SC (qstep_du_sc) estimator.

Uses synthetic tensors and fixed-tape batch/terminal providers, not the real
environment -- runs without the (heavy, optional) ``citylearn`` package
installed. Covers:

  - fresh-step sampling uses the updated (evolving) functional actor, not a
    fixed original actor (``test_batch_provider_receives_evolving_state``);
  - terminal data is disjoint from training data
    (``test_terminal_rollout_is_disjoint_from_training_batches``);
  - independent M replicas / LOO baseline combine correctly through the
    domain-agnostic ``lira.estimators.lira_gradient``
    (``test_qstep_lira_gradient_loo_sc_zero_when_terminal_welfare_identical``,
    ``test_qstep_lira_gradient_loo_sc_nonzero_when_terminal_welfare_varies``);
  - **decomposition pins**:
    ``du`` is exactly the terminal-return-weighted terminal score and carries
    no dependence on the training score's value
    (``test_du_equals_terminal_welfare_weighted_terminal_score``); ``score``
    is exactly the training-only sum and carries no terminal-score component
    (``test_score_is_training_only_excludes_terminal_score``); the q-step
    branch never reads an actor-loss value for DU
    (``test_collect_qstep_du_sc_replicate_never_reads_actor_loss``);
  - a synthetic q_meta=2 finite-difference check of the DU+SC surrogate's own
    internal autodiff consistency (``test_du_plus_score_gradient_matches_finite_difference``
    -- insufficient alone, since it only FDs the same surrogate scalar the
    module already differentiates; paired with
    ``test_qstep_du_sc_formula_matches_exact_finite_training_gradient_on_discrete_toy``,
    which compares the Eq. (gradient)/(terminal-estimator)/(loo) estimator's
    Monte-Carlo mean against an exactly enumerated finite-training response
    derivative on a tractable discrete toy);
  - no contamination of live state
    (``test_collect_qstep_du_sc_replicate_does_not_mutate_live_learner``);
  - q_meta validation (``test_collect_qstep_du_sc_replicate_requires_q_meta_at_least_two``);
  - nonzero gradient at initial uniform rho under heterogeneous lambda
    (``test_qstep_lira_gradient_nonzero_at_uniform_rho_with_heterogeneous_lambda``),
    without forcing nonzero in the symmetric (uniform-lambda) toy.
"""
from __future__ import annotations

import inspect

import torch

from lira.envs.citylearn.ppo_update import (
    Actor,
    Critic,
    CityLearnPPOBatch,
    _functional_actor_log_probs,
    action_scale_bias,
    sample_actions,
    uniform_lambda_per_agent,
)
from lira.envs.citylearn.qstep_lira import (
    QStepLiraReplicate,
    TerminalRollout,
    collect_qstep_du_sc_replicate,
    sample_actions_functional,
)
from lira.envs.citylearn.qstep_unroll import clone_functional_learner_state
from lira.estimators import lira_gradient
from lira.ppo import PPOConfig, PPOObjective
from lira.responsibility import SharedLambda

N_AGENTS = 3
OBS_DIM = 29
ACTION_DIM = 3
STATE_DIM = OBS_DIM * N_AGENTS
BATCH = 12
TERMINAL_STEPS = 5
LOW = torch.tensor([-1.0, -1.0, 0.0])
HIGH = torch.tensor([1.0, 1.0, 1.0])


def _build_learner(seed: int, lr: float = 3e-4, dtype: torch.dtype = torch.float32):
    torch.manual_seed(seed)
    actors = [Actor(OBS_DIM, ACTION_DIM).to(dtype) for _ in range(N_AGENTS)]
    reward_critic = Critic(STATE_DIM + N_AGENTS * ACTION_DIM).to(dtype)
    cost_critic = Critic(STATE_DIM + N_AGENTS * ACTION_DIM).to(dtype)
    optimizer = torch.optim.Adam(
        [p for actor in actors for p in actor.parameters()]
        + list(reward_critic.parameters()) + list(cost_critic.parameters()),
        lr=lr,
    )
    return actors, reward_critic, cost_critic, optimizer


def _synthetic_batch(actors, scale, bias, *, seed: int, dtype: torch.dtype = torch.float32) -> CityLearnPPOBatch:
    """A fixed, pre-recorded batch
    (real sampled actions/log-probs from ``actors``, not placeholder zeros)."""
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn(BATCH, N_AGENTS, OBS_DIM, generator=generator, dtype=dtype)
    with torch.no_grad():
        actions, old_log_probs = sample_actions(actors, obs, scale, bias)
    state = torch.randn(BATCH, STATE_DIM, generator=generator, dtype=dtype)
    reward_advantages = torch.randn(BATCH, N_AGENTS, generator=generator, dtype=dtype)
    cost_advantages = torch.randn(BATCH, generator=generator, dtype=dtype)
    value_target = torch.randn(BATCH, generator=generator, dtype=dtype)
    cost_target = torch.rand(BATCH, generator=generator, dtype=dtype)
    return CityLearnPPOBatch(
        observations=obs, actions=actions.detach(), state=state,
        old_log_probs=old_log_probs, reward_advantages=reward_advantages,
        cost_advantages=cost_advantages, value_target=value_target, cost_target=cost_target,
    )


def _functional_batch_from_state(actors, state, scale, bias, *, seed: int) -> CityLearnPPOBatch:
    """Build a batch by actually sampling on-policy from ``state``'s functional actor params."""
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn(BATCH, N_AGENTS, OBS_DIM, generator=generator)
    with torch.no_grad():
        actions, old_log_probs = sample_actions_functional(actors, state.actor_params, obs, scale, bias)
    synth_state = torch.randn(BATCH, STATE_DIM, generator=generator)
    reward_advantages = torch.randn(BATCH, N_AGENTS, generator=generator)
    cost_advantages = torch.randn(BATCH, generator=generator)
    value_target = torch.randn(BATCH, generator=generator)
    cost_target = torch.rand(BATCH, generator=generator)
    return CityLearnPPOBatch(
        observations=obs, actions=actions, state=synth_state,
        old_log_probs=old_log_probs, reward_advantages=reward_advantages,
        cost_advantages=cost_advantages, value_target=value_target, cost_target=cost_target,
    )


def _fixed_terminal_rollout(seed: int, welfare_value: float, steps: int = TERMINAL_STEPS) -> TerminalRollout:
    generator = torch.Generator().manual_seed(seed)
    # Repeat one 12-step tape so different terminal lengths differ only in
    # the number of score terms, not in the per-step observations/actions.
    base_obs = torch.randn(BATCH, N_AGENTS, OBS_DIM, generator=generator)
    scale, bias = action_scale_bias(LOW, HIGH)
    base_actions = torch.tanh(torch.randn(BATCH, N_AGENTS, ACTION_DIM, generator=generator)) * scale + bias
    repeats = (steps + BATCH - 1) // BATCH
    obs = base_obs.repeat(repeats, 1, 1)[:steps]
    actions = base_actions.repeat(repeats, 1, 1)[:steps]
    return TerminalRollout(observations=obs, actions=actions.detach(), welfare=torch.tensor(welfare_value))


def _snapshot_module_params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def _all_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def _collect_one_replicate(
    actors, reward_critic, cost_critic, optimizer, objective, lambda_per_agent, scale, bias, *,
    rho_logits_value, q_meta, welfare_value, tag, terminal_steps=TERMINAL_STEPS,
):
    state0 = clone_functional_learner_state(
        actors=actors, reward_critic=reward_critic, cost_critic=cost_critic, optimizer=optimizer,
    )
    recorded_states = []

    def batch_provider(t: int, state):
        recorded_states.append(state)
        # Sampled from THIS step's own (evolving) functional actor params,
        # not a fixed original-actor batch -- exercises the same
        # ``sample_actions_functional`` path a real driver would use.
        return _functional_batch_from_state(actors, state, scale, bias, seed=1000 * tag + t)

    def terminal_rollout_provider(final_state):
        recorded_states.append(("terminal", final_state))
        return _fixed_terminal_rollout(
            seed=9000 * tag, welfare_value=welfare_value, steps=terminal_steps,
        )

    replicate = collect_qstep_du_sc_replicate(
        actors=actors, reward_critic=reward_critic, cost_critic=cost_critic, state0=state0,
        batch_provider=batch_provider, terminal_rollout_provider=terminal_rollout_provider,
        rho_logits_value=rho_logits_value, lambda_per_agent=lambda_per_agent, objective=objective,
        n_agents=N_AGENTS, action_scale=scale, action_bias=bias, q_meta=q_meta,
    )
    return replicate, recorded_states


# --- fresh-step sampling uses the evolving functional actor ------------------------------


def test_batch_provider_receives_evolving_state():
    """Each step's ``batch_provider`` call gets a *different* (post-prior-step)
    functional state, not the same fixed clone reused at every step -- pins
    the "fresh trajectories under the evolving functional actor" requirement.
    """
    actors, rc, cc, opt = _build_learner(seed=1)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    rho_logits_value = torch.zeros(1, N_AGENTS)

    _replicate, recorded = _collect_one_replicate(
        actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
        rho_logits_value=rho_logits_value, q_meta=3, welfare_value=1.0, tag=1,
    )
    training_states = recorded[:3]
    assert len(training_states) == 3
    # Every step must be a distinct FunctionalLearnerState object (the prior
    # step's *output*, not a fixed original clone reused across all steps).
    assert len({id(s) for s in training_states}) == 3
    assert training_states[0].step == 0
    assert training_states[1].step == 1
    assert training_states[2].step == 2
    # And the actor parameter values themselves must genuinely differ step to step.
    p0 = training_states[0].actor_params[0]["net.0.weight"].detach()
    p1 = training_states[1].actor_params[0]["net.0.weight"].detach()
    assert not torch.equal(p0, p1)


# --- terminal data disjoint from training data --------------------------------------------


def test_terminal_rollout_is_disjoint_from_training_batches():
    """The terminal-rollout provider is called exactly once, after every
    training step, with the *final* state -- never mixed into a training
    batch, and never called before the last training step.
    """
    actors, rc, cc, opt = _build_learner(seed=2)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    rho_logits_value = torch.zeros(1, N_AGENTS)

    _replicate, recorded = _collect_one_replicate(
        actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
        rho_logits_value=rho_logits_value, q_meta=2, welfare_value=1.0, tag=2,
    )
    assert len(recorded) == 3  # 2 training-step calls + 1 terminal call
    assert recorded[0].step == 0
    assert recorded[1].step == 1
    tag, terminal_state = recorded[2]
    assert tag == "terminal"
    assert terminal_state.step == 2  # strictly after the last training step


# --- independent M replicas / LOO baseline (lira.estimators.lira_gradient) --


def test_qstep_lira_gradient_loo_sc_zero_when_terminal_welfare_identical():
    actors, rc, cc, opt = _build_learner(seed=3)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    rho_logits_value = torch.zeros(1, N_AGENTS)

    replicates = []
    for tag in range(3):
        replicate, _ = _collect_one_replicate(
            actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
            rho_logits_value=rho_logits_value, q_meta=2, welfare_value=3.5, tag=10 + tag,
        )
        replicates.append(replicate)

    welfares = [float(r.welfare) for r in replicates]
    assert len(set(welfares)) == 1

    _mean_grad, diagnostics = lira_gradient(replicates)
    assert diagnostics["loo_sc_mean"] == 0.0


def test_qstep_lira_gradient_loo_sc_nonzero_when_terminal_welfare_varies():
    actors, rc, cc, opt = _build_learner(seed=4)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    rho_logits_value = torch.zeros(1, N_AGENTS)

    replicates = []
    for tag, welfare_value in enumerate((0.0, 1.0, 2.0)):
        replicate, _ = _collect_one_replicate(
            actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
            rho_logits_value=rho_logits_value, q_meta=2, welfare_value=welfare_value, tag=20 + tag,
        )
        replicates.append(replicate)

    welfares = [float(r.welfare) for r in replicates]
    assert len(set(welfares)) == len(replicates)

    mean_grad, diagnostics = lira_gradient(replicates)
    assert torch.isfinite(mean_grad).all()
    assert diagnostics["loo_sc_mean"] != 0.0
    assert isinstance(replicates[0], QStepLiraReplicate)


# --- decomposition pins -------------------------------------------------------------------


def test_du_equals_terminal_welfare_weighted_terminal_score():
    """``replicate.du`` must equal ``welfare * terminal_score`` exactly, with

    no dependence on the training score's value -- i.e. this identity holds
    regardless of what the (generically nonzero) training score happens to
    be, which subsumes and is a strictly stronger pin than "at training
    score == 0, du equals the terminal-return-weighted terminal score": du
    has no training-score term to begin with, so its value can never be
    perturbed by one.
    """
    actors, rc, cc, opt = _build_learner(seed=11)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    rho_logits_value = torch.zeros(1, N_AGENTS)

    replicate, _ = _collect_one_replicate(
        actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
        rho_logits_value=rho_logits_value, q_meta=3, welfare_value=4.25, tag=30,
    )
    expected = replicate.welfare * torch.tensor(replicate.diagnostics["terminal_score"])
    assert torch.allclose(replicate.du.detach(), expected, atol=1e-6)
    # Training score is (generically) nonzero here, and du is unaffected by it.
    assert replicate.diagnostics["training_score"] != 0.0


def test_terminal_score_uses_training_time_denominator_when_lengths_differ():
    """A 48-step terminal score is four times its repeated 12-step tape.

    Training batches stay at 12 steps, so the common denominator is 12 for
    both DU and SC; averaging over each terminal rollout's own length would
    incorrectly make the two terminal scores equal.
    """
    actors, rc, cc, opt = _build_learner(seed=111)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    rho_logits_value = torch.zeros(1, N_AGENTS)

    torch.manual_seed(991)
    short, short_recorded = _collect_one_replicate(
        actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
        rho_logits_value=rho_logits_value, q_meta=2, welfare_value=2.0, tag=111,
        terminal_steps=12,
    )
    torch.manual_seed(991)
    long, _ = _collect_one_replicate(
        actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
        rho_logits_value=rho_logits_value, q_meta=2, welfare_value=2.0, tag=111,
        terminal_steps=48,
    )

    assert short.diagnostics["training_batch_steps"] == 12.0
    assert short.diagnostics["terminal_rollout_steps"] == 12.0
    assert long.diagnostics["training_batch_steps"] == 12.0
    assert long.diagnostics["terminal_rollout_steps"] == 48.0
    assert long.diagnostics["score_time_denominator"] == 12.0
    # At equal lengths, the new shared-denominator sum is numerically the old
    # terminal-time mean; this locks the 12-vs-12 behavior in particular.
    final_state = short_recorded[-1][1]
    terminal12 = _fixed_terminal_rollout(seed=9000 * 111, welfare_value=2.0, steps=12)
    old_equal_length_score = _functional_actor_log_probs(
        actors, final_state.actor_params, terminal12.observations, terminal12.actions, scale, bias,
    ).sum(dim=1).mean()
    assert torch.allclose(short.du.detach(), 2.0 * old_equal_length_score.detach(), atol=1e-6)
    assert torch.allclose(long.du.detach(), 4.0 * short.du.detach(), atol=2e-5, rtol=2e-5)


def test_score_is_training_only_excludes_terminal_score():
    """``replicate.score`` must equal the training-only sum exactly, with

    no terminal-score component -- i.e. this identity holds regardless of
    what the (generically nonzero) terminal score happens to be, which
    subsumes "at terminal score == 0, SC equals the LOO-weighted training
    score": score has no terminal-score term to begin with.
    """
    actors, rc, cc, opt = _build_learner(seed=12)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    rho_logits_value = torch.zeros(1, N_AGENTS)

    replicate, _ = _collect_one_replicate(
        actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
        rho_logits_value=rho_logits_value, q_meta=3, welfare_value=4.25, tag=31,
    )
    training_score = torch.tensor(replicate.diagnostics["training_score"])
    terminal_score = torch.tensor(replicate.diagnostics["terminal_score"])
    assert torch.allclose(replicate.score.detach(), training_score, atol=1e-6)
    assert terminal_score.item() != 0.0
    assert not torch.allclose(replicate.score.detach(), training_score + terminal_score, atol=1e-6)

    # And through the reused, unmodified lira_gradient, the LOO-SC term uses
    # exactly this training-only score (not training+terminal).
    other, _ = _collect_one_replicate(
        actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
        rho_logits_value=rho_logits_value, q_meta=3, welfare_value=1.0, tag=32,
    )
    _mean_grad, diagnostics = lira_gradient([replicate, other])
    # M=2, so each replicate's LOO baseline is exactly the other's welfare.
    other_training_score = torch.tensor(other.diagnostics["training_score"])
    loo_sc_replicate = float((replicate.welfare.detach() - other.welfare.detach()) * training_score)
    loo_sc_other = float((other.welfare.detach() - replicate.welfare.detach()) * other_training_score)
    expected_loo_sc_mean = (loo_sc_replicate + loo_sc_other) / 2
    assert abs(diagnostics["loo_sc_mean"] - expected_loo_sc_mean) < 1e-4


def test_collect_qstep_du_sc_replicate_never_reads_actor_loss():
    """No actor-loss derivative feeds the q-step DU/SC estimator directly.

    ``functional_transaction_step`` legitimately computes and differentiates
    through ``actor_loss``/``critic_loss`` internally as part of the learner
    *update* mechanics (that is what produces each step's Adam-updated
    parameters); this test pins that ``collect_qstep_du_sc_replicate``'s own
    body -- the DU/SC estimator construction -- never reads an actor-loss
    value itself (DU is the terminal-welfare-weighted terminal score, not
    ``-actor_loss`` at ``t == 0``).
    """
    source = inspect.getsource(collect_qstep_du_sc_replicate)
    body = source[source.index('"""', source.index('"""') + 3) + 3:]  # strip the docstring
    assert "actor_loss" not in body
    assert "citylearn_actor_critic_epoch_losses" not in body


def test_qstep_lira_gradient_nonzero_at_uniform_rho_with_heterogeneous_lambda():
    """Documents (does not merely assert) that initial uniform rho

    (``rho_logits == 0``) has a nonzero DU+LOO-SC gradient once the
    replicates are heterogeneous across agents (here: asymmetric
    ``lambda_per_agent``).
    The companion symmetric-toy tests above (uniform ``lambda_per_agent``)
    deliberately do NOT assert `mean_grad != 0` -- a perfectly symmetric
    setup may legitimately cancel to zero, and forcing nonzero there would
    be pinning an artifact of the toy, not a property of the estimator.
    """
    actors, rc, cc, opt = _build_learner(seed=13)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    lambda_per_agent = torch.tensor([0.5, 1.5, 3.0])  # heterogeneous across agents
    rho_logits_value = torch.zeros(1, N_AGENTS)  # initial uniform allocation

    replicates = []
    for tag, welfare_value in enumerate((0.0, 1.0, 2.0)):
        replicate, _ = _collect_one_replicate(
            actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
            rho_logits_value=rho_logits_value, q_meta=2, welfare_value=welfare_value, tag=40 + tag,
        )
        replicates.append(replicate)

    mean_grad, diagnostics = lira_gradient(replicates)
    assert torch.isfinite(mean_grad).all()
    assert torch.any(mean_grad != 0.0), (
        "expected a nonzero DU+LOO-SC gradient at uniform rho under heterogeneous "
        f"lambda_per_agent; got {mean_grad.tolist()} (diagnostics={diagnostics})"
    )


# --- synthetic q_meta=2 DU+SC finite-difference sign/value check -------------------------


def _du_plus_score(actors, rc, cc, opt, objective, lambda_per_agent, scale, bias, rho_logits_value, batches, terminal):
    """Returns ``(value, leaf)``: ``collect_qstep_du_sc_replicate`` clones
    ``rho_logits_value`` into its own independent leaf (matching
    the per-replicate-leaf convention needed so ``lira_gradient`` can differentiate each of M
    replicates' objectives independently) -- so gradients must be taken
    against the *returned* ``replicate.rho_logits`` leaf, not the external
    ``rho_logits_value`` tensor passed in (which is detached internally and
    therefore never appears in ``value``'s graph).
    """
    state0 = clone_functional_learner_state(actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt)
    replicate = collect_qstep_du_sc_replicate(
        actors=actors, reward_critic=rc, cost_critic=cc, state0=state0,
        batch_provider=lambda t, state: batches[t],
        terminal_rollout_provider=lambda state: terminal,
        rho_logits_value=rho_logits_value, lambda_per_agent=lambda_per_agent, objective=objective,
        n_agents=N_AGENTS, action_scale=scale, action_bias=bias, q_meta=2, actor_clip_norm=None,
    )
    return replicate.du + replicate.score, replicate.rho_logits


def test_du_plus_score_gradient_matches_finite_difference():
    """FD-checks the combined DU+SC quantity (``replicate.du + replicate.score``)
    that feeds ``lira_gradient``'s per-replicate objective (``du + (welfare -
    baseline).detach() * score``; the detached welfare-baseline scalar only
    rescales ``score``'s gradient, so checking ``du + score`` directly pins
    the same DU+SC gradient direction/value at unit weight). Float64,
    ``actor_clip_norm=None``, asymmetric ``lambda_per_agent`` and
    ``eps=1e-6`` central differences, mirroring
    ``test_citylearn_qstep_unroll.test_two_step_chain_gradient_matches_finite_difference``'s
    empirically-validated tolerance regime (two chained Adam steps through
    TanhNormal/clipped-PPO nonlinearities have real curvature at the
    naive ``eps=1e-3``-ish scale).
    """
    dtype = torch.float64
    actors, rc, cc, opt = _build_learner(seed=5, lr=1e-3, dtype=dtype)
    scale, bias = action_scale_bias(LOW, HIGH)
    scale, bias = scale.to(dtype), bias.to(dtype)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    lambda_per_agent = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    batches = [_synthetic_batch(actors, scale, bias, seed=60 + t, dtype=dtype) for t in range(2)]
    terminal = TerminalRollout(
        observations=torch.randn(TERMINAL_STEPS, N_AGENTS, OBS_DIM, dtype=dtype, generator=torch.Generator().manual_seed(70)),
        actions=(torch.tanh(torch.randn(TERMINAL_STEPS, N_AGENTS, ACTION_DIM, dtype=dtype, generator=torch.Generator().manual_seed(71))) * scale + bias).detach(),
        welfare=torch.tensor(1.0, dtype=dtype),
    )

    rho_logits_value = torch.zeros(1, N_AGENTS, dtype=dtype)
    value, leaf = _du_plus_score(actors, rc, cc, opt, objective, lambda_per_agent, scale, bias, rho_logits_value, batches, terminal)
    assert torch.isfinite(value)
    (analytic_grad,) = torch.autograd.grad(value, leaf)
    assert torch.isfinite(analytic_grad).all()
    assert torch.any(analytic_grad != 0.0)

    eps = 1e-6
    fd = torch.zeros_like(analytic_grad)
    for i in range(N_AGENTS):
        direction = torch.zeros_like(rho_logits_value)
        direction[0, i] = 1.0
        plus_rho = rho_logits_value + eps * direction
        minus_rho = rho_logits_value - eps * direction
        plus_value, _leaf = _du_plus_score(actors, rc, cc, opt, objective, lambda_per_agent, scale, bias, plus_rho, batches, terminal)
        minus_value, _leaf = _du_plus_score(actors, rc, cc, opt, objective, lambda_per_agent, scale, bias, minus_rho, batches, terminal)
        fd[0, i] = (plus_value.detach() - minus_value.detach()) / (2 * eps)

    assert torch.allclose(fd, analytic_grad, atol=2e-2, rtol=2e-2), (
        f"finite-difference {fd.tolist()} vs analytic {analytic_grad.tolist()} DU+SC gradient mismatch"
    )
    # Sign check: the FD and analytic gradients must agree in sign on every
    # nonzero analytic component (a coarser, more robust pin than the
    # elementwise allclose above, and the literal "sign" acceptance check).
    for i in range(N_AGENTS):
        if abs(float(analytic_grad[0, i])) > 1e-8:
            assert (float(fd[0, i]) > 0) == (float(analytic_grad[0, i]) > 0)


def test_collect_qstep_du_sc_replicate_requires_q_meta_at_least_two():
    actors, rc, cc, opt = _build_learner(seed=6)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    rho_logits_value = torch.zeros(1, N_AGENTS)
    state0 = clone_functional_learner_state(actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt)
    batch = _synthetic_batch(actors, scale, bias, seed=80)
    terminal = _fixed_terminal_rollout(seed=81, welfare_value=1.0)
    try:
        collect_qstep_du_sc_replicate(
            actors=actors, reward_critic=rc, cost_critic=cc, state0=state0,
            batch_provider=lambda t, state: batch, terminal_rollout_provider=lambda state: terminal,
            rho_logits_value=rho_logits_value, lambda_per_agent=lambda_per_agent, objective=objective,
            n_agents=N_AGENTS, action_scale=scale, action_bias=bias, q_meta=1,
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --- no contamination of live state --------------------------------------------------------


def test_collect_qstep_du_sc_replicate_does_not_mutate_live_learner():
    actors, rc, cc, opt = _build_learner(seed=7)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    rho_logits_value = torch.zeros(1, N_AGENTS)

    before_actor = [_snapshot_module_params(actor) for actor in actors]
    before_reward = _snapshot_module_params(rc)
    before_cost = _snapshot_module_params(cc)
    before_opt_exp_avg = {id(p): st["exp_avg"].clone() for p, st in opt.state.items() if "exp_avg" in st}

    _replicate, _recorded = _collect_one_replicate(
        actors, rc, cc, opt, objective, lambda_per_agent, scale, bias,
        rho_logits_value=rho_logits_value, q_meta=3, welfare_value=2.0, tag=90,
    )

    after_actor = [_snapshot_module_params(actor) for actor in actors]
    after_reward = _snapshot_module_params(rc)
    after_cost = _snapshot_module_params(cc)
    after_opt_exp_avg = {id(p): st["exp_avg"].clone() for p, st in opt.state.items() if "exp_avg" in st}

    for i in range(N_AGENTS):
        assert _all_equal(before_actor[i], after_actor[i])
    assert _all_equal(before_reward, after_reward)
    assert _all_equal(before_cost, after_cost)
    assert set(before_opt_exp_avg) == set(after_opt_exp_avg)


def test_sample_actions_functional_matches_live_actor_at_identical_params():
    """A sanity check that the functional sampler (used to build on-policy
    training/terminal batches from an evolving state) reduces to the live
    ``sample_actions`` when the functional params equal the live params --
    otherwise a subtle mismatch (e.g. a wrong tanh/bias convention) could
    silently bias every fresh on-policy batch.
    """
    actors, _rc, _cc, _opt = _build_learner(seed=8)
    scale, bias = action_scale_bias(LOW, HIGH)
    obs = torch.randn(BATCH, N_AGENTS, OBS_DIM, generator=torch.Generator().manual_seed(99))
    param_dicts = [dict(actor.named_parameters()) for actor in actors]

    torch.manual_seed(123)
    with torch.no_grad():
        live_actions, live_log_probs = sample_actions(actors, obs, scale, bias)
    torch.manual_seed(123)
    with torch.no_grad():
        functional_actions, functional_log_probs = sample_actions_functional(actors, param_dicts, obs, scale, bias)

    assert torch.allclose(live_actions, functional_actions, atol=1e-6)
    assert torch.allclose(live_log_probs, functional_log_probs, atol=1e-6)


# --- exact finite-training response derivative, discrete toy -----------------------------
#
# ``test_du_plus_score_gradient_matches_finite_difference`` above only finite-
# differences the *same surrogate scalar* (``du + score``) the module already
# differentiates -- that alone is insufficient evidence the
# surrogate is a correct (unbiased) estimator of the true finite-training
# response derivative nabla_phi V_q (the paper's finite-training gradient). A continuous
# TanhNormal actor's exact nabla_phi V_q has no tractable closed form (it
# requires integrating over the full trajectory sample space), so this toy
# swaps in a scalar Bernoulli-logit learner small enough to enumerate exactly
# (q_meta=2 means only 2x2=4 (a0, a1) training-data combinations), while
# reusing the *identical* du/score construction
# (``du = stopgrad(welfare) * terminal_score``, ``score = training-only
# score``) collect_qstep_du_sc_replicate now implements, combined through the
# same ``lira.estimators.lira_gradient`` LOO combiner used in
# production. ``theta_0`` is a fixed constant independent of ``phi`` (matching
# ``state0`` being cloned from the live learner, not a function of
# ``rho_logits``), so this toy also happens to exercise the appendix's
# "one-update horizon" edge case (d theta_0 / d phi = 0, so the
# first training step's score term s_0 is identically zero and only s_1
# contributes) without forcing it.


_TOY_PAYOFF_1 = 2.0
_TOY_PAYOFF_0 = -1.0


def _toy_update(theta_t: torch.Tensor, phi: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """One scalar analog of a differentiable functional learner step.

    ``action`` is raw, fixed sampled data (never differentiated through,
    matching the "raw actions fixed" convention); ``sigmoid(theta_t)`` is
    a deterministically-derived, fully differentiable quantity (matching
    ``functional_transaction_step``'s undetached actor/critic/Adam chain --
    nothing inside a learner update is artificially detached).
    """
    return theta_t + phi * (action - torch.sigmoid(theta_t))


def _toy_log_prob(theta_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(theta_t)
    return action * torch.log(p) + (1.0 - action) * torch.log(1.0 - p)


def _toy_welfare_of_action(action: torch.Tensor) -> torch.Tensor:
    return _TOY_PAYOFF_1 * action + _TOY_PAYOFF_0 * (1.0 - action)


def _discrete_toy_replicate(phi_value: torch.Tensor, *, theta0: float, seed: int) -> QStepLiraReplicate:
    """One q_meta=2 replicate, mirroring ``collect_qstep_du_sc_replicate``'s

    du/score construction exactly (``du = stopgrad(welfare) * terminal_score``,
    ``score`` = training-only score), on the scalar Bernoulli-logit toy.
    """
    generator = torch.Generator().manual_seed(seed)
    phi = phi_value.detach().clone().requires_grad_(True)
    theta_t = torch.tensor(theta0)

    training_score_terms = []
    for _ in range(2):  # q_meta = 2
        p_t = torch.sigmoid(theta_t)
        action = torch.bernoulli(p_t.detach(), generator=generator)
        training_score_terms.append(_toy_log_prob(theta_t, action))
        theta_t = _toy_update(theta_t, phi, action)
    training_score = torch.stack(training_score_terms).sum()

    p_terminal = torch.sigmoid(theta_t)
    terminal_action = torch.bernoulli(p_terminal.detach(), generator=generator)
    terminal_score = _toy_log_prob(theta_t, terminal_action)
    welfare = _toy_welfare_of_action(terminal_action).detach()

    du = welfare * terminal_score
    return QStepLiraReplicate(du=du, score=training_score, welfare=welfare, rho_logits=phi)


def _exact_discrete_toy_gradient(phi_value: torch.Tensor, theta0: float) -> torch.Tensor:
    """nabla_phi V_2(phi) computed exactly by enumerating all 4 (a0, a1) paths.

    ``V_2(phi) = sum_{a0,a1} P(a0) P(a1 | theta1(phi)) W(theta2(phi, a0, a1))``,
    with the terminal-rollout expectation folded into ``W``'s closed form
    (``E_b[c(b)] = c1 sigmoid(theta2) + c0 (1 - sigmoid(theta2))``) so only the
    two training actions need enumerating. Differentiating this single
    (fully differentiable) finite sum directly through autograd recovers
    Theorem (gradient)'s full ``g^DU + g^SC`` by the ordinary product rule --
    no REINFORCE/Monte-Carlo needed for this ground truth.
    """
    phi = phi_value.detach().clone().requires_grad_(True)
    theta_start = torch.tensor(theta0)
    total_V = torch.zeros(())
    for a0 in (0.0, 1.0):
        p0 = torch.sigmoid(theta_start)
        prob_a0 = p0 if a0 == 1.0 else (1.0 - p0)
        theta1 = _toy_update(theta_start, phi, torch.tensor(a0))
        for a1 in (0.0, 1.0):
            p1 = torch.sigmoid(theta1)
            prob_a1 = p1 if a1 == 1.0 else (1.0 - p1)
            theta2 = _toy_update(theta1, phi, torch.tensor(a1))
            p_terminal = torch.sigmoid(theta2)
            expected_terminal_welfare = _TOY_PAYOFF_1 * p_terminal + _TOY_PAYOFF_0 * (1.0 - p_terminal)
            total_V = total_V + prob_a0 * prob_a1 * expected_terminal_welfare
    (grad,) = torch.autograd.grad(total_V, phi)
    return grad


def test_qstep_du_sc_formula_matches_exact_finite_training_gradient_on_discrete_toy():
    """The Eq. (gradient)/(terminal-estimator)/(loo) estimator's Monte-Carlo

    mean (M=2000 replicates, through the unmodified ``lira_gradient`` LOO
    combiner) must land close to -- and share the sign of -- the exactly
    enumerated finite-training response derivative on the discrete toy
    (empirically: exact ~= 0.0667 at these parameters; M=2000 Monte-Carlo
    means observed in [0.050, 0.072] across independent seed blocks during
    development, well inside the tolerance below).
    """
    theta0 = -0.8
    phi0 = torch.tensor(1.2)
    exact_grad = _exact_discrete_toy_gradient(phi0, theta0)
    assert torch.isfinite(exact_grad).all()
    assert abs(float(exact_grad)) > 0.03  # not a degenerate near-zero point

    m_replicates = 2000
    replicates = [
        _discrete_toy_replicate(phi0, theta0=theta0, seed=7000 + i) for i in range(m_replicates)
    ]
    mc_grad, _diagnostics = lira_gradient(replicates)
    assert torch.isfinite(mc_grad).all()

    assert (float(mc_grad) > 0) == (float(exact_grad) > 0)
    assert abs(float(mc_grad) - float(exact_grad)) < 0.04
