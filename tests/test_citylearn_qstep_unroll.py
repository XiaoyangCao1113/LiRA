"""Synthetic-tensor tests for the q-step functional actor/critic/Adam kernel.

Uses synthetic tensors, not the real environment -- runs without the (heavy,
optional) ``citylearn`` package installed.

Covers:
  (a) one-step functional-versus-live parameter/optimizer-state value parity
      on identical fixed data (``test_one_step_functional_matches_live_update``);
  (b) a two-step chain differentiates w.r.t. rho logits and passes a
      finite-difference directional-gradient check
      (``test_two_step_chain_gradient_matches_finite_difference``);
  (c) clone isolation and deterministic fixed-tape replay
      (``test_clone_does_not_alias_or_mutate_live_learner``,
      ``test_fixed_tape_replay_is_deterministic``).
"""
from __future__ import annotations

import torch

from lira.envs.citylearn.ppo_update import (
    Actor,
    Critic,
    CityLearnPPOBatch,
    _functional_actor_log_probs,
    action_scale_bias,
    mutable_citylearn_ppo_update,
    sample_actions,
    uniform_lambda_per_agent,
)
from lira.envs.citylearn.qstep_unroll import (
    clone_functional_learner_state,
    functional_transaction_step,
    run_q_step_functional_chain,
)
from lira.ppo import PPOConfig, PPOObjective
from lira.responsibility import SharedLambda

N_AGENTS = 3
OBS_DIM = 29
ACTION_DIM = 3
STATE_DIM = OBS_DIM * N_AGENTS
BATCH = 12
LOW = torch.tensor([-1.0, -1.0, 0.0])
HIGH = torch.tensor([1.0, 1.0, 1.0])
PARITY_ATOL = 1e-6
PARITY_RTOL = 1e-5


def _build_actors_and_critics(seed: int, lr: float = 3e-4):
    torch.manual_seed(seed)
    actors = [Actor(OBS_DIM, ACTION_DIM) for _ in range(N_AGENTS)]
    reward_critic = Critic(STATE_DIM + N_AGENTS * ACTION_DIM)
    cost_critic = Critic(STATE_DIM + N_AGENTS * ACTION_DIM)
    optimizer = torch.optim.Adam(
        [p for actor in actors for p in actor.parameters()]
        + list(reward_critic.parameters()) + list(cost_critic.parameters()),
        lr=lr,
    )
    return actors, reward_critic, cost_critic, optimizer


def _synthetic_batch(actors, scale, bias, *, seed: int) -> CityLearnPPOBatch:
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn(BATCH, N_AGENTS, OBS_DIM, generator=generator)
    with torch.no_grad():
        actions, old_log_probs = sample_actions(actors, obs, scale, bias)
    state = torch.randn(BATCH, STATE_DIM, generator=generator)
    reward_advantages = torch.randn(BATCH, N_AGENTS, generator=generator)
    cost_advantages = torch.randn(BATCH, generator=generator)
    value_target = torch.randn(BATCH, generator=generator)
    cost_target = torch.rand(BATCH, generator=generator)
    return CityLearnPPOBatch(
        observations=obs, actions=actions.detach(), state=state,
        old_log_probs=old_log_probs.detach(), reward_advantages=reward_advantages,
        cost_advantages=cost_advantages, value_target=value_target, cost_target=cost_target,
    )


def _snapshot_module_params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def _all_close(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], *, atol: float, rtol: float) -> bool:
    return all(torch.allclose(a[k], b[k], atol=atol, rtol=rtol) for k in a)


def _all_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


# --- (a) one-step functional-versus-live parity -----------------------------------------


def test_one_step_functional_matches_live_update():
    """Cloning after a warm-up step (nonzero Adam moments), then taking one
    live step and one functional step from that same clone on the identical
    fresh batch, must produce numerically matching actor+critic parameters.
    """
    actors, rc, cc, opt = _build_actors_and_critics(seed=1)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    rho_logits = torch.zeros(1, N_AGENTS)
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)

    warmup_batch = _synthetic_batch(actors, scale, bias, seed=10)
    mutable_citylearn_ppo_update(
        actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt,
        rho_logits=rho_logits, lambda_per_agent=lambda_per_agent, batch=warmup_batch,
        objective=objective, epochs=1, n_agents=N_AGENTS, action_dim=ACTION_DIM,
        action_scale=scale, action_bias=bias,
    )

    state1 = clone_functional_learner_state(actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt)
    assert state1.step == 1

    step_batch = _synthetic_batch(actors, scale, bias, seed=11)

    mutable_citylearn_ppo_update(
        actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt,
        rho_logits=rho_logits, lambda_per_agent=lambda_per_agent, batch=step_batch,
        objective=objective, epochs=1, n_agents=N_AGENTS, action_dim=ACTION_DIM,
        action_scale=scale, action_bias=bias,
    )
    live_actor_params = [_snapshot_module_params(actor) for actor in actors]
    live_reward_critic_params = _snapshot_module_params(rc)
    live_cost_critic_params = _snapshot_module_params(cc)

    new_state = functional_transaction_step(
        state1, actors=actors, reward_critic=rc, cost_critic=cc, batch=step_batch,
        rho_logits=rho_logits.detach(), lambda_per_agent=lambda_per_agent, objective=objective,
        n_agents=N_AGENTS, action_scale=scale, action_bias=bias,
    )

    for i, actor_params in enumerate(new_state.actor_params):
        functional_detached = {k: v.detach() for k, v in actor_params.items()}
        assert _all_close(functional_detached, live_actor_params[i], atol=PARITY_ATOL, rtol=PARITY_RTOL), (
            f"actor {i} functional/live one-step parity gap exceeds atol={PARITY_ATOL}, rtol={PARITY_RTOL}"
        )
    functional_reward = {k: v.detach() for k, v in new_state.reward_critic_params.items()}
    functional_cost = {k: v.detach() for k, v in new_state.cost_critic_params.items()}
    assert _all_close(functional_reward, live_reward_critic_params, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    assert _all_close(functional_cost, live_cost_critic_params, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    assert new_state.step == 2


# --- (b) two-step chain differentiates w.r.t. rho_logits --------------------------------


def _terminal_score(actors, rc, cc, opt, batches, rho_logits_value, lambda_per_agent, objective, scale, bias):
    """Rebuild a fresh clone from the (untouched) live learner and run the q=2 chain."""
    state0 = clone_functional_learner_state(actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt)
    final_state = run_q_step_functional_chain(
        state0, actors=actors, reward_critic=rc, cost_critic=cc,
        batch_provider=lambda t: batches[t], q_meta=2, rho_logits=rho_logits_value,
        lambda_per_agent=lambda_per_agent, objective=objective, n_agents=N_AGENTS,
        action_scale=scale, action_bias=bias, actor_clip_norm=None,
    )
    return _functional_actor_log_probs(
        actors, final_state.actor_params, batches[-1].observations, batches[-1].actions, scale, bias,
    ).sum(dim=1).mean()


def test_two_step_chain_gradient_matches_finite_difference():
    """Central finite differences need ``eps`` far smaller than the ``1e-3``-ish
    scale that would suffice for a linear function: two chained Adam steps
    through ``TanhNormal``/clipped-PPO nonlinearities have real curvature at
    this scale (confirmed empirically -- ``eps=1e-3..1e-5`` central
    differences visibly fail to converge here, while ``eps<=1e-6`` agrees
    with the analytic gradient to <1%), so this test runs in float64 and at
    ``eps=1e-6`` rather than float32's usual ``~1e-3`` default.
    """
    torch.manual_seed(2)
    actors = [Actor(OBS_DIM, ACTION_DIM).double() for _ in range(N_AGENTS)]
    reward_critic = Critic(STATE_DIM + N_AGENTS * ACTION_DIM).double()
    cost_critic = Critic(STATE_DIM + N_AGENTS * ACTION_DIM).double()
    optimizer = torch.optim.Adam(
        [p for actor in actors for p in actor.parameters()]
        + list(reward_critic.parameters()) + list(cost_critic.parameters()),
        lr=1e-3,
    )
    scale, bias = action_scale_bias(LOW, HIGH)
    scale, bias = scale.double(), bias.double()
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    # Deliberately asymmetric per-agent duals (not the Uniform arm's equal
    # lambdas): at rho_logits == 0 with equal lambdas, the direct penalty
    # term's rho-sensitivity cancels exactly (Uniform "recovers the shared
    # multiplier" regardless of rho -- see ppo_update's module
    # docstring), which is a degenerate point for a gradient-check test.
    lambda_per_agent = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    def _double_batch(seed: int) -> CityLearnPPOBatch:
        generator = torch.Generator().manual_seed(seed)
        obs = torch.randn(BATCH, N_AGENTS, OBS_DIM, generator=generator, dtype=torch.float64)
        with torch.no_grad():
            actions, old_log_probs = sample_actions(actors, obs, scale, bias)
        state = torch.randn(BATCH, STATE_DIM, generator=generator, dtype=torch.float64)
        reward_advantages = torch.randn(BATCH, N_AGENTS, generator=generator, dtype=torch.float64)
        cost_advantages = torch.randn(BATCH, generator=generator, dtype=torch.float64)
        value_target = torch.randn(BATCH, generator=generator, dtype=torch.float64)
        cost_target = torch.rand(BATCH, generator=generator, dtype=torch.float64)
        return CityLearnPPOBatch(
            observations=obs, actions=actions.detach(), state=state,
            old_log_probs=old_log_probs.detach(), reward_advantages=reward_advantages,
            cost_advantages=cost_advantages, value_target=value_target, cost_target=cost_target,
        )

    batches = [_double_batch(seed=20 + t) for t in range(2)]

    rho_logits = torch.zeros(1, N_AGENTS, dtype=torch.float64, requires_grad=True)
    score = _terminal_score(actors, reward_critic, cost_critic, optimizer, batches, rho_logits, lambda_per_agent, objective, scale, bias)
    assert torch.isfinite(score)
    (analytic_grad,) = torch.autograd.grad(score, rho_logits)
    assert torch.isfinite(analytic_grad).all()
    assert torch.any(analytic_grad != 0.0)

    eps = 1e-6
    fd = torch.zeros_like(analytic_grad)
    for i in range(N_AGENTS):
        direction = torch.zeros_like(rho_logits)
        direction[0, i] = 1.0
        with torch.no_grad():
            plus_rho = rho_logits + eps * direction
            minus_rho = rho_logits - eps * direction
        plus_score = _terminal_score(
            actors, reward_critic, cost_critic, optimizer, batches, plus_rho, lambda_per_agent, objective, scale, bias,
        )
        minus_score = _terminal_score(
            actors, reward_critic, cost_critic, optimizer, batches, minus_rho, lambda_per_agent, objective, scale, bias,
        )
        fd[0, i] = (plus_score.detach() - minus_score.detach()) / (2 * eps)

    assert torch.allclose(fd, analytic_grad, atol=2e-2, rtol=2e-2), (
        f"finite-difference {fd.tolist()} vs analytic {analytic_grad.tolist()} gradient mismatch"
    )


def test_two_step_chain_requires_q_meta_at_least_two():
    actors, rc, cc, opt = _build_actors_and_critics(seed=3)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    batches = [_synthetic_batch(actors, scale, bias, seed=30)]
    rho_logits = torch.zeros(1, N_AGENTS, requires_grad=True)
    state0 = clone_functional_learner_state(actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt)
    try:
        run_q_step_functional_chain(
            state0, actors=actors, reward_critic=rc, cost_critic=cc,
            batch_provider=lambda t: batches[0], q_meta=1, rho_logits=rho_logits,
            lambda_per_agent=lambda_per_agent, objective=objective, n_agents=N_AGENTS,
            action_scale=scale, action_bias=bias,
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --- (c) clone isolation and deterministic fixed-tape replay ----------------------------


def test_clone_does_not_alias_or_mutate_live_learner():
    actors, rc, cc, opt = _build_actors_and_critics(seed=4)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)

    warmup_batch = _synthetic_batch(actors, scale, bias, seed=40)
    rho_logits = torch.zeros(1, N_AGENTS)
    mutable_citylearn_ppo_update(
        actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt,
        rho_logits=rho_logits, lambda_per_agent=lambda_per_agent, batch=warmup_batch,
        objective=objective, epochs=1, n_agents=N_AGENTS, action_dim=ACTION_DIM,
        action_scale=scale, action_bias=bias,
    )

    before_actor = [_snapshot_module_params(actor) for actor in actors]
    before_reward = _snapshot_module_params(rc)
    before_cost = _snapshot_module_params(cc)
    before_opt_exp_avg = {
        id(p): st["exp_avg"].clone() for p, st in opt.state.items() if "exp_avg" in st
    }

    state = clone_functional_learner_state(actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt)

    # No shared storage between the clone and the live modules/optimizer state.
    for i, actor in enumerate(actors):
        for name, p in actor.named_parameters():
            assert state.actor_params[i][name].data_ptr() != p.data_ptr()
            assert state.exp_avg[f"actor{i}.{name}"].data_ptr() != opt.state[p]["exp_avg"].data_ptr()

    batches = [_synthetic_batch(actors, scale, bias, seed=41 + t) for t in range(2)]
    rho_logits_leaf = torch.zeros(1, N_AGENTS, requires_grad=True)
    run_q_step_functional_chain(
        state, actors=actors, reward_critic=rc, cost_critic=cc,
        batch_provider=lambda t: batches[t], q_meta=2, rho_logits=rho_logits_leaf,
        lambda_per_agent=lambda_per_agent, objective=objective, n_agents=N_AGENTS,
        action_scale=scale, action_bias=bias,
    )

    after_actor = [_snapshot_module_params(actor) for actor in actors]
    after_reward = _snapshot_module_params(rc)
    after_cost = _snapshot_module_params(cc)
    after_opt_exp_avg = {
        id(p): st["exp_avg"].clone() for p, st in opt.state.items() if "exp_avg" in st
    }

    for i in range(N_AGENTS):
        assert _all_equal(before_actor[i], after_actor[i])
    assert _all_equal(before_reward, after_reward)
    assert _all_equal(before_cost, after_cost)
    assert set(before_opt_exp_avg) == set(after_opt_exp_avg)
    for key in before_opt_exp_avg:
        assert torch.equal(before_opt_exp_avg[key], after_opt_exp_avg[key])


def test_fixed_tape_replay_is_deterministic():
    actors, rc, cc, opt = _build_actors_and_critics(seed=5)
    scale, bias = action_scale_bias(LOW, HIGH)
    objective = PPOObjective(PPOConfig(clip_ratio=0.2))
    shared_lambda = SharedLambda(torch.tensor([1.0]))
    lambda_per_agent = uniform_lambda_per_agent(shared_lambda, N_AGENTS)
    batches = [_synthetic_batch(actors, scale, bias, seed=50 + t) for t in range(3)]
    rho_logits_value = torch.zeros(1, N_AGENTS)

    def _run_once():
        state0 = clone_functional_learner_state(actors=actors, reward_critic=rc, cost_critic=cc, optimizer=opt)
        rho_logits = rho_logits_value.detach().clone().requires_grad_(True)
        final_state = run_q_step_functional_chain(
            state0, actors=actors, reward_critic=rc, cost_critic=cc,
            batch_provider=lambda t: batches[t], q_meta=3, rho_logits=rho_logits,
            lambda_per_agent=lambda_per_agent, objective=objective, n_agents=N_AGENTS,
            action_scale=scale, action_bias=bias,
        )
        return final_state

    final_a = _run_once()
    final_b = _run_once()

    for i in range(N_AGENTS):
        a_params = {k: v.detach() for k, v in final_a.actor_params[i].items()}
        b_params = {k: v.detach() for k, v in final_b.actor_params[i].items()}
        assert _all_equal(a_params, b_params)
    a_reward = {k: v.detach() for k, v in final_a.reward_critic_params.items()}
    b_reward = {k: v.detach() for k, v in final_b.reward_critic_params.items()}
    assert _all_equal(a_reward, b_reward)
    assert final_a.step == final_b.step == 3
