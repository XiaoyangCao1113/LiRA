"""CityLearn training loops for the Uniform, PAL and LiRA arms.

All arms share the same environment, actor/critic architecture, live PPO
cadence and shared-cost budget; they differ only in how the shared-cost
penalty is distributed:

- ``uniform``: uniform responsibility and one shared dual variable;
- ``pal``: uniform responsibility and three independent per-building duals;
- ``lira``: one shared dual variable and a responsibility allocation ``rho``
  learned online by the q-step DU+LOO-SC lookahead estimator.

Each outer cycle of the LiRA arm (i) clones the live learner into ``M``
independent lookahead replicates (each a ``q_meta``-step functional PPO/Adam
unroll on fresh batches plus one terminal rollout), (ii) takes one Adam ascent
step on the responsibility logits with the mean DU+LOO-SC gradient, then
(iii) advances the live learner by ``q`` (the paper's ``h``) mutable PPO
updates under the new allocation and finally (iv) takes one projected dual
step from the mean live grid excess. Uniform/PAL cycles perform only (iii)
and (iv). After training, every arm is evaluated on one full held-out
episode (see :mod:`lira.envs.citylearn.evaluation`).
"""
from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import torch

from lira.estimators import apply_outer_ascent_step, lira_gradient
from lira.ppo import PPOConfig, PPOObjective
from lira.responsibility import SharedLambda

from .env import CityLearnDistributedEnv, CityLearnEnvConfig
from .evaluation import eval_seed_for, evaluate_heldout_arm
from .ppo_update import (
    Actor,
    Critic,
    CityLearnPPOBatch,
    action_scale_bias,
    mutable_citylearn_ppo_update,
    pal_lambda_per_agent,
    sample_actions,
    uniform_lambda_per_agent,
)
from .qstep_lira import TerminalRollout, collect_qstep_du_sc_replicate, sample_actions_functional
from .qstep_unroll import clone_functional_learner_state

ESTIMATOR_QSTEP_DU_SC = "qstep_du_sc"
ARMS = ("uniform", "pal", "lira")

N_AGENTS = 3
HIDDEN = 64
CLIP_RATIO = 0.2
ACTOR_LR = 3e-4
LAMBDA_INIT = 1.0


def _param_norm(actors) -> float:
    return float(torch.cat([p.detach().flatten() for actor in actors for p in actor.parameters()]).norm())


def _snapshot_actor_parameters(actors):
    """Take an exact, device-local snapshot for the PPO update gate.

    A norm is not a valid change detector: two different parameter vectors can
    have the same norm, while an optimizer step can legitimately be exactly
    zero (e.g. PPO clipping or a zero advantage batch).  The old screen used
    ``norm_before == norm_after`` and consequently treated a single valid
    zero-step as a failed run.
    """
    return [p.detach().clone() for actor in actors for p in actor.parameters()]


def _actor_parameters_changed(actors, before) -> bool:
    after = [p.detach() for actor in actors for p in actor.parameters()]
    if len(after) != len(before):
        raise ValueError("actor parameter snapshot does not match actor set")
    return any(not torch.equal(old, new) for old, new in zip(before, after))


def _capture_cpu_rng_state() -> tuple[object, tuple, torch.Tensor]:
    """Capture streams used by estimator-only response probes."""
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _restore_cpu_rng_state(state: tuple[object, tuple, torch.Tensor]) -> None:
    """Restore learner streams before the committed PPO transaction."""
    py_state, np_state, torch_state = state
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)


def _citylearn_time_step(env: CityLearnDistributedEnv) -> int:
    """Return the native current calendar index (adapter counter as fallback)."""
    native_env = getattr(env, "env", None)
    value = getattr(native_env, "time_step", None)
    return int(env.steps if value is None else value)


def _prepare_qstep_probe_start(
    *, dataset_name: str, seed: int, actors, scale: torch.Tensor, bias: torch.Tensor,
) -> tuple[CityLearnDistributedEnv, np.ndarray, np.ndarray, dict[str, int]]:
    """Create one isolated lookahead start: a fresh, independently seeded env
    advanced past the dynamics warm-up window under the current live actors."""
    probe_env = CityLearnDistributedEnv(
        CityLearnEnvConfig(dataset_name=dataset_name, seed=seed),
    )
    probe_obs, probe_state = probe_env.reset()
    probe_obs, probe_state = _burn_in(
        probe_env, actors, scale, bias, probe_env.dynamics_warmup_steps,
        probe_obs, probe_state,
    )
    return probe_env, probe_obs, probe_state, {
        "probe_start_adapter_step": int(probe_env.steps),
        "probe_start_time_step": _citylearn_time_step(probe_env),
    }


def _burn_in(env: CityLearnDistributedEnv, actors, scale: torch.Tensor, bias: torch.Tensor, steps: int, obs, state):
    """Advance ``env`` ``steps`` times without recording anything.

    ``env``'s dynamics-model-driven observations (e.g. indoor temperature,
    hence comfort welfare) are provably action-independent for the first
    ``env.dynamics_warmup_steps`` steps after a reset (see
    ``CityLearnDistributedEnv.dynamics_warmup_steps``'s docstring); a rollout
    batch collected entirely inside that window cannot show any welfare
    response to the actor's actions no matter how it is trained or scored.
    Call this once right after ``env.reset()`` with
    ``steps=env.dynamics_warmup_steps`` before starting a batch that will
    actually be scored.
    """
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_t, _log_prob_t = sample_actions(actors, obs_t, scale, bias)
        next_obs, next_state, _welfare, done, _accounting = env.step(action_t.squeeze(0).numpy())
        obs, state = next_obs, next_state
        if done:
            obs, state = env.reset()
    return obs, state


def _collect_rollout_batch(
    env: CityLearnDistributedEnv, actors, scale: torch.Tensor, bias: torch.Tensor,
    rollout_steps: int, obs, state,
) -> tuple[CityLearnPPOBatch, np.ndarray, np.ndarray, Any, Any]:
    """Collect ``rollout_steps`` consecutive env steps into one frozen PPO batch."""
    observations, states, actions_list, log_probs_list, welfares, costs = [], [], [], [], [], []
    for _ in range(rollout_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t = sample_actions(actors, obs_t, scale, bias)
        action_np = action_t.squeeze(0).numpy()
        observations.append(obs); states.append(state)
        actions_list.append(action_np); log_probs_list.append(log_prob_t.squeeze(0).numpy())
        next_obs, next_state, welfare, done, accounting = env.step(action_np)
        welfares.append(welfare); costs.append(accounting["C_cap"])
        obs, state = next_obs, next_state
        if done:
            obs, state = env.reset()

    observations = np.stack(observations); states = np.stack(states)
    actions_arr = np.stack(actions_list); log_probs_arr = np.stack(log_probs_list)
    welfares_arr = np.stack(welfares); costs_arr = np.asarray(costs, dtype=np.float64)

    reward_adv = torch.as_tensor(welfares_arr, dtype=torch.float32)
    reward_adv = (reward_adv - reward_adv.mean()) / (reward_adv.std() + 1e-6)
    cost_adv = torch.as_tensor(costs_arr, dtype=torch.float32)
    cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-6)
    value_target = torch.as_tensor(welfares_arr, dtype=torch.float32).sum(dim=1)  # Welfare = sum over agents
    cost_target = torch.as_tensor(costs_arr, dtype=torch.float32)

    batch = CityLearnPPOBatch(
        observations=torch.as_tensor(observations, dtype=torch.float32),
        actions=torch.as_tensor(actions_arr, dtype=torch.float32),
        state=torch.as_tensor(states, dtype=torch.float32),
        old_log_probs=torch.as_tensor(log_probs_arr, dtype=torch.float32),
        reward_advantages=reward_adv, cost_advantages=cost_adv,
        value_target=value_target, cost_target=cost_target,
    )
    return batch, welfares_arr, costs_arr, obs, state


def _collect_rollout_batch_functional(
    env: CityLearnDistributedEnv, actors, state, scale: torch.Tensor, bias: torch.Tensor,
    rollout_steps: int, obs, env_state,
) -> tuple[CityLearnPPOBatch, np.ndarray, np.ndarray, Any, Any]:
    """``_collect_rollout_batch``'s qstep_du_sc analog: samples on-policy from
    ``state``'s functional actor params (``sample_actions_functional``)
    instead of the live ``nn.Module`` actors, so a lookahead step's fresh
    batch is drawn from that step's own (evolving, rho-dependent) policy
    rather than a fixed original-actor batch.
    """
    observations, states, actions_list, log_probs_list, welfares, costs = [], [], [], [], [], []
    for _ in range(rollout_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t = sample_actions_functional(actors, state.actor_params, obs_t, scale, bias)
        action_np = action_t.squeeze(0).numpy()
        observations.append(obs); states.append(env_state)
        actions_list.append(action_np); log_probs_list.append(log_prob_t.squeeze(0).numpy())
        next_obs, next_env_state, welfare, done, accounting = env.step(action_np)
        welfares.append(welfare); costs.append(accounting["C_cap"])
        obs, env_state = next_obs, next_env_state
        if done:
            obs, env_state = env.reset()

    observations = np.stack(observations); states = np.stack(states)
    actions_arr = np.stack(actions_list); log_probs_arr = np.stack(log_probs_list)
    welfares_arr = np.stack(welfares); costs_arr = np.asarray(costs, dtype=np.float64)

    reward_adv = torch.as_tensor(welfares_arr, dtype=torch.float32)
    reward_adv = (reward_adv - reward_adv.mean()) / (reward_adv.std() + 1e-6)
    cost_adv = torch.as_tensor(costs_arr, dtype=torch.float32)
    cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-6)
    value_target = torch.as_tensor(welfares_arr, dtype=torch.float32).sum(dim=1)
    cost_target = torch.as_tensor(costs_arr, dtype=torch.float32)

    batch = CityLearnPPOBatch(
        observations=torch.as_tensor(observations, dtype=torch.float32),
        actions=torch.as_tensor(actions_arr, dtype=torch.float32),
        state=torch.as_tensor(states, dtype=torch.float32),
        old_log_probs=torch.as_tensor(log_probs_arr, dtype=torch.float32),
        reward_advantages=reward_adv, cost_advantages=cost_adv,
        value_target=value_target, cost_target=cost_target,
    )
    return batch, welfares_arr, costs_arr, obs, env_state


def _collect_terminal_rollout_functional(
    env: CityLearnDistributedEnv, actors, final_state, scale: torch.Tensor, bias: torch.Tensor,
    terminal_rollout_steps: int, obs, env_state, *,
    summary_out: dict[str, float] | None = None,
) -> tuple[TerminalRollout, Any, Any]:
    """One independent post-``q_meta`` rollout (``E=1`` support) under the final
    trained functional policy, disjoint from every training-step batch:
    called exactly once per replicate, after every ``functional_transaction_step``,
    continuing the replicate's own env forward from wherever its last
    training batch left off (never replaying or reusing training-step data).
    Its realized welfare (mean over steps of the sum-over-buildings reward) is
    the replicate's welfare; the optional summary also records the mean
    native cap cost for diagnostics.
    """
    observations, actions_list, welfares, costs = [], [], [], []
    for _ in range(terminal_rollout_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_t, _log_prob_t = sample_actions_functional(actors, final_state.actor_params, obs_t, scale, bias)
        action_np = action_t.squeeze(0).numpy()
        observations.append(obs); actions_list.append(action_np)
        next_obs, next_env_state, welfare, done, accounting = env.step(action_np)
        welfares.append(welfare); costs.append(float(accounting["C_cap"]))
        obs, env_state = next_obs, next_env_state
        if done:
            obs, env_state = env.reset()

    welfares_arr = np.stack(welfares)
    raw_welfare = float(welfares_arr.sum(axis=1).mean())
    raw_cost = float(np.mean(costs))
    if summary_out is not None:
        summary_out["terminal_welfare_raw"] = raw_welfare
        summary_out["terminal_cost_raw"] = raw_cost
    terminal = TerminalRollout(
        observations=torch.as_tensor(np.stack(observations), dtype=torch.float32),
        actions=torch.as_tensor(np.stack(actions_list), dtype=torch.float32),
        welfare=torch.as_tensor(raw_welfare, dtype=torch.float32),
    )
    return terminal, obs, env_state


def run_arm(
    *, arm: str, dataset_name: str, seed: int, q: int, cycles: int, rollout_steps: int,
    epochs: int, outer_lr: float, eval_seed: int | None = None,
    lambda_init: float = LAMBDA_INIT, lambda_lr: float | None = None, cost_budget: float = 0.0,
) -> dict[str, Any]:
    """Uniform or PAL arm: uniform responsibility with shared or per-agent duals."""
    if arm not in ("uniform", "pal"):
        raise ValueError("arm must be uniform or pal")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = CityLearnDistributedEnv(CityLearnEnvConfig(dataset_name=dataset_name, seed=seed))
    obs, state = env.reset()
    obs_dim, action_dim, state_dim = env.obs_dim, env.action_dim, env.state_dim
    scale, bias = action_scale_bias(
        torch.as_tensor(env.action_low[0], dtype=torch.float32),
        torch.as_tensor(env.action_high[0], dtype=torch.float32),
    )

    actors = [Actor(obs_dim, action_dim, hidden_dim=HIDDEN) for _ in range(N_AGENTS)]
    reward_critic = Critic(state_dim + N_AGENTS * action_dim, hidden_dim=HIDDEN)
    cost_critic = Critic(state_dim + N_AGENTS * action_dim, hidden_dim=HIDDEN)
    optimizer = torch.optim.Adam(
        [p for actor in actors for p in actor.parameters()]
        + list(reward_critic.parameters()) + list(cost_critic.parameters()),
        lr=ACTOR_LR,
    )
    objective = PPOObjective(PPOConfig(clip_ratio=CLIP_RATIO))
    rho_probs = torch.full((N_AGENTS,), 1.0 / N_AGENTS)
    rho_logits = torch.zeros(1, N_AGENTS)  # static uniform rho in both baseline arms
    obs, state = _burn_in(env, actors, scale, bias, env.dynamics_warmup_steps, obs, state)
    resolved_lambda_lr = outer_lr if lambda_lr is None else lambda_lr

    if arm == "uniform":
        dual = SharedLambda(torch.tensor([lambda_init]))
    else:  # pal
        dual = SharedLambda(torch.full((N_AGENTS,), lambda_init))

    cycle_rows = []
    actor_update_count = 0
    actor_changed_update_count = 0
    for cycle in range(cycles):
        cycle_losses = []
        cycle_welfare = []
        cycle_cost = []
        param_norm_before_cycle = _param_norm(actors)
        for _q_step in range(q):
            batch, welfares_arr, costs_arr, obs, state = _collect_rollout_batch(
                env, actors, scale, bias, rollout_steps, obs, state,
            )
            lambda_per_agent = (
                uniform_lambda_per_agent(dual, N_AGENTS) if arm != "pal"
                else pal_lambda_per_agent(dual, N_AGENTS)
            )
            parameter_snapshot = _snapshot_actor_parameters(actors)
            loss = mutable_citylearn_ppo_update(
                actors=actors, reward_critic=reward_critic, cost_critic=cost_critic,
                optimizer=optimizer, rho_logits=rho_logits, lambda_per_agent=lambda_per_agent,
                batch=batch, objective=objective, epochs=epochs, n_agents=N_AGENTS,
                action_dim=action_dim, action_scale=scale, action_bias=bias,
            )
            actor_changed = _actor_parameters_changed(actors, parameter_snapshot)
            actor_update_count += 1
            actor_changed_update_count += int(actor_changed)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arm}: non-finite PPO loss at cycle {cycle}")
            for actor in actors:
                for p in actor.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise FloatingPointError(f"{arm}: non-finite actor gradient at cycle {cycle}")
            cycle_losses.append(float(loss.detach()))
            cycle_welfare.append(float(welfares_arr.sum(axis=1).mean()))
            cycle_cost.append(float(costs_arr.mean()))

        # Every PAL multiplier sees the same district cost and budget.
        dual_gradient_value = float(np.mean(cycle_cost)) - cost_budget
        dual_gradient = torch.full((dual.num_constraints,), dual_gradient_value)
        dual.projected_update_(dual_gradient, step_size=resolved_lambda_lr)
        param_norm_after_cycle = _param_norm(actors)

        cycle_rows.append({
            "cycle": cycle + 1,
            "mean_ppo_loss": float(np.mean(cycle_losses)),
            "mean_welfare_sum": float(np.mean(cycle_welfare)),
            "mean_cap_cost": float(np.mean(cycle_cost)),
            "dual_values_after": dual.values.tolist(),
            "param_norm_before_cycle": param_norm_before_cycle,
            "param_norm_after_cycle": param_norm_after_cycle,
            "actor_updates": q,
            "actor_updates_changed": actor_changed_update_count,
        })
        print(json.dumps({"arm": arm, **cycle_rows[-1]}))

    resolved_eval_seed = eval_seed_for(seed) if eval_seed is None else int(eval_seed)
    heldout = evaluate_heldout_arm(
        actors=actors, dataset_name=dataset_name, eval_seed=resolved_eval_seed,
        action_scale=scale, action_bias=bias,
    )
    if actor_changed_update_count == 0:
        raise RuntimeError(f"{arm}: no actor parameters changed across {actor_update_count} PPO updates")
    print(json.dumps({"arm": arm, "heldout_eval": heldout.to_dict()}))

    return {
        "arm": arm, "seed": seed, "q": q, "cycles": cycles, "rollout_steps": rollout_steps,
        "epochs": epochs, "outer_lr": outer_lr, "obs_dim": obs_dim, "action_dim": action_dim,
        "state_dim": state_dim, "n_agents": N_AGENTS,
        "lambda_init": lambda_init, "lambda_lr": resolved_lambda_lr, "cost_budget": cost_budget,
        "final_dual_values": dual.values.tolist(), "cycles_detail": cycle_rows,
        "fixed_rho": rho_probs.tolist(),
        "heldout_eval": heldout.to_dict(),
    }


def run_lira_arm(
    *, dataset_name: str, seed: int, q: int, cycles: int, rollout_steps: int,
    epochs: int, outer_lr: float, m_replicates: int, eval_seed: int | None = None,
    rho_lr: float | None = None, rho_grad_clip: float | None = None,
    lambda_init: float = LAMBDA_INIT, lambda_lr: float | None = None, cost_budget: float = 0.0,
    q_meta: int = 2, terminal_rollout_steps: int | None = None,
) -> dict[str, Any]:
    """LiRA arm: rho_logits is LEARNED via the q-step DU+LOO-SC outer gradient.

    Each cycle: collect ``m_replicates`` independent lookahead replicates
    (each from a fresh, independently seeded env advanced past the dynamics
    warm-up under the current live actors) to build the DU+LOO-SC gradient
    and ascend ``rho_logits`` by one outer Adam step (``rho_lr``); then
    advance the mutable actor/critic via ``q`` PPO batches on the main env
    using the updated ``rho_logits`` and one shared dual (updated once per
    cycle from the mean live cap cost).

    Each replicate is a ``q_meta``-step functional actor/critic/Adam chain
    with a fresh on-policy batch per lookahead step plus one independent
    terminal rollout supplying the replicate's welfare and terminal
    joint-policy score (see :mod:`lira.envs.citylearn.qstep_lira`). ``q_meta``
    is independent of ``q`` (the live update cadence ``h``).
    """
    if q_meta < 2:
        raise ValueError("qstep_du_sc requires q_meta >= 2")
    resolved_terminal_rollout_steps = rollout_steps if terminal_rollout_steps is None else terminal_rollout_steps
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = CityLearnDistributedEnv(CityLearnEnvConfig(dataset_name=dataset_name, seed=seed))
    obs, state = env.reset()
    obs_dim, action_dim, state_dim = env.obs_dim, env.action_dim, env.state_dim
    scale, bias = action_scale_bias(
        torch.as_tensor(env.action_low[0], dtype=torch.float32),
        torch.as_tensor(env.action_high[0], dtype=torch.float32),
    )

    actors = [Actor(obs_dim, action_dim, hidden_dim=HIDDEN) for _ in range(N_AGENTS)]
    reward_critic = Critic(state_dim + N_AGENTS * action_dim, hidden_dim=HIDDEN)
    cost_critic = Critic(state_dim + N_AGENTS * action_dim, hidden_dim=HIDDEN)
    optimizer = torch.optim.Adam(
        [p for actor in actors for p in actor.parameters()]
        + list(reward_critic.parameters()) + list(cost_critic.parameters()),
        lr=ACTOR_LR,
    )
    objective = PPOObjective(PPOConfig(clip_ratio=CLIP_RATIO))
    rho_logits = torch.zeros(1, N_AGENTS, requires_grad=True)
    resolved_rho_lr = outer_lr if rho_lr is None else rho_lr
    outer_optimizer = torch.optim.Adam([rho_logits], lr=resolved_rho_lr)
    resolved_lambda_lr = outer_lr if lambda_lr is None else lambda_lr
    dual = SharedLambda(torch.tensor([lambda_init]))
    obs, state = _burn_in(env, actors, scale, bias, env.dynamics_warmup_steps, obs, state)

    cycle_rows = []
    actor_update_count = 0
    actor_changed_update_count = 0
    for cycle in range(cycles):
        rho_before_cycle = rho_logits.detach().clone().tolist()
        param_norm_before_cycle = _param_norm(actors)

        learner_rng_state = _capture_cpu_rng_state()
        replicates = []
        replicate_diagnostics = []
        for replicate_index in range(m_replicates):
            rep_env, rep_obs, rep_state, probe_start = _prepare_qstep_probe_start(
                dataset_name=dataset_name,
                seed=seed + 9973 * (cycle + 1) + replicate_index,
                actors=actors,
                scale=scale,
                bias=bias,
            )
            lambda_per_agent = uniform_lambda_per_agent(dual, N_AGENTS)
            cursor = {"obs": rep_obs, "env_state": rep_state}
            mean_costs_per_step: list[float] = []

            def _qstep_batch_provider(t, functional_state):
                batch, welfares_arr, costs_arr, next_obs, next_env_state = _collect_rollout_batch_functional(
                    rep_env, actors, functional_state, scale, bias, rollout_steps,
                    cursor["obs"], cursor["env_state"],
                )
                cursor["obs"], cursor["env_state"] = next_obs, next_env_state
                mean_costs_per_step.append(float(costs_arr.mean()))
                return batch

            terminal_holder: dict[str, Any] = {}

            def _qstep_terminal_provider(final_state):
                terminal, next_obs, next_env_state = _collect_terminal_rollout_functional(
                    rep_env, actors, final_state, scale, bias, resolved_terminal_rollout_steps,
                    cursor["obs"], cursor["env_state"],
                    summary_out=terminal_holder,
                )
                cursor["obs"], cursor["env_state"] = next_obs, next_env_state
                terminal_holder["terminal"] = terminal
                return terminal

            state0 = clone_functional_learner_state(
                actors=actors, reward_critic=reward_critic, cost_critic=cost_critic, optimizer=optimizer,
            )
            live_actor_snapshot = _snapshot_actor_parameters(actors)
            replicate = collect_qstep_du_sc_replicate(
                actors=actors, reward_critic=reward_critic, cost_critic=cost_critic, state0=state0,
                batch_provider=_qstep_batch_provider, terminal_rollout_provider=_qstep_terminal_provider,
                rho_logits_value=rho_logits.detach(), lambda_per_agent=lambda_per_agent, objective=objective,
                n_agents=N_AGENTS, action_scale=scale, action_bias=bias, q_meta=q_meta,
            )
            if _actor_parameters_changed(actors, live_actor_snapshot):
                raise RuntimeError(
                    "lira (qstep_du_sc): replicate collection mutated the live actor parameters"
                )
            replicates.append(replicate)
            replicate_diagnostics.append({
                "welfare": terminal_holder["terminal_welfare_raw"],
                "meta_objective": float(replicate.welfare),
                "terminal_cost": terminal_holder["terminal_cost_raw"],
                "terminal_native_shared_cost": terminal_holder["terminal_cost_raw"],
                "mean_cap_cost": float(np.mean(mean_costs_per_step)) if mean_costs_per_step else None,
                "q_meta": q_meta,
                **probe_start,
                "probe_end_adapter_step": int(rep_env.steps),
                "probe_end_time_step": _citylearn_time_step(rep_env),
                "training_score": replicate.diagnostics.get("training_score"),
                "terminal_score": replicate.diagnostics.get("terminal_score"),
            })

        # Independent response probes must not perturb the committed learner
        # transaction through process-global Python/NumPy/Torch RNG streams.
        # In particular, outer_lr=0 then becomes a meaningful same-S0 parity
        # protocol rather than an estimator-dependent training trajectory.
        _restore_cpu_rng_state(learner_rng_state)

        mean_grad, lira_diagnostics = lira_gradient(replicates)
        apply_outer_ascent_step(rho_logits, outer_optimizer, mean_grad, grad_clip_norm=rho_grad_clip)
        rho_after_cycle = rho_logits.detach().clone().tolist()

        cycle_losses, cycle_welfare, cycle_cost = [], [], []
        for _q_step in range(q):
            batch, welfares_arr, costs_arr, obs, state = _collect_rollout_batch(
                env, actors, scale, bias, rollout_steps, obs, state,
            )
            lambda_per_agent = uniform_lambda_per_agent(dual, N_AGENTS)
            parameter_snapshot = _snapshot_actor_parameters(actors)
            loss = mutable_citylearn_ppo_update(
                actors=actors, reward_critic=reward_critic, cost_critic=cost_critic,
                optimizer=optimizer, rho_logits=rho_logits.detach(), lambda_per_agent=lambda_per_agent,
                batch=batch, objective=objective, epochs=epochs, n_agents=N_AGENTS,
                action_dim=action_dim, action_scale=scale, action_bias=bias,
            )
            actor_changed = _actor_parameters_changed(actors, parameter_snapshot)
            actor_update_count += 1
            actor_changed_update_count += int(actor_changed)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"lira: non-finite PPO loss at cycle {cycle}")
            cycle_losses.append(float(loss.detach()))
            cycle_welfare.append(float(welfares_arr.sum(axis=1).mean()))
            cycle_cost.append(float(costs_arr.mean()))

        dual_gradient = torch.full((dual.num_constraints,), float(np.mean(cycle_cost)) - cost_budget)
        dual.projected_update_(dual_gradient, step_size=resolved_lambda_lr)
        param_norm_after_cycle = _param_norm(actors)

        rho_movement = [after - before for before, after in zip(rho_before_cycle[0], rho_after_cycle[0])]
        cycle_rows.append({
            "cycle": cycle + 1,
            "estimator": ESTIMATOR_QSTEP_DU_SC,
            "q_meta": q_meta,
            "m_replicates": m_replicates,
            "e_terminal_rollouts": 1,
            "live_actuation_cadence_h": q,
            "mean_ppo_loss": float(np.mean(cycle_losses)),
            "mean_welfare_sum": float(np.mean(cycle_welfare)),
            "mean_cap_cost": float(np.mean(cycle_cost)),
            "dual_values_after": dual.values.tolist(),
            "rho_logits_before_cycle": rho_before_cycle,
            "rho_logits_after_cycle": rho_after_cycle,
            "rho_movement": rho_movement,
            "lira_gradient_sign": mean_grad.sign().tolist(),
            "lira_gradient_norm": lira_diagnostics["gradient_norm"],
            "lira_du_mean": lira_diagnostics["du_mean"],
            "lira_loo_sc_mean": lira_diagnostics["loo_sc_mean"],
            "replicate_diagnostics": replicate_diagnostics,
            "live_state_nonmutation_checked": True,
            "learner_rng_restored": True,
            "param_norm_before_cycle": param_norm_before_cycle,
            "param_norm_after_cycle": param_norm_after_cycle,
            "actor_updates": q,
            "actor_updates_changed": actor_changed_update_count,
        })
        print(json.dumps({"arm": "lira", **cycle_rows[-1]}))

    resolved_eval_seed = eval_seed_for(seed) if eval_seed is None else int(eval_seed)
    heldout = evaluate_heldout_arm(
        actors=actors, dataset_name=dataset_name, eval_seed=resolved_eval_seed,
        action_scale=scale, action_bias=bias,
    )
    if actor_changed_update_count == 0:
        raise RuntimeError(f"lira: no actor parameters changed across {actor_update_count} PPO updates")
    print(json.dumps({"arm": "lira", "heldout_eval": heldout.to_dict()}))

    return {
        "arm": "lira", "seed": seed, "q": q, "cycles": cycles, "rollout_steps": rollout_steps,
        "epochs": epochs, "outer_lr": outer_lr, "m_replicates": m_replicates,
        "rho_lr": resolved_rho_lr, "rho_grad_clip": rho_grad_clip,
        "lambda_init": lambda_init, "lambda_lr": resolved_lambda_lr, "cost_budget": cost_budget,
        "obs_dim": obs_dim, "action_dim": action_dim, "state_dim": state_dim, "n_agents": N_AGENTS,
        "final_dual_values": dual.values.tolist(), "final_rho_logits": rho_logits.detach().tolist(),
        "estimator": ESTIMATOR_QSTEP_DU_SC, "q_meta": q_meta,
        "terminal_rollout_steps": resolved_terminal_rollout_steps,
        "cycles_detail": cycle_rows,
        "heldout_eval": heldout.to_dict(),
    }
