"""CityLearn q-step DU+LOO-SC lookahead replicate (``qstep_du_sc``).

One replicate runs a ``q_meta >= 2`` step functional lookahead from a clone
of the live learner (:mod:`lira.envs.citylearn.qstep_unroll`) and returns the
four quantities consumed by :func:`lira.estimators.lira_gradient`:

1. **Multi-step, on-policy training data.** Each of the ``q_meta`` lookahead
   steps gets its own *fresh* batch, sampled under that step's own (evolving,
   rho-dependent) functional actor state. ``batch_provider(t, state)``
   receives the pre-update ``FunctionalLearnerState`` for step ``t`` so the
   caller can sample on-policy (see ``sample_actions_functional``).
2. **Real critic/Adam chain.** Steps run through
   ``qstep_unroll.functional_transaction_step`` -- full actor+critic joint
   loss, actor-only clip, one hand-written Adam step per step, carried
   undetached -- so the final state is a genuine function of ``rho_logits``.
3. **``du`` = terminal-welfare-weighted terminal score.** After the last
   lookahead step, one independent terminal rollout's summed joint-policy
   log-likelihood ``u_m`` is evaluated under the final trained state; its
   time-summed score is normalized by the training-batch length ``T_train``
   (not the terminal length), so
   ``du = stopgrad(W_m) * (sum_t u_{m,t} / T_train)``. Welfare itself is a
   non-differentiable real-environment quantity, so its rho-sensitivity is
   estimated via the score of the policy that produced it.
4. **``score`` = training-data sampling correction only.** ``score`` is the
   sum, over every lookahead step ``t``, of the time-summed log-probability of
   that step's own realized actions evaluated under *that step's own
   pre-update state*, each divided by the common denominator ``T_train``.
   Each ``state_t`` is a differentiable function of ``rho_logits`` (via the
   ``t`` prior functional steps), so ``d(score)/d(rho_logits)`` captures how
   rho shifts the training-data distribution. ``lira_gradient`` combines it
   with ``du`` as ``du + stopgrad(welfare - baseline) * score``; the terminal
   score is never added into ``score``.
5. **Welfare = the independent terminal rollout's realized welfare**
   (mean over its steps of the sum-over-buildings comfort reward), sampled
   disjoint from every training-step batch.

``q_meta`` (the lookahead length) and the live update cadence ``h`` (``--q``
in ``scripts/train_citylearn.py``: live PPO batches per outer cycle) are
independent knobs; nothing here reads or writes ``h``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
from torch import nn
from torch.func import functional_call

from lira.ppo import PPOObjective

from .distributions import TanhNormal
from .ppo_update import ACTOR_CLIP_NORM, CityLearnPPOBatch, _functional_actor_log_probs
from .qstep_unroll import FunctionalLearnerState, functional_transaction_step


@dataclass(frozen=True)
class TerminalRollout:
    """One independent, post-``q_meta`` fresh rollout used only to read welfare/score.

    Disjoint by construction from every training-step batch: the caller
    builds this from steps taken *after* the last ``functional_transaction_step``
    of a replicate's chain, under the final trained functional state's own
    action distribution -- never from data already fed to
    ``functional_transaction_step``.
    """

    observations: torch.Tensor  # T x N x obs_dim
    actions: torch.Tensor       # T x N x action_dim
    welfare: torch.Tensor       # scalar, detached: mean over T of sum-over-agents realized welfare


def sample_actions_functional(
    actors: Sequence[nn.Module],
    param_dicts: Sequence[dict[str, torch.Tensor]],
    obs: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample B x N x A raw actions (and B x N summed log-probs) from a functional state.

    Functional analog of ``ppo_update.sample_actions`` (which
    samples from the live ``nn.Module`` actors' own parameters): here the
    forward pass goes through ``torch.func.functional_call`` against an
    explicit (possibly rho-dependent) parameter dict per actor, so a caller
    can sample fresh on-policy data under an evolving
    ``FunctionalLearnerState`` mid-unroll. Always called under
    ``torch.no_grad()`` by convention (sampled actions are data, not part of
    the differentiated graph -- only their log-probability under a
    parametrized policy is differentiated, the standard score-function
    separation used throughout this module).
    """
    action_rows, log_prob_rows = [], []
    for i, (actor, params) in enumerate(zip(actors, param_dicts)):
        mean, log_std = functional_call(actor, params, (obs[:, i],))
        per_dim_actions, per_dim_log_probs = [], []
        for d in range(actor.action_dim):
            dist = TanhNormal(mean[:, d], log_std[:, d], scale=float(scale[d]))
            centered = dist.sample(deterministic=False)
            per_dim_actions.append(centered + bias[d])
            per_dim_log_probs.append(dist.log_prob(centered))
        action_rows.append(torch.stack(per_dim_actions, dim=1))
        log_prob_rows.append(torch.stack(per_dim_log_probs, dim=1).sum(dim=1))
    return torch.stack(action_rows, dim=1), torch.stack(log_prob_rows, dim=1)


@dataclass
class QStepLiraReplicate:
    """One independent replicate's multi-step DU+SC decomposition plus its own rho_logits leaf.

    Attribute names (``du``/``score``/``welfare``/``rho_logits``) match the
    interface of the domain-agnostic :mod:`lira.estimators` functions
    (``leave_one_out_baselines``/``lira_gradient``/``apply_outer_ascent_step``).
    """

    du: torch.Tensor        # stopgrad(terminal welfare) * terminal joint-policy score, function of rho_logits
    score: torch.Tensor     # differentiable training-data-sampling score only (no terminal term)
    welfare: torch.Tensor   # detached scalar realized welfare from the independent terminal rollout
    rho_logits: torch.Tensor  # this replicate's own independent leaf (requires_grad)
    diagnostics: dict[str, float] = field(default_factory=dict)


def collect_qstep_du_sc_replicate(
    *,
    actors: Sequence[nn.Module],
    reward_critic: nn.Module,
    cost_critic: nn.Module,
    state0: FunctionalLearnerState,
    batch_provider: Callable[[int, FunctionalLearnerState], CityLearnPPOBatch],
    terminal_rollout_provider: Callable[[FunctionalLearnerState], TerminalRollout],
    rho_logits_value: torch.Tensor,
    lambda_per_agent: torch.Tensor,
    objective: PPOObjective,
    n_agents: int,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    q_meta: int,
    actor_clip_norm: float | None = ACTOR_CLIP_NORM,
) -> QStepLiraReplicate:
    """Build one qstep_du_sc replicate's (du, score, welfare) from a q_meta-step unroll.

    ``state0`` must already be an isolated clone (see
    ``qstep_unroll.clone_functional_learner_state``) -- this
    function only chains ``functional_transaction_step`` forward from it and
    never touches ``actors``/``reward_critic``/``cost_critic`` or any live
    optimizer, so passing the live modules' own parameters directly (instead
    of a clone) would be a caller bug, not something this function guards
    against (mirroring ``functional_transaction_step``'s own contract).

    ``batch_provider(t, state)`` is called once per step with the *pre-update*
    state for that step, so it can sample fresh on-policy data (e.g. via
    ``sample_actions_functional(actors, state.actor_params, ...)``) from the
    evolving policy rather than a fixed original-actor batch.
    ``terminal_rollout_provider(final_state)`` is called exactly once, after
    the last step, with the final (post-``q_meta``) state.

    ``du``/``score`` implement the direct and sampling-correction terms of
    the paper's finite-training gradient:
    ``du = stopgrad(W_m) * (sum_t u_{m,t} / T_train)`` (terminal-return-weighted
    terminal joint-policy score) and ``score = sum_k(sum_t s_{k,t} / T_train)``
    (training-trajectory score only, evaluated on each step's own pre-update
    state). ``T_train`` is the batch-time length, required to be equal across
    lookahead steps; the terminal rollout may have a different length. This
    shared denominator preserves previous scaling when lengths match and
    avoids terminal-vs-training scale drift when they differ. The terminal
    score is never added into ``score``, since ``lira.estimators.lira_gradient``
    already folds ``du`` into the per-replicate objective separately from its
    LOO-baseline-weighted use of ``score``. ``functional_transaction_step``
    still computes and differentiates through ``actor_loss``/``critic_loss``
    internally (that is the learner-update mechanics, not the DU/SC
    estimator); this function itself never reads an actor-loss value.
    """
    if q_meta < 2:
        raise ValueError("collect_qstep_du_sc_replicate requires q_meta >= 2")

    rho_logits_i = rho_logits_value.detach().clone().requires_grad_(True)
    state = state0
    training_score_terms: list[torch.Tensor] = []
    training_time_steps: int | None = None

    for t in range(q_meta):
        batch = batch_provider(t, state)
        batch_time_steps = int(batch.observations.shape[0])
        if batch_time_steps <= 0:
            raise ValueError("collect_qstep_du_sc_replicate requires non-empty training batches")
        if training_time_steps is None:
            training_time_steps = batch_time_steps
        elif batch_time_steps != training_time_steps:
            raise ValueError(
                "collect_qstep_du_sc_replicate requires a constant training-batch time length "
                f"across lookahead steps; got {training_time_steps} then {batch_time_steps}"
            )
        new_log_probs_t = _functional_actor_log_probs(
            actors, state.actor_params, batch.observations, batch.actions, action_scale, action_bias,
        )
        training_score_terms.append(new_log_probs_t.sum() / training_time_steps)

        state = functional_transaction_step(
            state, actors=actors, reward_critic=reward_critic, cost_critic=cost_critic,
            batch=batch, rho_logits=rho_logits_i, lambda_per_agent=lambda_per_agent,
            objective=objective, n_agents=n_agents, action_scale=action_scale, action_bias=action_bias,
            actor_clip_norm=actor_clip_norm,
        )

    terminal = terminal_rollout_provider(state)
    terminal_log_probs = _functional_actor_log_probs(
        actors, state.actor_params, terminal.observations, terminal.actions, action_scale, action_bias,
    )
    assert training_time_steps is not None
    terminal_time_steps = int(terminal.observations.shape[0])
    if terminal_time_steps <= 0:
        raise ValueError("collect_qstep_du_sc_replicate requires a non-empty terminal rollout")
    terminal_score = terminal_log_probs.sum() / training_time_steps

    welfare = terminal.welfare.detach()
    du = welfare * terminal_score
    score = torch.stack(training_score_terms).sum()

    if not torch.isfinite(du).all():
        raise RuntimeError("collect_qstep_du_sc_replicate: non-finite DU term")
    if not torch.isfinite(score).all():
        raise RuntimeError("collect_qstep_du_sc_replicate: non-finite score term")

    return QStepLiraReplicate(
        du=du, score=score, welfare=welfare, rho_logits=rho_logits_i,
        diagnostics={
            "q_meta": float(q_meta),
            "training_score": float(score.detach()),
            "terminal_score": float(terminal_score.detach()),
            "training_batch_steps": float(training_time_steps),
            "terminal_rollout_steps": float(terminal_time_steps),
            "score_time_denominator": float(training_time_steps),
            "terminal_welfare": float(welfare),
        },
    )
