"""Differentiable q-step functional actor/critic/Adam learner transaction.

Clones a live actor/critic/Adam state and chains ``q_meta >= 2`` functional
updates, one fresh batch per step, carrying Adam's first/second moments (and
the parameter values themselves) forward *undetached*, so the final learner
state is a genuine function of ``rho_logits``. This is the lookahead kernel
used by :mod:`lira.envs.citylearn.qstep_lira`.

Adam replication note: the functional update below reimplements
``torch.optim.Adam``'s default (non-fused, non-amsgrad, no weight decay)
step formula by hand, operating on cloned leaf tensors with
``create_graph=True`` autograd so the whole ``q_meta``-step chain stays
differentiable. ``SafeSqrt`` defines ``d(sqrt(x))/dx = 0`` at ``x = 0``
instead of the ordinary ``inf`` gradient, since a parameter can genuinely
have zero accumulated second moment (e.g. an unused output slot) after few
steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn
from torch.func import functional_call

from lira.ppo import PPOObjective

from .ppo_update import (
    ACTOR_CLIP_NORM,
    CityLearnPPOBatch,
    _functional_actor_log_probs,
    citylearn_actor_critic_epoch_losses,
)

_CLIP_NORM_EPS = 1e-6


class SafeSqrt(torch.autograd.Function):
    """``sqrt`` with a zero (not ``inf``) gradient at exactly zero input."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        y = torch.sqrt(torch.clamp(x, min=0.0))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        positive = y > 0
        safe_y = torch.where(positive, y, torch.ones_like(y))
        grad = grad_output * 0.5 / safe_y
        return torch.where(positive, grad, torch.zeros_like(grad))


@dataclass
class FunctionalLearnerState:
    """A cloned, self-contained actor/critic/Adam state, disjoint from the live modules.

    Every tensor here is either a fresh leaf clone (``actor_params`` /
    ``*_critic_params`` at construction time) or a fresh, non-in-place
    result of a prior functional Adam step -- never the same storage as the
    live ``nn.Module`` parameters or the live ``torch.optim.Adam``'s state,
    so mutating/differentiating through this state can never perturb the
    live learner (see ``clone_functional_learner_state``).
    """

    actor_params: list[dict[str, torch.Tensor]]
    reward_critic_params: dict[str, torch.Tensor]
    cost_critic_params: dict[str, torch.Tensor]
    exp_avg: dict[str, torch.Tensor]
    exp_avg_sq: dict[str, torch.Tensor]
    step: int
    lr: float
    betas: tuple[float, float]
    eps: float


def _flat_keys(n_actors: int, actor_param_names: Sequence[Sequence[str]]) -> list[str]:
    keys = []
    for i in range(n_actors):
        keys.extend(f"actor{i}.{name}" for name in actor_param_names[i])
    return keys


def clone_functional_learner_state(
    *,
    actors: Sequence[nn.Module],
    reward_critic: nn.Module,
    cost_critic: nn.Module,
    optimizer: torch.optim.Adam,
) -> FunctionalLearnerState:
    """Clone actor/critic parameter values and Adam moments off the live learner.

    Every returned tensor is ``.detach().clone()``-ed (values) or freshly
    zero-initialized (moments not yet present because ``optimizer`` has
    never stepped) so the result shares no storage with the live modules or
    optimizer state. Adam's ``step`` counter is read off the first
    parameter that has one (all parameters in a single ``optimizer.step()``
    call advance together) and defaults to 0.
    """
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("clone_functional_learner_state requires a torch.optim.Adam optimizer")
    defaults = optimizer.param_groups[0]
    lr, betas, eps = float(defaults["lr"]), tuple(defaults["betas"]), float(defaults["eps"])

    actor_params: list[dict[str, torch.Tensor]] = []
    exp_avg: dict[str, torch.Tensor] = {}
    exp_avg_sq: dict[str, torch.Tensor] = {}
    step = 0

    def _clone_module(prefix: str, module: nn.Module) -> dict[str, torch.Tensor]:
        # NOTE: this dict's keys are raw ``named_parameters()`` names (no
        # ``prefix``) because it is later fed straight to
        # ``torch.func.functional_call(module, this_dict, ...)``, which
        # requires the module's own (unprefixed) parameter names. The
        # ``prefix``-qualified key is used only for the flat, globally
        # unique ``exp_avg``/``exp_avg_sq`` namespace below.
        nonlocal step
        cloned: dict[str, torch.Tensor] = {}
        for name, p in module.named_parameters():
            flat_key = f"{prefix}.{name}"
            cloned[name] = p.detach().clone().requires_grad_(True)
            state = optimizer.state.get(p, {})
            exp_avg[flat_key] = state.get("exp_avg", torch.zeros_like(p)).detach().clone()
            exp_avg_sq[flat_key] = state.get("exp_avg_sq", torch.zeros_like(p)).detach().clone()
            if "step" in state:
                step = int(state["step"])
        return cloned

    for i, actor in enumerate(actors):
        actor_params.append(_clone_module(f"actor{i}", actor))
    reward_critic_params = _clone_module("reward_critic", reward_critic)
    cost_critic_params = _clone_module("cost_critic", cost_critic)

    return FunctionalLearnerState(
        actor_params=actor_params, reward_critic_params=reward_critic_params,
        cost_critic_params=cost_critic_params, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq,
        step=step, lr=lr, betas=betas, eps=eps,
    )


def _functional_clip_grad_norm(
    grads: Sequence[torch.Tensor], max_norm: float | None,
) -> list[torch.Tensor]:
    """Differentiable equivalent of ``torch.nn.utils.clip_grad_norm_`` (no ``.grad`` mutation).

    Reproduces the exact formula PyTorch uses (``clip_coef = max_norm /
    (total_norm + 1e-6)``, clamped to at most 1) so a one-step functional
    replica matches the live in-place call's *values* bit-for-bit, while
    staying a pure function of ``grads`` so the whole chain differentiates.
    """
    if max_norm is None:
        return list(grads)
    total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
    clip_coef = max_norm / (total_norm + _CLIP_NORM_EPS)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return [g * clip_coef_clamped for g in grads]


def _functional_adam_update(
    params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor],
    exp_avg: Sequence[torch.Tensor], exp_avg_sq: Sequence[torch.Tensor],
    step: int, lr: float, betas: tuple[float, float], eps: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """One differentiable Adam step, matching ``torch.optim.Adam``'s default math exactly.

    Every output is a fresh (non-in-place) tensor so ``create_graph=True``
    autograd through this function -- and through however many chained
    calls -- stays connected back to whatever ``grads`` depended on (here,
    ``rho_logits``).
    """
    beta1, beta2 = betas
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    new_params, new_exp_avg, new_exp_avg_sq = [], [], []
    for p, g, m, v in zip(params, grads, exp_avg, exp_avg_sq):
        m_new = beta1 * m + (1.0 - beta1) * g
        v_new = beta2 * v + (1.0 - beta2) * g * g
        step_size = lr / bias_correction1
        denom = SafeSqrt.apply(v_new) / (bias_correction2 ** 0.5) + eps
        p_new = p - step_size * m_new / denom
        new_params.append(p_new)
        new_exp_avg.append(m_new)
        new_exp_avg_sq.append(v_new)
    return new_params, new_exp_avg, new_exp_avg_sq


def functional_transaction_step(
    state: FunctionalLearnerState,
    *,
    actors: Sequence[nn.Module],
    reward_critic: nn.Module,
    cost_critic: nn.Module,
    batch: CityLearnPPOBatch,
    rho_logits: torch.Tensor,
    lambda_per_agent: torch.Tensor,
    objective: PPOObjective,
    n_agents: int,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    actor_clip_norm: float | None = ACTOR_CLIP_NORM,
) -> FunctionalLearnerState:
    """One differentiable actor+critic+Adam step from ``state``, on one fresh batch.

    Mirrors ``ppo_update.mutable_citylearn_ppo_update`` with
    ``epochs=1`` exactly (same joint actor+critic loss, same actor-only
    gradient-norm clip, same single Adam step over actor+critic
    parameters), except every operation is functional/out-of-place so the
    result is a genuine (non-detached) function of ``rho_logits`` and of
    ``state`` -- never mutating ``actors``/``reward_critic``/``cost_critic``
    or any live optimizer.
    """
    actor_param_names = [list(d.keys()) for d in state.actor_params]
    new_log_probs = _functional_actor_log_probs(
        actors, state.actor_params, batch.observations, batch.actions, action_scale, action_bias,
    )
    critic_in = torch.cat([batch.state, batch.actions.reshape(batch.actions.shape[0], -1)], dim=1)
    reward_pred = functional_call(reward_critic, state.reward_critic_params, (critic_in,))
    cost_pred = functional_call(cost_critic, state.cost_critic_params, (critic_in,))

    actor_loss, critic_loss, _total = citylearn_actor_critic_epoch_losses(
        new_log_probs=new_log_probs, reward_critic_prediction=reward_pred,
        cost_critic_prediction=cost_pred, rho_logits=rho_logits,
        lambda_per_agent=lambda_per_agent, batch=batch, objective=objective, n_agents=n_agents,
    )

    actor_keys = _flat_keys(len(actors), actor_param_names)
    critic_keys = (
        [f"reward_critic.{name}" for name in state.reward_critic_params]
        + [f"cost_critic.{name}" for name in state.cost_critic_params]
    )
    flat_keys = actor_keys + critic_keys
    flat_params = (
        [state.actor_params[i][name] for i, names in enumerate(actor_param_names) for name in names]
        + [state.reward_critic_params[name] for name in state.reward_critic_params]
        + [state.cost_critic_params[name] for name in state.cost_critic_params]
    )
    grads = torch.autograd.grad(actor_loss + critic_loss, flat_params, create_graph=True, allow_unused=False)
    if not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("functional_transaction_step: non-finite gradient")

    n_actor_params = len(actor_keys)
    actor_grads = list(grads[:n_actor_params])
    critic_grads = list(grads[n_actor_params:])
    clipped_actor_grads = _functional_clip_grad_norm(actor_grads, actor_clip_norm)
    all_grads = clipped_actor_grads + critic_grads

    ordered_exp_avg = [state.exp_avg[k] for k in flat_keys]
    ordered_exp_avg_sq = [state.exp_avg_sq[k] for k in flat_keys]
    new_step = state.step + 1
    new_params_flat, new_exp_avg_flat, new_exp_avg_sq_flat = _functional_adam_update(
        flat_params, all_grads, ordered_exp_avg, ordered_exp_avg_sq,
        new_step, state.lr, state.betas, state.eps,
    )

    new_params_by_key = dict(zip(flat_keys, new_params_flat))
    new_exp_avg = dict(zip(flat_keys, new_exp_avg_flat))
    new_exp_avg_sq = dict(zip(flat_keys, new_exp_avg_sq_flat))

    new_actor_params = [
        {name: new_params_by_key[f"actor{i}.{name}"] for name in names}
        for i, names in enumerate(actor_param_names)
    ]
    new_reward_critic_params = {
        name: new_params_by_key[f"reward_critic.{name}"] for name in state.reward_critic_params
    }
    new_cost_critic_params = {
        name: new_params_by_key[f"cost_critic.{name}"] for name in state.cost_critic_params
    }

    return FunctionalLearnerState(
        actor_params=new_actor_params, reward_critic_params=new_reward_critic_params,
        cost_critic_params=new_cost_critic_params, exp_avg=new_exp_avg, exp_avg_sq=new_exp_avg_sq,
        step=new_step, lr=state.lr, betas=state.betas, eps=state.eps,
    )


def run_q_step_functional_chain(
    state0: FunctionalLearnerState,
    *,
    actors: Sequence[nn.Module],
    reward_critic: nn.Module,
    cost_critic: nn.Module,
    batch_provider: Callable[[int], CityLearnPPOBatch],
    q_meta: int,
    rho_logits: torch.Tensor,
    lambda_per_agent: torch.Tensor,
    objective: PPOObjective,
    n_agents: int,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    actor_clip_norm: float | None = ACTOR_CLIP_NORM,
) -> FunctionalLearnerState:
    """Chain ``q_meta >= 2`` functional transaction steps, one fresh batch each.

    ``batch_provider(t)`` is called once per step (``t`` in ``0..q_meta-1``)
    so the caller controls whether each step's batch is freshly collected
    from an env or a pre-built fixed tape (e.g. for deterministic replay).
    ``state`` is carried forward *undetached* between steps -- the defining
    property of a multi-step unroll versus ``q_meta`` independent one-step
    probes.
    """
    if q_meta < 2:
        raise ValueError("run_q_step_functional_chain requires q_meta >= 2")
    state = state0
    for t in range(q_meta):
        batch = batch_provider(t)
        state = functional_transaction_step(
            state, actors=actors, reward_critic=reward_critic, cost_critic=cost_critic,
            batch=batch, rho_logits=rho_logits, lambda_per_agent=lambda_per_agent,
            objective=objective, n_agents=n_agents, action_scale=action_scale,
            action_bias=action_bias, actor_clip_norm=actor_clip_norm,
        )
    return state
