"""CityLearn actor/critic modules and the mutable PPO+critic update.

The shared, domain-agnostic ``PPOBatch``/``PPOObjective``/``simplex_from_logits``
primitives are reused unmodified (imported from :mod:`lira.ppo` and
:mod:`lira.responsibility`). Two CityLearn-specific details:

1. Each building's action is 3-dimensional (``dhw_storage``,
   ``electrical_storage``, ``cooling_device``) and ``cooling_device`` is
   bounded to ``[0, 1]`` rather than a symmetric range. One ``TanhNormal`` is
   used per action dimension; an explicit per-dimension ``(scale, bias)``
   affine (computed once from the real CityLearn action space) recenters the
   asymmetric dimension onto ``TanhNormal``'s symmetric-about-zero convention
   before sampling, and undoes it before scoring. The bias is a constant
   shift, so it exactly cancels in every PPO log-ratio.
2. The three arms differ only in how the length-``N`` ``lambda_per_agent``
   vector and ``rho`` are fed into the same per-agent
   ``PPOObjective.evaluate`` call:

   - **Uniform**: one shared scalar dual broadcast to every agent with uniform
     ``rho`` (``rho_i = 1/N``), so the effective penalty ``N * lambda * rho_i``
     reduces to exactly ``lambda`` for every agent. A ``SharedLambda`` with
     ``num_constraints=1`` holds this scalar.
   - **PAL** (per-agent Lagrangian): ``lambda_per_agent`` holds ``N``
     *independently* dual-ascended scalars, one per building, with no
     conservation simplex coupling them. With uniform ``rho`` the effective
     penalty of agent ``i`` reduces to exactly ``lambda_i``. A second
     ``SharedLambda`` with ``num_constraints=N`` holds these independent values.
   - **LiRA**: one shared scalar dual as in Uniform, with learned ``rho``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from torch.func import functional_call

from lira.ppo import PPOBatch, PPOObjective
from lira.responsibility import SharedLambda, simplex_from_logits

from .distributions import TanhNormal

HIDDEN = 64
ACTOR_CLIP_NORM = 1.0


def action_scale_bias(low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension ``TanhNormal`` (scale, bias) recovering ``[low, high]``.

    ``scale = (high - low) / 2``, ``bias = (high + low) / 2``, so
    ``bias + scale * tanh(pre) in (low, high)`` for every real ``pre``.
    """
    if torch.any(high <= low):
        raise ValueError("each action dimension needs high > low")
    return (high - low) / 2.0, (high + low) / 2.0


class Actor(nn.Module):
    """Per-building tanh-Gaussian actor over CityLearn's real action_dim.

    CityLearn actions are ``action_dim``-dimensional with heterogeneous
    bounds, so both the mean head and ``log_std`` are ``action_dim``-vectors
    (a diagonal Gaussian over the pre-tanh action), and sampling/scoring go
    through one independent ``TanhNormal`` per action dimension using that
    dimension's own real ``(scale, bias)``.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = HIDDEN) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = self.log_std.expand(obs.shape[0], self.action_dim)
        return mean, log_std


class Critic(nn.Module):
    """Centralized critic over (global state, joint action)."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN, output_dim: int = 1) -> None:
        if output_dim < 1:
            raise ValueError("critic output_dim must be positive")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return output.squeeze(-1) if output.shape[-1] == 1 else output


def sample_actions(
    actors: Sequence[nn.Module], obs: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor,
    *, deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample B x N x A raw actions and their B x N summed log-probs."""
    action_rows = []
    log_prob_rows = []
    for i, actor in enumerate(actors):
        mean, log_std = actor(obs[:, i])
        per_dim_actions = []
        per_dim_log_probs = []
        for d in range(actor.action_dim):
            dist = TanhNormal(mean[:, d], log_std[:, d], scale=float(scale[d]))
            centered = dist.sample(deterministic)
            per_dim_actions.append(centered + bias[d])
            per_dim_log_probs.append(dist.log_prob(centered))
        action_rows.append(torch.stack(per_dim_actions, dim=1))
        log_prob_rows.append(torch.stack(per_dim_log_probs, dim=1).sum(dim=1))
    return torch.stack(action_rows, dim=1), torch.stack(log_prob_rows, dim=1)


def action_log_probs(
    actors: Sequence[nn.Module], obs: torch.Tensor, actions: torch.Tensor,
    scale: torch.Tensor, bias: torch.Tensor,
) -> torch.Tensor:
    """Recompute B x N live summed log-probabilities for fixed raw actions."""
    rows = []
    for i, actor in enumerate(actors):
        mean, log_std = actor(obs[:, i])
        per_dim = []
        for d in range(actor.action_dim):
            dist = TanhNormal(mean[:, d], log_std[:, d], scale=float(scale[d]))
            per_dim.append(dist.log_prob(actions[:, i, d] - bias[d]))
        rows.append(torch.stack(per_dim, dim=1).sum(dim=1))
    return torch.stack(rows, dim=1)


@dataclass(frozen=True)
class CityLearnPPOBatch:
    """One frozen rollout batch for the CityLearn PPO+critic update.

    ``actions`` carries a trailing ``action_dim`` axis (CityLearn actions are
    per-building vectors, not scalars).
    """

    observations: torch.Tensor       # B x N x obs_dim
    actions: torch.Tensor            # B x N x action_dim
    state: torch.Tensor              # B x state_dim
    old_log_probs: torch.Tensor      # B x N   (summed over action_dim)
    reward_advantages: torch.Tensor  # B x N   (already normalized)
    cost_advantages: torch.Tensor    # B       (already normalized; shared across agents, K=1)
    value_target: torch.Tensor       # B       (raw sum-over-agents welfare regression target)
    cost_target: torch.Tensor        # B       (raw shared district cap-cost regression target)

    def validate(self, *, n_agents: int, action_dim: int) -> int:
        if self.observations.ndim != 3 or self.observations.shape[1] != n_agents:
            raise ValueError("observations must have shape B x N x D")
        b = int(self.observations.shape[0])
        if self.actions.shape != (b, n_agents, action_dim):
            raise ValueError("actions must have shape B x N x A")
        if self.old_log_probs.shape != (b, n_agents) or self.reward_advantages.shape != (b, n_agents):
            raise ValueError("old_log_probs/reward_advantages must have shape B x N")
        if self.state.ndim != 2 or self.state.shape[0] != b:
            raise ValueError("state must have shape B x state_dim")
        for name in ("cost_advantages", "value_target", "cost_target"):
            tensor = getattr(self, name)
            if tensor.shape != (b,):
                raise ValueError(f"{name} must have shape (B,)")
        return b


def citylearn_actor_critic_epoch_losses(
    *,
    new_log_probs: torch.Tensor,
    reward_critic_prediction: torch.Tensor,
    cost_critic_prediction: torch.Tensor,
    rho_logits: torch.Tensor,
    lambda_per_agent: torch.Tensor,
    batch: CityLearnPPOBatch,
    objective: PPOObjective,
    n_agents: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The one canonical per-epoch CityLearn PPO+critic loss.

    ``rho`` is recomputed from ``rho_logits`` every call via the shared
    ``simplex_from_logits`` (uniform for the Uniform/PAL arms, learned for
    LiRA). ``lambda_per_agent`` is a length-``n_agents`` vector; see the
    module docstring for how each arm constructs it.
    """
    rho = simplex_from_logits(rho_logits.reshape(1, -1), floor=0.0)
    ones = torch.ones_like(batch.old_log_probs[:, 0])
    zeros = torch.zeros_like(batch.old_log_probs[:, 0])
    cost_advantages_kb = batch.cost_advantages.reshape(1, -1)
    actor_loss = new_log_probs.new_zeros(())
    for agent in range(n_agents):
        agent_cost_advantages = cost_advantages_kb
        ppo_batch = PPOBatch(
            old_log_probs=batch.old_log_probs[:, agent],
            reward_advantages=batch.reward_advantages[:, agent],
            cost_advantages=agent_cost_advantages,
            active_masks=ones, factor_weights=ones, entropies=zeros,
        )
        lam = lambda_per_agent[agent].reshape(1)
        actor_loss = actor_loss + objective.evaluate(
            new_log_probs[:, agent], ppo_batch, rho, lam, agent,
        ).total / n_agents
    if cost_critic_prediction.shape != batch.cost_target.shape:
        raise ValueError("shared-cost critic must predict one native global cost per sample")
    cost_critic_loss = (cost_critic_prediction - batch.cost_target).pow(2).mean()
    critic_loss = (reward_critic_prediction - batch.value_target).pow(2).mean() + cost_critic_loss
    return actor_loss, critic_loss, actor_loss + critic_loss


def run_citylearn_ppo_epochs(
    *,
    epochs: int,
    n_agents: int,
    rho_logits: torch.Tensor,
    lambda_per_agent: torch.Tensor,
    batch: CityLearnPPOBatch,
    objective: PPOObjective,
    action_dim: int,
    recompute,
    take_step,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``epochs`` canonical CityLearn PPO+critic updates over one frozen batch.

    ``recompute()`` returns this epoch's live ``(new_log_probs, reward_pred,
    cost_pred)``; ``take_step(actor_loss, critic_loss)`` performs the update.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    batch.validate(n_agents=n_agents, action_dim=action_dim)
    last_actor_loss = None
    last_critic_loss = None
    for _ in range(epochs):
        new_log_probs, reward_pred, cost_pred = recompute()
        actor_loss, critic_loss, _total = citylearn_actor_critic_epoch_losses(
            new_log_probs=new_log_probs, reward_critic_prediction=reward_pred,
            cost_critic_prediction=cost_pred, rho_logits=rho_logits,
            lambda_per_agent=lambda_per_agent, batch=batch, objective=objective, n_agents=n_agents,
        )
        take_step(actor_loss, critic_loss)
        last_actor_loss, last_critic_loss = actor_loss, critic_loss
    if last_actor_loss is None or last_critic_loss is None:  # pragma: no cover
        raise RuntimeError("unreachable: epochs >= 1 must produce a loss")
    return last_actor_loss, last_critic_loss


def mutable_citylearn_ppo_update(
    *,
    actors: Sequence[nn.Module],
    reward_critic: nn.Module,
    cost_critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    rho_logits: torch.Tensor,
    lambda_per_agent: torch.Tensor,
    batch: CityLearnPPOBatch,
    objective: PPOObjective,
    epochs: int,
    n_agents: int,
    action_dim: int,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    actor_clip_norm: float = ACTOR_CLIP_NORM,
) -> torch.Tensor:
    """The live (non-meta) mutable PPO update: real modules, one shared Adam.

    Each epoch recomputes the actor+critic loss on the same frozen batch, clips the
    joint L2 norm across only the actor parameters, then takes one
    ``optimizer.step()`` advancing all actor+critic Adam slots together.
    """
    actor_parameters = [p for actor in actors for p in actor.parameters()]

    def recompute() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_log_probs = action_log_probs(actors, batch.observations, batch.actions, action_scale, action_bias)
        critic_in = torch.cat([batch.state, batch.actions.reshape(batch.actions.shape[0], -1)], dim=1)
        return new_log_probs, reward_critic(critic_in), cost_critic(critic_in)

    def take_step(actor_loss: torch.Tensor, critic_loss: torch.Tensor) -> None:
        optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(actor_parameters, actor_clip_norm)
        optimizer.step()

    actor_loss, _critic_loss = run_citylearn_ppo_epochs(
        epochs=epochs, n_agents=n_agents, rho_logits=rho_logits, lambda_per_agent=lambda_per_agent,
        batch=batch, objective=objective, action_dim=action_dim, recompute=recompute, take_step=take_step,
    )
    return actor_loss


def uniform_lambda_per_agent(shared_lambda: SharedLambda, n_agents: int) -> torch.Tensor:
    """Uniform arm: one shared scalar broadcast to every agent."""
    if shared_lambda.num_constraints != 1:
        raise ValueError("Uniform arm's SharedLambda must hold exactly one scalar")
    return shared_lambda.values.expand(n_agents)


def pal_lambda_per_agent(pal_lambda: SharedLambda, n_agents: int) -> torch.Tensor:
    """PAL arm: N independently dual-ascended scalars, no coupling."""
    if pal_lambda.num_constraints != n_agents:
        raise ValueError("PAL arm's SharedLambda must hold exactly n_agents independent values")
    return pal_lambda.values


def _functional_actor_log_probs(
    actors: Sequence[torch.nn.Module], param_dicts: Sequence[dict[str, torch.Tensor]],
    obs: torch.Tensor, actions: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor,
) -> torch.Tensor:
    """Recompute B x N summed log-probs using explicit (functional) parameter dicts."""
    rows = []
    for i, (actor, params) in enumerate(zip(actors, param_dicts)):
        mean, log_std = functional_call(actor, params, (obs[:, i],))
        per_dim = []
        for d in range(actor.action_dim):
            dist = TanhNormal(mean[:, d], log_std[:, d], scale=float(scale[d]))
            per_dim.append(dist.log_prob(actions[:, i, d] - bias[d]))
        rows.append(torch.stack(per_dim, dim=1).sum(dim=1))
    return torch.stack(rows, dim=1)
