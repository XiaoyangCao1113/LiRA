"""Functional q-step learner transaction for MABIM (DU + score correction).

Learner axis vs. data axis
---------------------------
LiRA separates two channels through which the shared responsibility
allocation ``rho`` (parameterized by softmax logits ``z``) affects terminal
welfare:

* *Learner axis* ("direct" term): holding the realized inner-tape actions
  fixed, ``rho`` still changes the PPO penalty each inner Adam step applies,
  which changes the trained parameters and therefore the terminal
  policy-score.  This is the pathwise/autodiff term through the q-step
  functional Adam trace.
* *Data axis* ("sampling correction" term): each inner tape is collected
  on-policy from the *currently trained* functional parameters, which are
  themselves a function of ``rho``.  Autodiff cannot see through the
  ``.sample()`` call, so this channel is recovered with a score-function
  (REINFORCE) correction: ``(terminal_welfare.detach() - baseline) *
  sum_{q,t,i} log pi(a_t^i | s_t^i; params_at_collection_time)`` (zero
  baseline, sum reduction, no normalization; see :class:`SamplingCorrectionSpec`).
  The leave-one-out baseline across independent replicates is applied in
  :mod:`lira.envs.mabim.meta_gradient`.

The functions here are used by the training loop
(:mod:`lira.envs.mabim.trainer`) and the held-out evaluation.
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.func import functional_call

from lira.ppo import PPOBatch

from .env import MABIMSharedCostEnv
from .learner import (
    SharedStaticCategoricalRunner,
    SharedStaticConfig,
    atomic_save_shared_static_checkpoint,
    shared_static_checkpoint,
)
from .live_dual import normalize_cost_advantages, projected_dual_update


@dataclass(frozen=True)
class SamplingCorrectionSpec:
    """Likelihood-ratio correction contract: zero baseline, unnormalized score sum."""

    enabled: bool = True
    baseline: float = 0.0
    baseline_kind: str = "zero"
    score_reduction: str = "sum"
    normalization: str = "none"


SC_SPEC = SamplingCorrectionSpec()


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _bind_runner_rng(runner, seed: int) -> None:
    """Give one runner an RNG stream without perturbing another runner.

    MABIM uses process-global Python/NumPy/Torch RNGs.  Multiple in-process
    meta replicas therefore are not independent merely because the factory
    called ``manual_seed`` once: the last constructed replica otherwise owns
    the process-global stream.  Store a complete stream on each runner and
    swap it around every rollout instead.
    """
    ambient = _capture_rng_state()
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    runner._mabim_rng_state = _capture_rng_state()
    _restore_rng_state(ambient)


def _phase_seed(seed: int, phase: str, cycle: int, replica: int = 0) -> int:
    raw = f"mabim-v2|{int(seed)}|{phase}|{int(cycle)}|{int(replica)}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**31 - 1)


def _optimizer_state_digest(states: Mapping[str, tuple[torch.Tensor, torch.Tensor, int]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(states):
        exp_avg, exp_avg_sq, step = states[name]
        digest.update(name.encode())
        digest.update(int(step).to_bytes(8, "little", signed=False))
        digest.update(exp_avg.detach().cpu().contiguous().numpy().tobytes())
        digest.update(exp_avg_sq.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()



def env_runner(env_root: str | os.PathLike | None = None) -> SharedStaticCategoricalRunner:
    """Build the all-agent MABIM environment and the shared categorical learner."""
    env = MABIMSharedCostEnv(env_root)
    mask = torch.ones(2, 400, dtype=torch.bool)
    rho = mask.double() / 400.0
    cfg = SharedStaticConfig(400, 2, 4, categories=3, horizon=60,
                             n_warehouses=2, n_skus=200, hidden=32)
    runner = SharedStaticCategoricalRunner(env, cfg, rho, mask,
        [0] * 200 + [1] * 200, list(range(200)) * 2)
    return runner


@dataclass(frozen=True)
class Tape:
    obs: torch.Tensor; actions: torch.Tensor; old_log_probs: torch.Tensor
    reward_adv: torch.Tensor; cost_adv: torch.Tensor
    reward_ret: torch.Tensor; cost_ret: torch.Tensor
    masks: torch.Tensor; rewards: torch.Tensor; costs: torch.Tensor; dones: torch.Tensor


@dataclass(frozen=True)
class TerminalScore:
    """Raw terminal welfare and its joint-policy likelihood-ratio objective."""
    welfare: torch.Tensor
    objective: torch.Tensor


def capture(runner):
    params={}; states={}
    for name, module in (("actor", runner.actor), ("critic", runner.critic)):
        for n,p in module.named_parameters():
            key=f"{name}.{n}"; params[key]=p.detach().clone().requires_grad_(True)
            raw=runner.opt.state.get(p,{})
            step=raw.get("step",0); step=int(step.item() if isinstance(step,torch.Tensor) else step)
            states[key]=(torch.as_tensor(raw.get("exp_avg",torch.zeros_like(p))).detach().clone(), torch.as_tensor(raw.get("exp_avg_sq",torch.zeros_like(p))).detach().clone(), step)
    return params, states


def load_functional_state(runner, params, states) -> None:
    """Commit functional parameters *and* Adam state to ``runner``.

    Functional Adam is only faithful across blocks when its moments and step
    counter advance with the parameters.  Loading modules alone silently
    restarted every later block from the checkpoint optimizer state.
    """
    actor = {k.removeprefix("actor."): v.detach() for k, v in params.items() if k.startswith("actor.")}
    critic = {k.removeprefix("critic."): v.detach() for k, v in params.items() if k.startswith("critic.")}
    runner.actor.load_state_dict(actor)
    runner.critic.load_state_dict(critic)
    named = {
        **{f"actor.{name}": parameter for name, parameter in runner.actor.named_parameters()},
        **{f"critic.{name}": parameter for name, parameter in runner.critic.named_parameters()},
    }
    runner.opt.state.clear()
    for name, parameter in named.items():
        exp_avg, exp_avg_sq, step = states[name]
        runner.opt.state[parameter] = {
            "step": torch.tensor(float(step)),
            "exp_avg": exp_avg.detach().clone().to(parameter),
            "exp_avg_sq": exp_avg_sq.detach().clone().to(parameter),
        }


def adam(params, states, loss, lr, betas, eps):
    grads=torch.autograd.grad(loss, tuple(params.values()), create_graph=True, allow_unused=True)
    out={}; nxt={}; b1,b2=betas
    for (k,v),g in zip(params.items(),grads,strict=True):
        m0,v0,step0=states[k]; g=torch.zeros_like(v) if g is None else g; step=step0+1
        m=b1*m0+(1-b1)*g; vv=b2*v0+(1-b2)*g.square()
        out[k]=v-lr*(m/(1-b1**step))/((vv/(1-b2**step)+eps**2).sqrt())
        nxt[k]=(m,vv,step)
    return out,nxt


def _actor_params(params: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix("actor."): v for k, v in params.items() if k.startswith("actor.")}


def _critic_params(params: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix("critic."): v for k, v in params.items() if k.startswith("critic.")}


def functional_actor_logits(runner, params, obs, ctx):
    logits, _ = functional_call(runner.actor, _actor_params(params), (obs, runner.warehouse_ids, runner.sku_ids, ctx))
    return logits


def functional_critic_values(runner, params, obs_TND, context_TK):
    t, n, _ = obs_TND.shape; k = runner.config.n_constraints
    state = torch.cat([obs_TND.reshape(t, -1), context_TK.reshape(t, k)], -1)
    out = functional_call(runner.critic, _critic_params(params), (state,)).reshape(t, n, 1 + k)
    return out[..., 0], out[..., 1:]


def actor_critic_forward(runner, params, tape):
    T=tape.actions.shape[0]; ctx=torch.zeros((T,runner.config.n_constraints),dtype=torch.float64)
    logits = functional_actor_logits(runner, params, tape.obs, ctx)
    values = functional_call(runner.critic, _critic_params(params), (torch.cat([tape.obs.reshape(T,-1),ctx],-1),)).reshape(T,runner.config.n_agents,1+runner.config.n_constraints)
    return logits,values


def collect_onpolicy(runner, params) -> tuple[Tape, torch.Tensor]:
    """Collect one fresh episode sampled from ``params``, on-policy.

    Returns the frozen ``Tape`` (every field detached, matching PPO's
    fixed-batch contract used by ``loss``) and a *differentiable* joint
    behavior log-probability: the sum, over every sampled step/agent, of
    ``log pi(a | s; params)``.  Actions themselves are sampled (no pathwise
    gradient through the environment), but the log-probability of the
    realized action remains a function of ``params`` -- and therefore, when
    ``params`` is itself the output of a rho-dependent q-step Adam trace, a
    function of rho.  This is the data-axis score-function term.
    """
    cfg = runner.config; zero = torch.zeros((1, cfg.n_constraints), dtype=torch.float64)
    ambient_rng = _capture_rng_state()
    if not hasattr(runner, "_mabim_rng_state"):
        runner._mabim_rng_state = ambient_rng
    _restore_rng_state(runner._mabim_rng_state)
    try:
        obs = [torch.as_tensor(runner.env.reset(), dtype=torch.float64)]
        actions=[]; old=[]; behavior=[]; masks=[]; rewards=[]; costs=[]; done=False
        for _ in range(cfg.horizon):
            o = obs[-1].unsqueeze(0); mask = torch.as_tensor(runner.env.action_masks, dtype=torch.bool)
            if mask.ndim == 3 and mask.shape[1] == 1: mask = mask[:, 0, :]
            logits = functional_actor_logits(runner, params, o, zero)
            dist = torch.distributions.Categorical(logits=logits[0].masked_fill(~mask, float("-inf")))
            action = dist.sample(); lp = dist.log_prob(action)
            actions.append(action.detach()); old.append(lp.detach()); behavior.append(lp)
            masks.append(mask.detach())
            nxt, rew, cost, done, _ = runner.env.step(action)
            rewards.append(torch.as_tensor(rew, dtype=torch.float64)); costs.append(torch.as_tensor(cost, dtype=torch.float64))
            obs.append(torch.as_tensor(nxt, dtype=torch.float64))
            if done: break
    finally:
        runner._mabim_rng_state = _capture_rng_state()
        _restore_rng_state(ambient_rng)
    O=torch.stack(obs[:-1]); A=torch.stack(actions); LP=torch.stack(old); M=torch.stack(masks)
    R=torch.stack(rewards); C=torch.stack(costs); T=A.shape[0]; ctx=zero.expand(T, cfg.n_constraints)
    behavior_score = torch.stack(behavior).sum()
    with torch.no_grad():
        vr, vc = functional_critic_values(runner, params, O, ctx)
        br, bc = functional_critic_values(runner, params, obs[-1].unsqueeze(0), zero)
        dones=torch.zeros(T, dtype=torch.bool)
        if done: dones[-1]=True
        ar, rr=runner._gae(R,vr,torch.zeros_like(br[0]) if done else br[0],dones,cfg.gamma,cfg.gae_lambda)
        ac, rc=runner._gae(C.unsqueeze(1).expand(T,cfg.n_agents,cfg.n_constraints),vc,torch.zeros_like(bc[0]) if done else bc[0],dones,cfg.gamma,cfg.gae_lambda)
    tape = Tape(O.detach(), A.detach(), LP.detach(), ar.detach(), ac.detach(), rr.detach(), rc.detach(), M.detach(), R.detach(), C.detach(), dones.detach())
    return tape, behavior_score


def loss(runner, params, rho_logits, tape, dual, cost_budgets=None):
    raw, values=actor_critic_forward(runner,params,tape); T=tape.actions.shape[0]
    if cost_budgets is None:
        cost_adv = tape.cost_adv
    else:
        cost_adv = normalize_cost_advantages(tape.cost_adv, cost_budgets)
    rho=torch.softmax(rho_logits,dim=-1); actor=torch.zeros((),dtype=torch.float64)
    for i in range(runner.config.n_agents):
        d=torch.distributions.Categorical(logits=raw[:,i,:].masked_fill(~tape.masks[:,i,:],float("-inf")))
        lam = dual[:,i] if dual.ndim == 2 else dual
        penalties=lam
        combined=tape.reward_adv[:,i]-torch.einsum("k,kb->b",penalties,cost_adv[:,i,:].T)
        batch=PPOBatch(tape.old_log_probs[:,i],combined*0+tape.reward_adv[:,i],cost_adv[:,i,:].T,torch.ones(T,dtype=torch.float64),torch.ones(T,dtype=torch.float64),d.entropy())
        if dual.ndim == 2:
            # direct independent-dual penalty, avoiding false shared-simplex semantics
            batch=PPOBatch(tape.old_log_probs[:,i],tape.reward_adv[:,i],cost_adv[:,i,:].T,torch.ones(T,dtype=torch.float64),torch.ones(T,dtype=torch.float64),d.entropy())
            ratio=torch.exp(d.log_prob(tape.actions[:,i])-batch.old_log_probs); adv=combined
            actor=actor-(torch.minimum(ratio*adv,torch.clamp(ratio,.8,1.2)*adv).mean()+runner.config.entropy_coefficient*d.entropy().mean())/runner.config.n_agents
        else:
            actor=actor+runner.ppo.evaluate(d.log_prob(tape.actions[:,i]),batch,rho,dual,i).total/runner.config.n_agents
    critic=torch.nn.functional.mse_loss(values[...,0],tape.reward_ret)+torch.nn.functional.mse_loss(values[...,1:],tape.cost_ret)
    return actor+runner.config.critic_loss_coefficient*critic


def terminal_score(runner, params, tape) -> TerminalScore:
    raw,_=actor_critic_forward(runner,params,tape); scores=[]
    for i in range(runner.config.n_agents):
        d=torch.distributions.Categorical(logits=raw[:,i,:].masked_fill(~tape.masks[:,i,:],float("-inf")))
        scores.append(d.log_prob(tape.actions[:,i]))
    r=tape.rewards.mean(-1); ret=torch.zeros_like(r); running=torch.zeros((),dtype=r.dtype)
    for j in range(len(r)-1,-1,-1): running=r[j]+(1-tape.dones[j].double())*running; ret[j]=running
    welfare = r.sum()
    objective = (ret*torch.stack(scores,-1).sum(-1)).sum()
    return TerminalScore(welfare, objective)


def du_sc_outer_gradient(
    runner, dual, q, *, z_init=None, return_score_sum: bool = False,
    cost_budgets=None, dual_eta: float | None = None, dual_max: float = 100.0,
):
    """On-policy q-step lookahead returning the DU and score-correction gradients.

    Runs ``q`` on-policy inner functional Adam updates from the runner's
    current state with ``z`` (rho logits) as a live leaf requiring grad,
    differentiable end to end.  When ``dual_eta`` is given, the shared
    multipliers inside the lookahead follow the same projected
    ``C_k/d_k - 1`` update as the live learner.  Returns the leaf ``z``, the
    three gradients (learner-axis "direct", data-axis "correction", and
    their sum "corrected"), the trained parameters, and the terminal score.

    ``z_init`` seeds the ``z`` leaf (default: zero logits, i.e. uniform rho).
    With ``return_score_sum=True`` the live (rho-differentiable) ``score_sum``
    is appended as a 7th return value so that a leave-one-out baseline can
    recombine welfare and score across replicates.
    """
    params, states = capture(runner)
    if z_init is None:
        z = torch.zeros((runner.config.n_constraints, runner.config.n_agents), dtype=torch.float64, requires_grad=True)
    else:
        z = z_init.detach().clone().requires_grad_(True)
    cur, st = params, states
    dual_cur = dual.detach().clone()
    if dual_eta is not None and cost_budgets is None:
        raise ValueError("live dual_eta requires cost_budgets for C_k/d_k normalization")
    scores: list[torch.Tensor] = []
    for _ in range(q):
        tape, score = collect_onpolicy(runner, cur)
        scores.append(score)
        cur, st = adam(
            cur, st, loss(runner, cur, z, tape, dual_cur, cost_budgets=cost_budgets),
            runner.config.actor_lr, (.9, .999), 1e-8,
        )
        if dual_eta is not None:
            dual_cur, _ = projected_dual_update(
                dual_cur, tape.costs.sum(0).detach(), cost_budgets,
                dual_eta, dual_max=dual_max,
            )
    terminal_tape, _ = collect_onpolicy(runner, cur)
    term = terminal_score(runner, cur, terminal_tape)
    if not SC_SPEC.enabled or SC_SPEC.baseline_kind != "zero" or SC_SPEC.score_reduction != "sum":
        raise RuntimeError("MABIM DU+SC estimation requires the frozen zero-baseline sum-score contract")
    baseline = torch.as_tensor(SC_SPEC.baseline, dtype=term.welfare.dtype)
    score_sum = torch.stack(scores).sum() if scores else torch.zeros_like(term.welfare)
    correction = (term.welfare.detach() - baseline) * score_sum
    corrected = term.objective + correction

    def _grad(target: torch.Tensor, *, retain: bool) -> torch.Tensor:
        if not target.requires_grad:
            return torch.zeros_like(z)
        g = torch.autograd.grad(target, z, retain_graph=retain, allow_unused=True)[0]
        return torch.zeros_like(z) if g is None else g

    direct_grad = _grad(term.objective, retain=True)
    correction_grad = _grad(correction, retain=True)
    corrected_grad = _grad(corrected, retain=return_score_sum)
    if return_score_sum:
        return z, direct_grad, correction_grad, corrected_grad, cur, term, score_sum
    return z, direct_grad, correction_grad, corrected_grad, cur, term


def save_arm(runner, params, states, rho, out, seed, arm, dual_meta, counters, source_digests=None):
    load_functional_state(runner, params, states)
    runner.rho=rho.detach().clone()
    output_digests = source_digests or {
        "learner.py": sha(Path(__file__).with_name("learner.py")),
        "transaction.py": sha(Path(__file__)),
    }
    payload=shared_static_checkpoint(runner,counters=counters,source_digests=output_digests)
    if hasattr(runner, "_mabim_rng_state"):
        payload["rng"] = {
            "python": runner._mabim_rng_state["python"],
            "numpy": runner._mabim_rng_state["numpy"],
            "torch_cpu": runner._mabim_rng_state["torch"].clone(),
        }
    out.parent.mkdir(parents=True,exist_ok=True); atomic_save_shared_static_checkpoint(out,payload)
    return {"checkpoint":str(out),"checkpoint_sha256":sha(out),"seed":seed,"arm":arm,"dual_meta":dual_meta,"counters":counters}
