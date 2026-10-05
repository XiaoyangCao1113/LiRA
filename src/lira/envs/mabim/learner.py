"""Shared-parameter categorical actor-critic for the N=400 MABIM agents.

A single FC-GRU actor (with warehouse and SKU embeddings) and a centralized
critic are shared by all agents.  The LiRA, Uniform and PAL training loops use the functional learner in
:mod:`lira.envs.mabim.transaction`, which reuses these modules and the
checkpoint format defined here.  ``response_method='disabled'`` is the only
accepted mode.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import copy
import hashlib
import os
from pathlib import Path
import random
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn
from lira.ppo import PPOBatch, PPOConfig, PPOObjective

class SharedFCGRUActor(nn.Module):
    def __init__(self, obs_dim, n_agents, n_constraints, n_warehouses, n_skus, categories=3, hidden=32, embed=8):
        super().__init__(); self.n_agents=n_agents; self.categories=categories; self.context_dim=n_constraints
        self.warehouse_embedding=nn.Embedding(n_warehouses, embed)
        self.sku_embedding=nn.Embedding(n_skus, embed)
        self.gru=nn.GRUCell(obs_dim + 2*embed + n_constraints, hidden)
        self.head=nn.Linear(hidden, categories)
        self.hidden=hidden
    def forward(self, obs, warehouse_ids, sku_ids, resource_context, h=None):
        # obs: T x N x D; resource_context: T x K, shared across agents, one slot per constraint (never averaged away).
        if obs.ndim != 3: raise ValueError("obs must be T x N x D")
        t,n,_=obs.shape
        if n != self.n_agents: raise ValueError("agent count mismatch")
        if resource_context.numel() != t*self.context_dim: raise ValueError("resource context must carry K unaveraged values per step")
        ctx=resource_context.reshape(t,1,-1).expand(t,n,-1)
        e=torch.cat([self.warehouse_embedding(warehouse_ids), self.sku_embedding(sku_ids)],-1)
        x=torch.cat([obs,e.unsqueeze(0).expand(t,-1,-1),ctx],-1).reshape(t*n,-1)
        if h is None: h=x.new_zeros((t*n,self.hidden))
        h=self.gru(x,h)
        return self.head(h).reshape(t,n,self.categories), h.reshape(t,n,self.hidden)

@dataclass
class SharedStaticConfig:
    n_agents:int; n_constraints:int; obs_dim:int; categories:int=3; horizon:int=60
    n_warehouses:int=2; n_skus:int=200; hidden:int=32; actor_lr:float=.01
    gamma:float=1.0; gae_lambda:float=.95; clip_ratio:float=.2; entropy_coefficient:float=.01
    critic_loss_coefficient:float=.5; lambda_value:tuple=(.01,.01)
    response_method:str="disabled"; device:str="cpu"

class SharedStaticCategoricalRunner:
    def __init__(self, env, config, rho, support_mask, warehouse_ids, sku_ids):
        if config.n_agents < 1 or config.n_constraints < 1: raise ValueError("invalid N/K")
        if config.device != "cpu": raise ValueError("the MABIM learner is CPU-only")
        if config.response_method != "disabled": raise ValueError("shared categorical runner is static-only; response_method must be 'disabled'")
        self.env,self.config=env,config
        self.rho=torch.as_tensor(rho,dtype=torch.float64)
        self.support_mask=torch.as_tensor(support_mask,dtype=torch.bool)
        if self.rho.shape != (config.n_constraints,config.n_agents) or self.support_mask.shape != self.rho.shape: raise ValueError("rho/support shape")
        if torch.any(self.rho < 0) or not torch.allclose(self.rho.sum(1),torch.ones(config.n_constraints,dtype=torch.float64)): raise ValueError("rho simplex")
        if torch.any((self.rho > 0) & ~self.support_mask): raise ValueError("rho outside support")
        self.actor=SharedFCGRUActor(config.obs_dim,config.n_agents,config.n_constraints,config.n_warehouses,config.n_skus,config.categories,config.hidden).double()
        self.critic=nn.Sequential(nn.Linear(config.n_agents*config.obs_dim+config.n_constraints,config.hidden),nn.Tanh(),nn.Linear(config.hidden,config.n_agents*(1+config.n_constraints))).double()
        self.opt=torch.optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()),lr=config.actor_lr)
        self.ppo=PPOObjective(PPOConfig(config.clip_ratio,config.entropy_coefficient,0.0))
        self.warehouse_ids=torch.as_tensor(warehouse_ids,dtype=torch.long); self.sku_ids=torch.as_tensor(sku_ids,dtype=torch.long)

    def _critic_values(self, obs_TND, context_TK):
        """Centralized reward + K-cost value predictions, per agent, per step."""
        t,n,_=obs_TND.shape; k=self.config.n_constraints
        state=torch.cat([obs_TND.reshape(t,-1),context_TK.reshape(t,k)],-1)
        out=self.critic(state).reshape(t,n,1+k)
        return out[...,0], out[...,1:]

    @staticmethod
    def _gae(rewards, values, bootstrap, dones, gamma, lam):
        t=rewards.shape[0]; advantages=torch.zeros_like(rewards)
        next_value=bootstrap; next_adv=torch.zeros_like(bootstrap)
        for i in range(t-1,-1,-1):
            mask=0.0 if bool(dones[i]) else 1.0
            delta=rewards[i]+gamma*next_value*mask-values[i]
            next_adv=delta+gamma*lam*mask*next_adv
            advantages[i]=next_adv; next_value=values[i]
        return advantages, advantages+values

    def update(self):
        cfg=self.config; k=cfg.n_constraints
        # No observation field carries pre-action resource state, so the same
        # zero context (never a post-action cost) feeds every actor call below,
        # both during rollout sampling and PPO recomputation.
        zero_ctx=torch.zeros((1,k),dtype=torch.float64)
        obs0=self.env.reset(); obs=[torch.as_tensor(obs0,dtype=torch.float64)]; acts=[]; oldlp=[]; masks=[]; rewards=[]; costs=[]
        done=False
        for _ in range(cfg.horizon):
            o=obs[-1].unsqueeze(0); mask=torch.as_tensor(self.env.action_masks,dtype=torch.bool)
            if mask.ndim == 3 and mask.shape[1] == 1: mask = mask[:,0,:]
            logits,_=self.actor(o,self.warehouse_ids,self.sku_ids,zero_ctx); logits=logits[0].masked_fill(~mask,float('-inf'))
            dist=torch.distributions.Categorical(logits=logits); a=dist.sample(); lp=dist.log_prob(a).detach()
            no,r,c,done,info=self.env.step(a); obs.append(torch.as_tensor(no,dtype=torch.float64)); acts.append(a); oldlp.append(lp); masks.append(mask); rewards.append(torch.as_tensor(r,dtype=torch.float64)); costs.append(torch.as_tensor(c,dtype=torch.float64))
            if done: break
        T=len(acts); O=torch.stack(obs[:-1]); A=torch.stack(acts); LP=torch.stack(oldlp); R=torch.stack(rewards); C=torch.stack(costs); M=torch.stack(masks)
        context_seq=zero_ctx.expand(T,k)

        with torch.no_grad():
            v_reward_t,v_cost_t=self._critic_values(O,context_seq)
            final_reward_b,final_cost_b=self._critic_values(obs[-1].unsqueeze(0),zero_ctx)
            bootstrap_reward=torch.zeros(cfg.n_agents,dtype=torch.float64) if done else final_reward_b[0]
            bootstrap_cost=torch.zeros(cfg.n_agents,k,dtype=torch.float64) if done else final_cost_b[0]
            dones=torch.zeros(T,dtype=torch.bool)
            if done: dones[-1]=True
            adv_reward,ret_reward=self._gae(R,v_reward_t,bootstrap_reward,dones,cfg.gamma,cfg.gae_lambda)
            cost_expanded=C.unsqueeze(1).expand(T,cfg.n_agents,k)
            adv_cost,ret_cost=self._gae(cost_expanded,v_cost_t,bootstrap_cost,dones,cfg.gamma,cfg.gae_lambda)

        self.opt.zero_grad()
        logits,_=self.actor(O,self.warehouse_ids,self.sku_ids,context_seq)
        v_reward,v_cost=self._critic_values(O,context_seq)
        actor_total=torch.zeros((),dtype=torch.float64); max_abs_log_ratio=torch.zeros((),dtype=torch.float64)
        for i in range(cfg.n_agents):
            dist=torch.distributions.Categorical(logits=logits[:,i,:]); nlp=dist.log_prob(A[:,i].reshape(-1)); ent=dist.entropy()
            batch=PPOBatch(LP[:,i],adv_reward[:,i],adv_cost[:,i,:].T,self._ones(T),self._ones(T),ent)
            actor_total=actor_total+self.ppo.evaluate(nlp,batch,self.rho,torch.as_tensor(cfg.lambda_value,dtype=torch.float64),i).total/cfg.n_agents
            max_abs_log_ratio=torch.maximum(max_abs_log_ratio,(nlp-LP[:,i]).abs().max())
        critic_loss=torch.nn.functional.mse_loss(v_reward,ret_reward)+torch.nn.functional.mse_loss(v_cost,ret_cost)
        total=actor_total+cfg.critic_loss_coefficient*critic_loss
        total.backward(); self.opt.step()
        return {
            "steps":T,"loss":float(total.detach()),"actor_loss":float(actor_total.detach()),
            "critic_loss":float(critic_loss.detach()),"shared_parameter_update":True,"rho":self.rho.tolist(),
            "raw_cost":float(C.sum()),"raw_reward":R.sum(0).tolist(),
            "initial_log_ratio_max_abs":float(max_abs_log_ratio.detach()),
        }

    @staticmethod
    def _ones(t): return torch.ones(t,dtype=torch.float64)


_CHECKPOINT_SCHEMA = "mabim-shared-static-checkpoint-v1"
_COUNTER_KEYS = {"updates", "steps", "seed"}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shared_static_checkpoint(runner: SharedStaticCategoricalRunner, *, counters: Mapping[str, int], source_digests: Mapping[str, str]) -> dict[str, Any]:
    """Snapshot every mutable learner state at an episode/reset boundary."""
    if set(counters) != _COUNTER_KEYS or any(not isinstance(counters[key], int) or counters[key] < 0 for key in counters):
        raise ValueError("checkpoint counters must be nonnegative updates/steps/seed integers")
    if not source_digests or any(not isinstance(key, str) or not isinstance(value, str) or len(value) != 64 for key, value in source_digests.items()):
        raise ValueError("checkpoint requires named SHA-256 source digests")
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "config": asdict(runner.config),
        "rho": _cpu_tree(runner.rho),
        "support_mask": _cpu_tree(runner.support_mask),
        "lambda_value": tuple(runner.config.lambda_value),
        "actor": _cpu_tree(runner.actor.state_dict()),
        "critic": _cpu_tree(runner.critic.state_dict()),
        "optimizer": _cpu_tree(runner.opt.state_dict()),
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu().clone(),
        },
        "counters": dict(counters),
        "source_digests": dict(source_digests),
    }


def atomic_save_shared_static_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> str:
    if payload.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError("refusing to write malformed MABIM checkpoint")
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), tmp)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists(): tmp.unlink()
    return _sha256(target)


def load_shared_static_checkpoint(runner: SharedStaticCategoricalRunner, path: str | Path, *, expected_source_digests: Mapping[str, str]) -> dict[str, int]:
    """Strictly restore a boundary snapshot; reject semantic/source drift."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    expected = {"schema", "config", "rho", "support_mask", "lambda_value", "actor", "critic", "optimizer", "rng", "counters", "source_digests"}
    if not isinstance(payload, dict) or set(payload) != expected or payload["schema"] != _CHECKPOINT_SCHEMA:
        raise RuntimeError("malformed MABIM checkpoint")
    if payload["config"] != asdict(runner.config) or tuple(payload["lambda_value"]) != tuple(runner.config.lambda_value):
        raise RuntimeError("MABIM checkpoint config/lambda drift")
    if payload["source_digests"] != dict(expected_source_digests):
        raise RuntimeError("MABIM checkpoint source digest drift")
    if not torch.equal(torch.as_tensor(payload["rho"]), runner.rho) or not torch.equal(torch.as_tensor(payload["support_mask"], dtype=torch.bool), runner.support_mask):
        raise RuntimeError("MABIM checkpoint rho/support drift")
    counters = payload["counters"]
    if set(counters) != _COUNTER_KEYS or any(not isinstance(counters[key], int) or counters[key] < 0 for key in counters):
        raise RuntimeError("MABIM checkpoint counters malformed")
    runner.actor.load_state_dict(payload["actor"], strict=True)
    runner.critic.load_state_dict(payload["critic"], strict=True)
    runner.opt.load_state_dict(payload["optimizer"])
    random.setstate(payload["rng"]["python"]); np.random.set_state(payload["rng"]["numpy"]); torch.set_rng_state(payload["rng"]["torch_cpu"])
    return dict(counters)
