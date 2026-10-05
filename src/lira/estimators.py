"""Domain-agnostic DU + leave-one-out score-correction (LOO-SC) outer gradient.

Each of ``M`` independent lookahead replicates supplies four attributes:

- ``du``: the differentiable direct (terminal-score) term, a function of
  that replicate's own ``rho_logits`` leaf;
- ``score``: the differentiable training-data sampling score;
- ``welfare``: the replicate's detached realized welfare;
- ``rho_logits``: the replicate's own independent leaf tensor.

The per-replicate objective is ``du + stopgrad(welfare - b_loo) * score``
where ``b_loo`` is the mean welfare of the *other* replicates; the outer
gradient is the mean over replicates of its gradient with respect to the
replicate's ``rho_logits`` (Eq. (loo) in the paper). The allocation logits are
then moved by one Adam *ascent* step.

These functions only operate on scalar tensors and the replicate leaves, so
they are shared by every environment adapter.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch


def leave_one_out_baselines(welfare_values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Each replicate's cross-fitted baseline: mean welfare of the OTHERS."""
    m = len(welfare_values)
    if m < 2:
        raise ValueError("a leave-one-out baseline requires at least two independent replicates")
    detached = [value.detach() for value in welfare_values]
    if any(value.ndim != 0 for value in detached):
        raise ValueError("welfare values must be scalar tensors")
    total = sum(detached)
    return tuple((total - value) / (m - 1) for value in detached)


def lira_gradient(replicates: Sequence[Any]) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean DU+LOO-SC gradient w.r.t. rho_logits across M independent replicates.

    ``replicates`` may be any objects exposing ``du``/``score``/``welfare``/
    ``rho_logits`` as described in the module docstring.
    """
    if len(replicates) < 2:
        raise ValueError("LiRA needs at least two independent replicates")
    baselines = leave_one_out_baselines([rep.welfare for rep in replicates])
    grads = []
    du_values, sc_values, welfare_values = [], [], []
    for rep, baseline in zip(replicates, baselines):
        loo_sc = (rep.welfare.detach() - baseline).detach() * rep.score
        objective = rep.du + loo_sc
        (grad,) = torch.autograd.grad(objective, rep.rho_logits, retain_graph=False, allow_unused=True)
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError("LiRA DU+LOO-SC gradient is disconnected or non-finite")
        grads.append(grad.detach())
        du_values.append(float(rep.du.detach()))
        sc_values.append(float(loo_sc.detach()))
        welfare_values.append(float(rep.welfare.detach()))
    mean_grad = torch.stack(grads).mean(dim=0)
    if not torch.isfinite(mean_grad).all():
        raise RuntimeError("LiRA mean outer gradient is invalid")
    return mean_grad, {
        "du_mean": sum(du_values) / len(du_values),
        "loo_sc_mean": sum(sc_values) / len(sc_values),
        "welfare_mean": sum(welfare_values) / len(welfare_values),
        "gradient_norm": float(mean_grad.norm()),
    }


def apply_outer_ascent_step(
    rho_logits: torch.Tensor, outer_optimizer: torch.optim.Optimizer, mean_grad: torch.Tensor,
    grad_clip_norm: float | None = None,
) -> None:
    """One outer Adam step ASCENDING the DU+LOO-SC objective.

    Adam's default semantics are descent (``param -= lr * f(grad)``); setting
    ``.grad = -mean_grad`` before ``optimizer.step()`` turns that into ascent
    on the DU+LOO-SC utility.

    ``grad_clip_norm``, if given, caps the L2 norm of ``rho_logits.grad``
    before the optimizer step -- a trust region on the outer step so that a
    single noisy DU+LOO-SC estimate cannot cause a runaway simplex jump.
    """
    outer_optimizer.zero_grad()
    rho_logits.grad = -mean_grad
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_([rho_logits], max_norm=grad_clip_norm)
    outer_optimizer.step()
