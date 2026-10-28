"""The policy's gradient through one virtual step of the task model (paper Eq. 5 and 8).

    ŵ(θ) = w - α ∇_w L_tr(w, θ)          (Eq. 5, a plain SGD step)
    ∇_θ L_val(ŵ(θ)) = -α ∇_θ (∇_w L_tr(w, θ) · v),   v = ∇_ŵ L_val(ŵ)

L_tr is the relaxed batch's loss (it depends on θ through the blend), L_val the
loss on a validation batch (no θ). MADAug unrolls the step with ``higher`` around the
task model's own optimizer (SGD with momentum and weight decay); the paper's Eq. 5 is
plain SGD at the learning rate α, which is what this takes.

``mode``:

- ``"exact"``: the right-hand side above. ∇_w L_tr is taken with ``create_graph``, v at
  ŵ first-order, then one backward of the dot product: a Hessian-vector product, the
  same number ``higher`` gives for a plain SGD step. About twice a training step's memory.
- ``"finite_diff"``: DARTS' approximation of that product (Liu et al. 2019, Eq. 8):
  ``-α (∇_θ L_tr(w + εv) - ∇_θ L_tr(w - εv)) / 2ε``, ε = ``eps / ‖v‖``. Only first-order
  passes, so a training step's memory; O(ε²) error. For backbones the exact form does
  not fit.

The task model is called functionally (``torch.func.functional_call``) with the weights
each loss is given, so its parameters and their ``.grad`` are never touched.
"""

from typing import Callable, Dict, List, Sequence

import torch

__all__ = ["MODES", "policy_gradient"]

MODES = ("exact", "finite_diff")

Weights = Dict[str, torch.Tensor]


def _grads(output, inputs: Sequence[torch.Tensor], **kwargs) -> List[torch.Tensor]:
    """``torch.autograd.grad`` with zeros for the inputs ``output`` doesn't reach."""
    grads = torch.autograd.grad(output, list(inputs), allow_unused=True, **kwargs)
    return [torch.zeros_like(x) if g is None else g for x, g in zip(inputs, grads)]


def policy_gradient(
    model: torch.nn.Module,
    theta: Sequence[torch.Tensor],
    train_loss: Callable[[Weights], torch.Tensor],
    val_loss: Callable[[Weights], torch.Tensor],
    lr: float,
    *,
    mode: str = "exact",
    eps: float = 0.01,
) -> List[torch.Tensor]:
    """∇_θ L_val(ŵ(θ)) for the policy parameters ``theta``, aligned with them.

    ``train_loss(weights)`` / ``val_loss(weights)`` evaluate the task model at
    ``weights`` (a dict of its trainable parameters by name, for ``functional_call``).
    ``train_loss`` must reach ``theta``; its θ-graph is reused across calls, so build the
    relaxed batch once, before calling this."""
    if mode not in MODES:
        raise ValueError(f"unknown hypergradient mode {mode!r}, expected one of {MODES}")
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    names = list(params)
    w = [params[n] for n in names]
    lr = float(lr)

    loss = train_loss(params)
    g = _grads(loss, w, create_graph=(mode == "exact"), retain_graph=True)
    w_hat = {n: (p - lr * gi).detach().requires_grad_(True) for n, p, gi in zip(names, w, g)}
    v = _grads(val_loss(w_hat), list(w_hat.values()))

    if mode == "exact":
        dot = sum((gi * vi).sum() for gi, vi in zip(g, v) if gi.requires_grad)
        if not torch.is_tensor(dot):
            raise RuntimeError("the training loss does not depend on the task model's weights")
        return [-lr * x for x in _grads(dot, theta)]

    norm = torch.sqrt(sum((vi.double() ** 2).sum() for vi in v))
    if not float(norm) > 0:
        return [torch.zeros_like(t) for t in theta]
    step = float(eps / norm)
    plus = {n: (p + step * vi).detach() for n, p, vi in zip(names, w, v)}
    minus = {n: (p - step * vi).detach() for n, p, vi in zip(names, w, v)}
    g_plus = _grads(train_loss(plus), theta, retain_graph=True)
    g_minus = _grads(train_loss(minus), theta)
    return [-lr * (a - b) / (2.0 * step) for a, b in zip(g_plus, g_minus)]
