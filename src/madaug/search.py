"""One MADAug iteration's two policy calls (Algorithm 1), with the framework passed in.

    policy_step   lines 8-10: relax the training batch, take the policy's gradient
                  through a virtual step of the task model, return it
    exploit       line 11: the policy on the training batch, the host's draw from it

The framework supplies callables: ``features`` (model inputs -> (B, C) pooled
features), ``logits`` (weights, model inputs -> (B, K, H, W) at the label's size) and
``val_loss`` (weights -> scalar on a validation batch). The caller averages the gradient
across ranks and steps the policy's optimizer.
"""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .hypergrad import policy_gradient
from .policy import Projection, predict
from .relax import relaxed_batch, soft_cross_entropy

__all__ = ["exploit", "policy_step"]


def policy_step(
    *,
    model: torch.nn.Module,
    policy: Projection,
    images01: torch.Tensor,
    label: torch.Tensor,
    ops: Dict,
    normalize: Callable[[torch.Tensor], torch.Tensor],
    features: Callable[[torch.Tensor], torch.Tensor],
    logits: Callable[[Dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    val_loss: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    lr: float,
    rng: np.random.Generator,
    mode: str = "exact",
    eps: float = 0.01,
    ignore_index: int = 255,
) -> Tuple[List[torch.Tensor], dict]:
    """The policy's gradient for one training batch (clean ``images01``, ``label``) and
    the stats worth logging. The model should be in eval (``eval_mode``), as MADAug's
    update runs it. The policy is relaxed at T = 1, as ``explore`` does."""
    with torch.no_grad():
        feats = features(normalize(images01))
    levels, p = predict(policy, feats, temperature=1.0)
    mixed, labels, weights = relaxed_batch(images01, label, levels, p, ops, normalize, rng)

    def train_loss(weights_):
        return soft_cross_entropy(logits(weights_, mixed), labels, weights, ignore_index)

    theta = [t for t in policy.parameters() if t.requires_grad]
    grads = policy_gradient(model, theta, train_loss, val_loss, lr, mode=mode, eps=eps)
    norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads)))
    return grads, {"policy_grad_norm": norm, "p_identity": float(p[:, -1].detach().mean())}


def exploit(
    policy: Projection,
    features: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    host,
    rng: np.random.Generator,
) -> Sequence[List[Tuple[str, float]]]:
    """The training draw for one batch: the policy at the host's temperature on
    ``inputs`` (the model's input space), its mean p folded into the host's prior, and
    the host's slots (``MADAugHost.draw``: curriculum, q, k ops, δ, signs)."""
    with torch.no_grad():
        levels, p = predict(policy, features(inputs), host.temperature)
    p = p.double().cpu().numpy()
    host.observe(p.mean(axis=0))
    return host.draw(p, levels.double().cpu().numpy(), rng)
