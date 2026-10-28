"""MADAug's relaxed batch: what the policy update differentiates (``MDAAug.explore``).

Sampling k ops has no gradient in p or λ, so the update trains on a stand-in: every op
applied to the image at the policy's level, the outputs blended by p,

    x~ = sum_j p_j * ST_j,   ST_j = op_j(x, λ_j).detach() + λ_j - λ_j.detach()

with ST_j MADAug's ``stop_gradient``: the straight-through estimator of §3.3, Eq. 7,
which takes d pixel / d λ_j = 1 in the model's normalized input space. Identity's
output is the image. The ops run at the policy's levels with no δ perturbation and with
a random sign on the geometric ones, as ``explore`` calls ``apply_augment``.

Two departures from the paper, both from MADAug's own code or forced by segmentation:

- **One op per term.** The paper's Eq. 6 blends pairs (k = 2, 17 x 16 composed images per
  input); the released code blends the 17 single ops, and so does this. Pairs at
  segmentation resolution would not fit.
- **The label is blended too** (segmentation only). A geometric op moves the label, so
  the blend has none. The soft label is ``sum_j p_j * onehot(label_j)``, each term the
  label op j leaves, and ``soft_cross_entropy`` scores it. p gets gradient through the
  image and through the label. Every photometric op and Identity leave the label as it
  is, so they share one term.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hosts.madaug import IDENTITY, OPS_NAMES, SIGNED, VOCAB_NAMES, level_to_magnitude

__all__ = ["RelaxedBatch", "relaxed_batch", "soft_cross_entropy"]

#: ``(mixed inputs (B, C, H, W), labels [(B, 1, H, W) or None], weights (B, len(labels)))``.
RelaxedBatch = Tuple[torch.Tensor, List[Optional[torch.Tensor]], torch.Tensor]


def relaxed_batch(
    images01: torch.Tensor,
    label: Optional[torch.Tensor],
    levels: torch.Tensor,
    p: torch.Tensor,
    ops: Dict,
    normalize: Callable[[torch.Tensor], torch.Tensor],
    rng: np.random.Generator,
) -> RelaxedBatch:
    """The relaxed batch for clean ``images01`` ([0, 1], (B, 3, H, W)) and ``label``
    ((B, 1, H, W), or None), at the policy's ``levels`` and ``p`` ((B, 17), MADAug's
    order, with grad). ``ops`` maps vocab names to OpSpecs; ``normalize`` takes [0, 1]
    images into the model's input space.

    Returns the blended inputs, the distinct labels (index 0 the unmoved one) and each
    image's weight on each, which ``soft_cross_entropy`` takes."""
    images01 = images01.detach()
    level_values = levels.detach().double().cpu().numpy()
    mixed = torch.zeros_like(images01)
    labels: List[Optional[torch.Tensor]] = [label]
    weights: List[torch.Tensor] = [torch.zeros_like(p[:, 0])]
    for j, op in enumerate(OPS_NAMES):
        if op == IDENTITY:
            out, out_label = images01, label
        else:
            negate = rng.random(images01.shape[0]) < 0.5 if op in SIGNED else False
            magnitude = level_to_magnitude(op, level_values[:, j], negate)
            spec = ops[VOCAB_NAMES[op]]
            with torch.no_grad():
                out, out_label = spec.apply(
                    images01,
                    torch.as_tensor(magnitude, dtype=images01.dtype) if spec.has_magnitude else None,
                    label,
                )
        straight = normalize(out).detach() + (levels[:, j] - levels[:, j].detach()).view(-1, 1, 1, 1)
        mixed = mixed + p[:, j].view(-1, 1, 1, 1) * straight
        if out_label is label:
            weights[0] = weights[0] + p[:, j]
        else:
            labels.append(out_label)
            weights.append(p[:, j])
    return mixed, labels, torch.stack(weights, dim=1)


def soft_cross_entropy(
    logits: torch.Tensor, labels: List[Optional[torch.Tensor]], weights: torch.Tensor, ignore_index: int = 255
) -> torch.Tensor:
    """Cross-entropy against the soft label ``sum_j weights[:, j] * onehot(labels[j])``,
    summed over pixels and divided by their count (mmseg's ``CrossEntropyLoss`` default,
    which averages over every pixel, ignored ones scoring 0). ``logits`` is (B, K, H, W)
    at the labels' size; ``ignore_index`` pixels of a label score 0 in its term."""
    logp = F.log_softmax(logits, dim=1)
    total = logits.new_zeros(())
    count = logits.shape[0] * logits.shape[-2] * logits.shape[-1]
    for j, label in enumerate(labels):
        if label is None:
            raise ValueError("soft_cross_entropy needs labels; the batch carries none")
        label = label.reshape(label.shape[0], *label.shape[-2:]).long()
        valid = label != ignore_index
        picked = logp.gather(1, torch.where(valid, label, 0).unsqueeze(1)).squeeze(1)
        total = total - (weights[:, j].view(-1, 1, 1) * picked * valid).sum()
    return total / count
