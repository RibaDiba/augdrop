"""AdaAug's exploration: the loss its policy h is trained on (``AdaAug.explore``).

For each held-out image x, h (on the search model's feature, the model in eval, at
temperature 1) gives p and λ over its outputs. x is copied once per output, each copy the
output's op at its λ (``get_aug_valid_imgs``, a fresh ``random_mirror`` coin per copy, as
each ``apply_augment`` call draws one), passed straight through in λ:

    ST_k = normalize(op_k(x, λ_k)).detach() + λ_k - λ_k.detach()

(``stop_gradient``: d pixel / d λ_k = 1 in the model's normalized input space, every
output, Identity and the ops with no magnitude included, as the code has it). Every copy
goes through the search model's feature extractor f, the features are mixed by p,
``Σ_k p_k f(ST_k)``, and the head g scores the mix: the cross-entropy against x's label
is h's loss (``search.py``: ``gf_model.g(mixed_features)``, then ``criterion``). p gets
its gradient through the mix, λ through the straight-through copies and f.

For segmentation (AdaAug is classification-only):

- f is the segmentor's ``extract_feat`` (backbone, and neck), g its decode head on the
  mixed features, every level mixed alike; the logits are resized to the label as mmseg's
  loss resizes them
- **the label is blended** (flagged in CLAUDE.md): a geometric op (and Flip) moves the
  label, so the mix has none; the soft label is ``Σ_k p_k onehot(label_k)``, each term the
  label op k leaves (``madaug.relax.soft_cross_entropy``, mmseg's average over every
  pixel), photometric ops and Identity sharing the unmoved one. AdaAug's class label
  never moves
- **one image per backward**: an image's copies are mixed only with each other and the
  model is in eval (nothing couples the images), so each image's loss, divided by the
  batch's size, is backpropagated on its own and h's gradient accumulates: the batch's
  gradient exactly, at one image's copies in memory

The ops are the vocab's ``adaaug`` ops, with no gradient (AdaAug's are PIL's); the
host maps each output's λ to them (``hosts.adaaug.to_columns``). A search over the ops a
prune leaves has fewer outputs (``hosts.adaaug.searched_outputs``), so fewer copies.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from hosts.adaaug import AdaAugHost, to_columns
from hosts.madaug import IDENTITY, SIGNED
from madaug.policy import Projection, predict
from madaug.relax import soft_cross_entropy

__all__ = ["copies", "explore_loss", "mix_features", "pooled", "straight_through"]


def straight_through(normalized: Tensor, level: Tensor) -> Tensor:
    """AdaAug's ``stop_gradient``: ``normalized`` as it is, with d / d ``level`` = 1 at
    every pixel. ``level`` is (B,)."""
    return normalized.detach() + (level - level.detach()).view(-1, 1, 1, 1)


def pooled(segmentor, inputs: Tensor) -> Tensor:
    """(B, C): the segmentor's deepest ``extract_feat`` output, average-pooled
    (``tasks.mmseg.backend.pooled_features``, without the framework): AdaAug's
    ``gf_model.f``, a classifier's penultimate feature, for a segmentor."""
    feats = segmentor.extract_feat(inputs)
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    return feats.mean(dim=(-2, -1)) if feats.dim() == 4 else feats.flatten(1)


def copies(
    image01: Tensor,
    label: Optional[Tensor],
    levels: np.ndarray,
    host: AdaAugHost,
    ops: Dict,
    rng: np.random.Generator,
) -> List[Tuple[Tensor, Optional[Tensor]]]:
    """One image's copies, one per output of ``host.outputs``: ``(image01, label)`` of
    output k's op at ``levels[k]`` (no gradient), the label moved by a geometric op and
    otherwise the same object as ``label``. ``image01`` is (1, 3, H, W) in [0, 1]."""
    live = host._live()
    out = []
    for k, op in enumerate(host.outputs):
        if op == IDENTITY:
            out.append((image01, label))
            continue
        positive = host.direction(op, 1, rng, live) if op in SIGNED else None
        names, magnitudes = to_columns(op, levels[k : k + 1], positive, live)
        spec = ops[names[0]]
        magnitude = torch.as_tensor(magnitudes, dtype=image01.dtype, device=image01.device) if spec.has_magnitude else None
        with torch.no_grad():
            out.append(spec.apply(image01, magnitude, label))
    return out


def mix_features(feats, weights: Tensor):
    """``Σ_k weights[k] feats[k]`` over the copies (dim 0), at every level of ``feats``
    (a tensor, or a list / tuple of them, as ``extract_feat`` returns): one image's mix,
    batch size 1, the same structure."""
    if isinstance(feats, (list, tuple)):
        return type(feats)(mix_features(f, weights) for f in feats)
    return (weights.view(-1, *([1] * (feats.dim() - 1))) * feats).sum(dim=0, keepdim=True)


def explore_loss(
    segmentor,
    policy: Projection,
    feature: Tensor,
    image01: Tensor,
    label: Tensor,
    host: AdaAugHost,
    ops: Dict,
    normalize: Callable[[Tensor], Tensor],
    rng: np.random.Generator,
    ignore_index: int = 255,
) -> Tuple[Tensor, Dict[str, np.ndarray]]:
    """One held-out image's exploration loss (module docstring), with the graph to h's
    parameters: ``feature`` (1, F) is its clean feature (detached), ``image01`` (1, 3, H,
    W) and ``label`` (1, 1, H, W) the image. The segmentor should be in eval with no
    gradient to its parameters. Returns the loss and the policy's (p, levels) on it."""
    levels, p = predict(policy, feature.detach(), 1.0)
    level_values = levels.detach().double().cpu().numpy()[0]
    made = copies(image01, label, level_values, host, ops, rng)
    inputs = torch.cat(
        [straight_through(normalize(out), levels[:, k]) for k, (out, _) in enumerate(made)], dim=0
    )
    labels: List[Optional[Tensor]] = [label]
    weights: List[Tensor] = [p.new_zeros(1)]
    for k, (_, out_label) in enumerate(made):
        if out_label is label:
            weights[0] = weights[0] + p[:, k]
        else:
            labels.append(out_label)
            weights.append(p[:, k])
    mixed = mix_features(segmentor.extract_feat(inputs), p[0])
    logits = segmentor.decode_head.forward(mixed)
    logits = F.interpolate(
        logits, size=tuple(label.shape[-2:]), mode="bilinear", align_corners=bool(getattr(segmentor, "align_corners", False))
    )
    loss = soft_cross_entropy(logits, labels, torch.stack(weights, dim=1), ignore_index)
    return loss, {"p": p.detach().double().cpu().numpy()[0], "levels": level_values}
