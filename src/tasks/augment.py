"""Training-time augmentation from a host, for any backend: draw ``(op, level)`` per
image from the host's policy and apply the draws to a batch on its device.

Ported from Sensaug's ``GpuAugSegDataPreProcessor`` (sensaug/dataset/gpu_augment.py)
and moved out of ``tasks/mmseg/transforms.py``, which imports mmseg at module load, so
that the cls loop draws exactly as the mmseg preprocessor does. The sampling is
Sensaug's, read off the host:

- a ``sensaug`` host with an SA pdf: ``(op, level)`` drawn from ``host.pdf()`` (q
  already applied), level jittered by N(0, 0.1) and clipped to [0, 1], as
  ``RandomTrainTransformNew`` did
- otherwise (a ``uniform`` host, or ``sensaug`` before its first pdf): op drawn from
  ``host.sampling_probs()`` with ``host.none_prob`` for "none", level uniform on
  [0, 1], as ``RandomAlphaTrainTransform`` did

Both are Sensaug's DIFF32 ops in their unsigned [0, 1] units, and those are the only
ops training augments with (``check_trainable``). The torchvision methods are refused
until they have training-time ops of their own.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["LEVEL_JITTER", "NO_OP", "TRAINABLE_HOSTS", "apply_draws", "check_trainable", "sample_draws"]

#: The "leave this image alone" draw: the op half of hosts.sensaug.NONE_KEY.
NO_OP = "none"

#: The hosts whose draws this reproduces. Both sample Sensaug's ops.
TRAINABLE_HOSTS = ("sensaug", "uniform")

#: Sensaug's level jitter on a pdf draw, so the pdf's discrete levels are not the
#: only magnitudes ever trained on.
LEVEL_JITTER = 0.1


def sample_draws(host, batch_size: int, rng=np.random) -> List[Tuple[str, float]]:
    """One ``(op, level)`` per image, ``NO_OP`` for none.

    Per IMAGE, not per batch, as the transforms Sensaug replaced sampled. ``rng`` is
    numpy's global RNG by default, as in Sensaug, so the framework's seeding governs it.
    """
    pdf = host.pdf() if hasattr(host, "pdf") else None
    if pdf is not None:
        keys = list(pdf)
        probs = np.asarray([pdf[k] for k in keys], dtype=np.float64)
        probs = probs / probs.sum()
        out = []
        for _ in range(batch_size):
            op, level = keys[rng.choice(len(keys), p=probs)]
            if op != NO_OP:
                level = float(np.clip(level + rng.normal(0, scale=LEVEL_JITTER), 0.0, 1.0))
            out.append((str(op), float(level)))
        return out

    names = list(host.vocabulary()) + [NO_OP]
    probs = np.append((1.0 - host.none_prob) * host.sampling_probs(), host.none_prob)
    return [
        (names[rng.choice(len(names), p=probs)], float(rng.uniform(0.0, 1.0)))
        for _ in range(batch_size)
    ]


def apply_draws(rgb01, labels: Optional[torch.Tensor], draws: Sequence[Tuple[str, float]], ops: Dict):
    """Apply each drawn op to the sub-batch that drew it; ``labels`` move with the
    geometric ones (the vocab op warps them). Grouped rather than looped per image,
    so the ops still run batched. Sensaug's ``_apply_ops``."""
    groups: Dict[str, List[Tuple[int, float]]] = {}
    for i, (name, level) in enumerate(draws):
        if name != NO_OP:
            groups.setdefault(name, []).append((i, level))

    for name, members in groups.items():
        idx = torch.as_tensor([i for i, _ in members], device=rgb01.device)
        levels = torch.as_tensor([m for _, m in members], dtype=rgb01.dtype, device=rgb01.device)
        sub_labels = None if labels is None else labels[idx]
        out, out_labels = ops[name].apply(rgb01[idx], levels, sub_labels)
        rgb01 = rgb01.index_copy(0, idx, out)
        if labels is not None:
            labels = labels.index_copy(0, idx, out_labels)
    return rgb01, labels


def check_trainable(host, ops) -> None:
    """Refuse a host whose draws can't be applied. Sensaug's sampling is over its
    differentiable ops in unsigned [0, 1] units; anything else would train on
    magnitudes nobody chose."""
    foreign = [
        name
        for name, op in ops.items()
        if op.grad_mode != "autograd" or op.mag_range != (0.0, 1.0) or op.identity_at != 0.0
    ]
    if host.name not in TRAINABLE_HOSTS or foreign:
        raise NotImplementedError(
            f"training augments with Sensaug's differentiable ops on the {TRAINABLE_HOSTS} "
            f"hosts only; got host {host.name!r}"
            + (f" with ops {foreign[:3]}..." if foreign else "")
            + ". The torchvision methods are not wired for training yet. augment=False "
            "probes them on a clean-trained model."
        )
