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

Both are Sensaug's DIFF32 ops in their unsigned [0, 1] units. One more host trains
on its method's own policy:

- ``trivialaugment_diff``: TrivialAugmentWide's. One draw per image from
  ``host.sampling_probs()`` (q applied) with ``host.none_prob`` for "none", at the
  magnitude of one uniform bin (``host.draw_bins``, read through
  ``vocab.magnitude_bins``), in the op's own units

``check_trainable`` refuses every other host: the torchvision methods (randaugment,
trivialaugment, autoaugment) until they have training-time ops of their own.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "BIN_HOSTS",
    "LEVEL_JITTER",
    "NO_OP",
    "TRAINABLE_HOSTS",
    "apply_draws",
    "check_trainable",
    "sample_draws",
]

#: The "leave this image alone" draw: the op half of hosts.sensaug.NONE_KEY.
NO_OP = "none"

#: The hosts that sample Sensaug's ops at a level in [0, 1].
TRAINABLE_HOSTS = ("sensaug", "uniform")

#: The hosts that sample at their method's own magnitude bins (``draw_bins``).
BIN_HOSTS = ("trivialaugment_diff",)

#: Sensaug's level jitter on a pdf draw, so the pdf's discrete levels are not the
#: only magnitudes ever trained on.
LEVEL_JITTER = 0.1


def sample_draws(host, batch_size: int, rng=np.random) -> List[Tuple[str, float]]:
    """One ``(op, level)`` per image, ``NO_OP`` for none.

    Per IMAGE, not per batch, as the transforms Sensaug replaced sampled. ``rng`` is
    numpy's global RNG by default, as in Sensaug, so the framework's seeding governs it.
    """
    if host.name in BIN_HOSTS:
        return _sample_bin_draws(host, batch_size, rng)

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


def _sample_bin_draws(host, batch_size: int, rng) -> List[Tuple[str, float]]:
    """One ``(op, magnitude)`` per image, the magnitude in the op's units: the op from
    ``sampling_probs()`` / ``none_prob``, the magnitude its method's bin for the image
    (``host.draw_bins``), negated as the method does for a signed bin. ``draw_bins``
    takes a ``np.random.Generator``; it is seeded from ``rng``, so the framework's
    seeding still governs the draw."""
    from vocab import get_op, magnitude_bins

    names = list(host.vocabulary()) + [NO_OP]
    probs = np.append((1.0 - host.none_prob) * host.sampling_probs(), host.none_prob)
    picks = [names[rng.choice(len(names), p=probs)] for _ in range(batch_size)]
    seed = rng.integers(0, 2**31 - 1) if hasattr(rng, "integers") else rng.randint(0, 2**31 - 1)
    bins, flips = host.draw_bins(batch_size, np.random.default_rng(int(seed)))

    out = []
    for i, name in enumerate(picks):
        op = None if name == NO_OP else get_op(name)
        if op is None or not op.has_magnitude:
            out.append((name, 0.0))
            continue
        values, signed = magnitude_bins(name, host.num_magnitude_bins)
        magnitude = float(values[int(bins[name][i])])
        if signed and flips[name][i]:
            lo, hi = op.mag_range
            magnitude = min(max(2.0 * op.identity_at - magnitude, lo), hi)
        out.append((name, magnitude))
    return out


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
    differentiable ops in unsigned [0, 1] units, and a ``BIN_HOSTS`` host's over ops
    with its method's bins; anything else would train on magnitudes nobody chose."""
    if host.name in BIN_HOSTS:
        from vocab import magnitude_bins

        foreign = []
        for name, op in ops.items():
            if not op.has_magnitude:
                continue
            try:
                magnitude_bins(name, host.num_magnitude_bins)
            except KeyError:
                foreign.append(name)
        if not foreign:
            return
    else:
        foreign = [
            name
            for name, op in ops.items()
            if op.grad_mode != "autograd" or op.mag_range != (0.0, 1.0) or op.identity_at != 0.0
        ]
        if host.name in TRAINABLE_HOSTS and not foreign:
            return
    raise NotImplementedError(
        f"training augments with Sensaug's differentiable ops on the {TRAINABLE_HOSTS} hosts, "
        f"or with its method's binned ops on the {BIN_HOSTS} hosts, only; got host {host.name!r}"
        + (f" with ops {foreign[:3]}..." if foreign else "")
        + ". The torchvision methods are not wired for training yet (trivialaugment_diff "
        "trains TrivialAugment). augment=False probes them on a clean-trained model."
    )
