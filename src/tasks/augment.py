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

Both are Sensaug's DIFF32 ops in their unsigned [0, 1] units. Five more hosts train
on their method's own policy, over its ops made differentiable (``BIN_HOSTS``). For the
first three each draw is an op from ``host.sampling_probs()`` (q applied) with
``host.none_prob`` for "none", at the magnitude of the bin ``host.draw_bins`` gives it,
read through ``vocab.magnitude_bins``, in the op's own units:

- ``trivialaugment_diff``: TrivialAugmentWide's. One draw per image, at one uniform bin
- ``randaugment_diff``: RandAugment's at its published (N, M). ``num_ops`` draws per
  image, applied in turn, each at level M; geometric ops move the label
- ``augmix_diff``: AugMix's as Sensaug ran it (``MIX_HOSTS``). ``mixture_width`` chains
  of 1 to 3 draws, each at a bin uniform on the first ``severity``, mixed with the clean
  image by ``host.draw_mix``'s weights. The label is NOT moved, by any chain: Sensaug's
  AugMix augmented the image and left the mask as it was (hosts/augmix.py)
- ``smartsampling_diff``: SmartSamplingAugment's (``SLOT_HOSTS``). Its ops and their
  level are drawn jointly per image, so the host draws them itself
  (``host.draw_slots``: 2 distinct ops at one level, with probability P annealed by the
  training step), and each slot's bins are read through ``vocab.magnitude_bins`` as
  above; geometric ops move the label (hosts/smartsampling.py)
- ``autoaugment_diff``: AutoAugment's ImageNet policy (``SLOT_HOSTS``). Per image a
  sub-policy, then each of its two steps with its probability, at its bin, a signed op's
  direction drawn by the step (``host.draw_slots``); geometric ops move the label
  (hosts/autoaugment.py)

One more trains on its method's own ops in their own units, drawn by a policy network
(``POLICY_HOSTS``):

- ``madaug``: MADAug's (hosts/madaug.py). Its draws depend on the model (the policy reads
  each image's feature), so the training loop computes them (``madaug.exploit``) and the
  preprocessor applies them as given, ``k_ops`` slots of ``(op, magnitude)`` per image
  (``apply_slots``); geometric ops move the label

And one draws whole stages of ``(op, magnitude)`` from a searched policy
(``STAGE_HOSTS``):

- ``faster_autoaugment``: Faster AutoAugment's (hosts/faster_autoaugment). Per image, a
  sub-policy, then in each of its stages an op and its searched magnitude, in the op's
  units, or nothing (``host.draw_stages``), applied stage by stage; geometric ops move the
  label. SamplePairing blends in an image of the batch (``couples_images``), so
  ``apply_draws`` runs it over the whole batch (below)

``augment_batch`` applies whichever of the others a host has. ``check_trainable`` refuses
every other host: the torchvision methods (randaugment, trivialaugment, autoaugment),
whose ops are 8-bit torchvision calls, train through their ``*_diff`` hosts.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "BIN_HOSTS",
    "LEVEL_JITTER",
    "MIX_HOSTS",
    "NO_OP",
    "POLICY_HOSTS",
    "SLOT_HOSTS",
    "STAGE_HOSTS",
    "TRAINABLE_HOSTS",
    "apply_draws",
    "apply_slots",
    "augment_batch",
    "check_trainable",
    "sample_draws",
    "slots",
    "stages",
]

#: The "leave this image alone" draw: the op half of hosts.sensaug.NONE_KEY.
NO_OP = "none"

#: The hosts that sample Sensaug's ops at a level in [0, 1].
TRAINABLE_HOSTS = ("sensaug", "uniform")

#: The hosts that sample at their method's own magnitude bins (``draw_bins``).
BIN_HOSTS = ("trivialaugment_diff", "randaugment_diff", "augmix_diff", "smartsampling_diff", "autoaugment_diff")

#: The BIN_HOSTS that mix chains of draws with the clean image (``draw_mix``).
MIX_HOSTS = ("augmix_diff",)

#: The BIN_HOSTS that draw a whole batch's ops and bins themselves (``draw_slots``).
SLOT_HOSTS = ("smartsampling_diff", "autoaugment_diff")

#: The hosts whose draws come from a policy network on the model's features, computed by
#: the training loop and applied with ``apply_slots``.
POLICY_HOSTS = ("madaug",)

#: The hosts that draw a batch's stages of ``(op, magnitude)`` from a searched policy
#: (``draw_stages``), the magnitude in the op's units.
STAGE_HOSTS = ("faster_autoaugment",)

#: Sensaug's level jitter on a pdf draw, so the pdf's discrete levels are not the
#: only magnitudes ever trained on.
LEVEL_JITTER = 0.1


def sample_draws(host, batch_size: int, rng=np.random) -> List[Tuple[str, float]]:
    """One ``(op, level)`` per image, ``NO_OP`` for none.

    Per IMAGE, not per batch, as the transforms Sensaug replaced sampled. ``rng`` is
    numpy's global RNG by default, as in Sensaug, so the framework's seeding governs it.
    A ``SLOT_HOSTS`` host has no per-draw policy to sample from: use ``augment_batch``.
    """
    if host.name in SLOT_HOSTS or host.name in STAGE_HOSTS:
        raise ValueError(
            f"host {host.name!r} draws its ops and their magnitudes jointly (draw_slots / "
            f"draw_stages); augment with augment_batch"
        )
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
    names = list(host.vocabulary()) + [NO_OP]
    probs = np.append((1.0 - host.none_prob) * host.sampling_probs(), host.none_prob)
    picks = [names[rng.choice(len(names), p=probs)] for _ in range(batch_size)]
    from vocab import get_op

    bins, flips = host.draw_bins(batch_size, _generator(rng))
    out = []
    for i, name in enumerate(picks):
        if name == NO_OP or not get_op(name).has_magnitude:
            out.append((name, 0.0))
        else:
            out.append((name, _bin_magnitude(host, name, bins[name][i], flips[name][i])))
    return out


def _generator(rng) -> np.random.Generator:
    """A ``np.random.Generator`` for a host's own draws, seeded from ``rng`` (numpy's
    global RNG or a Generator), so the framework's seeding governs it."""
    seed = rng.integers(0, 2**31 - 1) if hasattr(rng, "integers") else rng.randint(0, 2**31 - 1)
    return np.random.default_rng(int(seed))


def _bin_magnitude(host, name: str, index, flip) -> float:
    """Op ``name``'s magnitude at bin ``index`` of the host's table, in the op's units,
    negated (mirrored about the identity) when the method flips a signed bin. 0 for an
    op with no magnitude."""
    from vocab import get_op, magnitude_bins

    op = get_op(name)
    if not op.has_magnitude:
        return 0.0
    values, signed = magnitude_bins(name, host.num_magnitude_bins)
    magnitude = float(values[int(index)])
    if signed and flip:
        lo, hi = op.mag_range
        magnitude = min(max(2.0 * op.identity_at - magnitude, lo), hi)
    return magnitude


def apply_draws(rgb01, labels: Optional[torch.Tensor], draws: Sequence[Tuple[str, float]], ops: Dict):
    """Apply each drawn op to the sub-batch that drew it; ``labels`` move with the
    geometric ones (the vocab op warps them). Grouped rather than looped per image,
    so the ops still run batched. Sensaug's ``_apply_ops``.

    An op that ``couples_images`` (SamplePairing) runs over the whole batch as it came
    in, every other image at the op's identity, so an image's partner can be any image
    of the batch, not only one that drew the same op; only the images that drew it keep
    the result."""
    groups: Dict[str, List[Tuple[int, float]]] = {}
    for i, (name, level) in enumerate(draws):
        if name != NO_OP:
            groups.setdefault(name, []).append((i, level))

    source = rgb01
    for name, members in groups.items():
        idx = torch.as_tensor([i for i, _ in members], device=rgb01.device)
        levels = torch.as_tensor([m for _, m in members], dtype=rgb01.dtype, device=rgb01.device)
        if getattr(ops[name], "couples_images", False):
            full = torch.full((int(source.shape[0]),), float(ops[name].identity_at), dtype=rgb01.dtype, device=rgb01.device)
            out, _ = ops[name].apply(source, full.index_copy(0, idx, levels), None)
            rgb01 = rgb01.index_copy(0, idx, out[idx])
            continue
        sub_labels = None if labels is None else labels[idx]
        out, out_labels = ops[name].apply(rgb01[idx], levels, sub_labels)
        rgb01 = rgb01.index_copy(0, idx, out)
        if labels is not None:
            labels = labels.index_copy(0, idx, out_labels)
    return rgb01, labels


def augment_batch(host, rgb01, labels: Optional[torch.Tensor], ops: Dict, rng=np.random):
    """One training batch augmented by ``host``'s policy: ``num_ops`` rounds of
    ``sample_draws`` / ``apply_draws`` (1 for every host but RandAugment's), for a
    ``MIX_HOSTS`` host its chains mixed with the clean image (``augmix``), or for a
    ``SLOT_HOSTS`` host its own slots (``slots``). A ``POLICY_HOSTS`` host's draws need
    the model: its loop computes them, and they are applied with ``apply_slots``."""
    if host.name in POLICY_HOSTS:
        raise ValueError(
            f"host {host.name!r} draws from a policy on the model's features; its training "
            f"loop computes the draws, applied with apply_slots"
        )
    if host.name in MIX_HOSTS:
        return augmix(host, rgb01, ops, rng), labels
    if host.name in SLOT_HOSTS:
        return slots(host, rgb01, labels, ops, rng)
    if host.name in STAGE_HOSTS:
        return stages(host, rgb01, labels, ops, rng)
    for _ in range(int(getattr(host, "num_ops", 1))):
        draws = sample_draws(host, int(rgb01.shape[0]), rng)
        rgb01, labels = apply_draws(rgb01, labels, draws, ops)
    return rgb01, labels


def augmix(host, rgb01, ops: Dict, rng=np.random):
    """AugMix as torchvision's ``AugMix.forward`` computes it, per image: ``m0 * x +
    sum_i w_i * chain_i(x)``, with the depths and weights from ``host.draw_mix`` and every
    step of every chain one ``sample_draws`` draw (an image whose chain is shorter gets
    ``NO_OP``). No label goes in, so the geometric ops move the image alone; the caller
    keeps the mask as it was, as Sensaug's AugMix did. The mix stays in float, where
    torchvision casts it back to uint8."""
    batch_size = int(rgb01.shape[0])
    depths, clean, chains = host.draw_mix(batch_size, _generator(rng))

    def per_image(weights):
        return torch.as_tensor(weights, dtype=rgb01.dtype, device=rgb01.device).view(-1, 1, 1, 1)

    mix = per_image(clean) * rgb01
    for i in range(depths.shape[1]):
        chain = rgb01
        for step in range(int(depths[:, i].max())):
            draws = sample_draws(host, batch_size, rng)
            draws = [draw if step < depths[b, i] else (NO_OP, 0.0) for b, draw in enumerate(draws)]
            chain, _ = apply_draws(chain, None, draws, ops)
        mix = mix + per_image(chains[:, i]) * chain
    return mix


def slots(host, rgb01, labels: Optional[torch.Tensor], ops: Dict, rng=np.random):
    """A ``SLOT_HOSTS`` host's draw applied to one batch: ``host.draw_slots`` gives each
    image's ops in order with their bins, each slot is read through
    ``vocab.magnitude_bins`` and applied in turn (``apply_draws``), and the label moves
    with the geometric ops. An image the host leaves alone is ``NO_OP`` in every slot."""
    for names, bins in host.draw_slots(int(rgb01.shape[0]), _generator(rng)):
        draws = [
            (str(name), 0.0 if name == NO_OP else _bin_magnitude(host, str(name), index, False))
            for name, index in zip(names, bins)
        ]
        rgb01, labels = apply_draws(rgb01, labels, draws, ops)
    return rgb01, labels


def stages(host, rgb01, labels: Optional[torch.Tensor], ops: Dict, rng=np.random):
    """A ``STAGE_HOSTS`` host's draw applied to one batch: ``host.draw_stages`` gives
    each stage's op per image with its magnitude in the op's units, applied in turn
    (``apply_draws``), the label moving with the geometric ops. An image a stage leaves
    alone is ``NO_OP`` there."""
    for names, magnitudes in host.draw_stages(int(rgb01.shape[0]), _generator(rng)):
        draws = [(str(name), float(m)) for name, m in zip(names, magnitudes)]
        rgb01, labels = apply_draws(rgb01, labels, draws, ops)
    return rgb01, labels


def apply_slots(rgb01, labels: Optional[torch.Tensor], slot_draws: Sequence[Sequence[Tuple[str, float]]], ops: Dict):
    """Slots of per-image ``(op, magnitude)`` draws, the magnitude in the op's units,
    applied in turn (``apply_draws``), the label moving with the geometric ops. A
    ``POLICY_HOSTS`` host's draw (``MADAugHost.draw``)."""
    for draws in slot_draws:
        if len(draws) != int(rgb01.shape[0]):
            raise ValueError(f"a slot holds {len(draws)} draws for a batch of {int(rgb01.shape[0])}")
        rgb01, labels = apply_draws(rgb01, labels, draws, ops)
    return rgb01, labels


def check_trainable(host, ops) -> None:
    """Refuse a host whose draws can't be applied. Sensaug's sampling is over its
    differentiable ops in unsigned [0, 1] units, and a ``BIN_HOSTS`` host's over ops
    with its method's bins; anything else would train on magnitudes nobody chose. A
    ``POLICY_HOSTS`` host's draws are in its own ops' units (its vocabulary, which
    ``dropaug.config.check`` holds the vocab to)."""
    if host.name in POLICY_HOSTS:
        foreign = [name for name in ops if name not in host.vocabulary()]
        if not foreign:
            return
    elif host.name in STAGE_HOSTS:
        if not getattr(host, "searched", True):
            raise NotImplementedError(
                f"host {host.name!r} has no searched policy (host.policy_file is null), only "
                f"the search's starting point; run scripts/faa_search.py and set "
                f"host.policy_file to its final.npz"
            )
        foreign = [name for name in ops if name not in host.vocabulary()]
        if not foreign:
            return
    elif host.name in BIN_HOSTS:
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
        + ". The torchvision methods train through their differentiable hosts "
        "(trivialaugment_diff, randaugment_diff, autoaugment_diff, augmix_diff); SmartSamplingAugment through "
        "smartsampling_diff; MADAug through madaug; Faster AutoAugment through faster_autoaugment. augment=False probes them on a clean-trained "
        "model."
    )
