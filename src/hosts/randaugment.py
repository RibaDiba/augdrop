"""RandAugment (Cubuk et al. 2020) as a host, with torchvision's op space.

Each image gets ``num_ops`` independent draws, uniform over 14 entries, all at
bin ``magnitude`` of ``num_magnitude_bins``. ``Identity`` is one of the 14. It is
the no-op draw, so it stays out of the vocabulary and keeps its 1/14 when q
reweights the other 13.

Two hosts. ``randaugment`` is torchvision's RandAugment with its defaults (N = 2, bin 9
of 31) over vocab's ``randaug_default`` (torchvision's own ops), probe-only.
``randaugment_diff`` is the paper's published (N, M) for the run's dataset
(``PUBLISHED`` / ``STAND_INS``, kept here rather than in a config because they are the
paper's numbers, not knobs) over vocab's ``randaug_diff`` (the ops made
differentiable), and trains.

The vocabulary is a vocab family's ops by name and in its order, so the probe can look
each op up with ``vocab.get_op``. Hardcoded rather than imported so the hosts stay free
of torch; tests/hosts pins the two.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .base import AugHost, draw_flips

__all__ = [
    "IDENTITY",
    "OP_ORDER",
    "PUBLISHED",
    "RandAugmentDiffHost",
    "RandAugmentHost",
    "SIGNED_OPS",
    "STAND_INS",
    "family_ops",
    "published",
    "split_ops",
    "split_weights",
]

#: The RandAugment paper's Fig. 2 op order, which vocab's ``randaug_default`` and
#: ``trivialaug_wide`` families follow. torchvision implements the same 14 (in
#: another order). Signed ops (ShearX, Rotate, Brightness, ...) flip their sign
#: with p=0.5 inside the op, so one name covers both directions.
OP_ORDER = (
    "Identity",
    "AutoContrast",
    "Equalize",
    "Rotate",
    "Solarize",
    "Color",
    "Posterize",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
)
IDENTITY = "Identity"


def family_ops(family: str) -> tuple:
    """A vocab family's op names for OP_ORDER, minus Identity."""
    return tuple(f"{family}.{op}" for op in OP_ORDER if op != IDENTITY)


#: The ops RandAugment, TrivialAugmentWide and AugMix negate with p = 0.5, which the
#: ``*_diff`` families split into ``_pos``/``_neg``.
SIGNED_OPS = frozenset(
    {"Rotate", "Color", "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateX", "TranslateY"}
)


def split_ops(family: str) -> tuple:
    """A split family's op names for OP_ORDER, minus Identity: each signed op as its
    ``_pos`` then ``_neg``."""
    names = []
    for op in OP_ORDER:
        if op == IDENTITY:
            continue
        names.extend([f"{family}.{op}_pos", f"{family}.{op}_neg"] if op in SIGNED_OPS else [f"{family}.{op}"])
    return tuple(names)


def split_weights(names) -> np.ndarray:
    """A split vocabulary's base distribution: uniform over the 13 ops, a signed op's
    share split evenly between its two directions (the method negates it with p = 0.5)."""
    weights = np.array([0.5 if name.endswith(("_pos", "_neg")) else 1.0 for name in names])
    return weights / weights.sum()


class RandAugmentHost(AugHost):
    name = "randaugment"
    family = "randaug_default"

    def __init__(
        self,
        *,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        if num_ops < 1:
            raise ValueError(f"num_ops must be >= 1, got {num_ops}")
        if num_magnitude_bins < 2:
            raise ValueError(f"num_magnitude_bins must be >= 2, got {num_magnitude_bins}")
        if not 0 <= magnitude < num_magnitude_bins:
            raise ValueError(
                f"magnitude must be a bin index in [0, {num_magnitude_bins}), got {magnitude}"
            )
        self.num_ops = int(num_ops)
        self.magnitude = int(magnitude)
        self.num_magnitude_bins = int(num_magnitude_bins)

        names = family_ops(self.family)
        n = len(names)
        super().__init__(
            names,
            np.full(n, 1.0 / n),
            none_prob=1.0 / (n + 1),
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    def draw_bins(self, batch_size, rng):
        """Every op at bin ``magnitude``, as RandAugment applies it, signed ops
        negated at random."""
        bins = {name: np.full(batch_size, self.magnitude) for name in self._names}
        return bins, draw_flips(self._names, batch_size, rng)


# --- the paper's published (N, M) ----------------------------------------------------

#: RandAugment's published (N, M): Cubuk et al., "RandAugment: Practical automated data
#: augmentation with a reduced search space" (arXiv:1909.13719v2), (dataset, model) ->
#: (N ops per image, M). M is on the paper's scale, 0 to 10 with 10 the end of
#: AutoAugment's Table 6 ranges, and runs past 10 where the paper's search did (its
#: footnote 2). Every row the paper states, with where it states it.
PUBLISHED: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("CIFAR-10", "Wide-ResNet-28-2"): (3, 4),  # A.2.1
    ("CIFAR-10", "Wide-ResNet-28-10"): (3, 5),  # A.2.1
    ("CIFAR-10", "Shake-Shake (26 2x96d)"): (3, 9),  # A.2.1
    ("CIFAR-10", "PyramidNet + ShakeDrop"): (3, 7),  # A.2.1
    ("CIFAR-100", "Wide-ResNet-28-2"): (1, 2),  # 4.2
    ("CIFAR-100", "Wide-ResNet-28-10"): (2, 14),  # 4.2
    ("SVHN core", "Wide-ResNet-28-2"): (3, 9),  # A.2.2
    ("SVHN core", "Wide-ResNet-28-10"): (3, 9),  # A.2.2
    ("SVHN", "Wide-ResNet-28-2"): (3, 5),  # A.2.2
    ("SVHN", "Wide-ResNet-28-10"): (3, 7),  # A.2.2
    ("ImageNet", "ResNet-50"): (2, 9),  # A.2.3
    ("ImageNet", "EfficientNet-B5"): (2, 17),  # A.2.3: N = 2 and 3 "about the same"
    ("ImageNet", "EfficientNet-B7"): (2, 28),  # A.2.3
    ("COCO", "RetinaNet ResNet-101"): (1, 5),  # A.3
    ("COCO", "RetinaNet ResNet-200"): (1, 6),  # A.3
}

#: dropaug dataset -> the published row that stands in for it. The paper publishes none
#: of dropaug's pairs (ResNet-18 on CIFAR, any segmentation model). CIFAR takes the
#: Wide-ResNet-28-10 row, the paper's model nearest ResNet-18 in size (36.5M parameters
#: against 11M; Wide-ResNet-28-2 has 1.5M); segmentation takes ImageNet ResNet-50's, as
#: its backbones are ImageNet-pretrained. A dataset not listed raises.
STAND_INS: Dict[str, Tuple[str, str]] = {
    "cifar10": ("CIFAR-10", "Wide-ResNet-28-10"),
    "cifar100": ("CIFAR-100", "Wide-ResNet-28-10"),
    **{
        name: ("ImageNet", "ResNet-50")
        for name in (
            "a2i2haze", "acdc", "ade20k", "cityscapes", "idd", "isic2017", "isic2017_test", "loveda",
            "pascal_voc12", "polyp", "polyp_colondb", "polyp_cvc300", "polyp_etis", "potsdam", "synapse",
        )
    },
}

#: torchvision RandAugment's default bins. Bin b is the paper's level b / 3, so level 10
#: is bin 30; vocab's ``randaug_diff`` continues the grid past it for a level above 10.
GRID_BINS = 31
BINS_PER_LEVEL = 3


def published(dataset: str) -> Tuple[Tuple[str, str], Tuple[int, int]]:
    """``(row, (N, M))``: the published row that stands in for ``dataset`` and its value."""
    try:
        row = STAND_INS[dataset]
    except KeyError:
        raise ValueError(
            f"RandAugment publishes no (N, M) for dataset {dataset!r} and none stands in for "
            f"it; add one to STAND_INS (known: {sorted(STAND_INS)})"
        ) from None
    return row, PUBLISHED[row]


class RandAugmentDiffHost(AugHost):
    """RandAugment over vocab's ``randaug_diff``, at the paper's published (N, M) for
    ``dataset`` (``published``). Each image gets N independent draws, uniform over the
    13 ops plus Identity, each op at level M. A signed op's share of a draw is split
    evenly between its ``_pos`` and ``_neg`` (as in ``TrivialAugmentDiffHost``), so
    uniform q is RandAugment exactly; Identity keeps its 1/14 as ``none_prob``.

    Level M is bin 3M of torchvision's 31-bin grid. A level above 10 (CIFAR-100's 14)
    lies past Table 6's ranges, and the grid then has 3M + 1 bins (``randaug_diff``'s
    module docstring). ``draw_bins`` gives that bin for every op and image; the bins are
    unsigned, so the flips it returns are ignored."""

    name = "randaugment_diff"
    family = "randaug_diff"

    def __init__(
        self,
        *,
        dataset: str,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        self.dataset = str(dataset)
        self.row, (num_ops, level) = published(self.dataset)
        self.num_ops = int(num_ops)
        self.level = int(level)
        self.magnitude = BINS_PER_LEVEL * self.level
        self.num_magnitude_bins = max(GRID_BINS, self.magnitude + 1)

        names = split_ops(self.family)
        super().__init__(
            names,
            split_weights(names),
            none_prob=1.0 / len(OP_ORDER),
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    def draw_bins(self, batch_size, rng):
        bins = {name: np.full(batch_size, self.magnitude) for name in self._names}
        return bins, draw_flips(self._names, batch_size, rng)
