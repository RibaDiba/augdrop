"""RandAugment (Cubuk et al. 2020) as a host, with torchvision's op space and defaults.

Each image gets ``num_ops`` independent draws, uniform over 14 entries, all at
bin ``magnitude`` of ``num_magnitude_bins``. ``Identity`` is one of the 14. It is
the no-op draw, so it stays out of the vocabulary and keeps its 1/14 when q
reweights the other 13.

The vocabulary is vocab's ``randaug_default`` family minus Identity, by name and in
its order, so the probe can look each op up with ``vocab.get_op``. Hardcoded
rather than imported so the hosts stay free of torch; tests/hosts pins the two.
"""

from typing import Iterable, Optional

import numpy as np

from .base import AugHost

__all__ = ["RandAugmentHost", "OP_ORDER", "IDENTITY", "family_ops"]

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
