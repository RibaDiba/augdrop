"""TrivialAugmentWide (Müller & Hutter 2021) as a host, with torchvision's defaults.

One draw per image, uniform over the same 14 entries RandAugment uses, at a
magnitude drawn uniformly from ``num_magnitude_bins`` bins over the wide ranges.
``Identity`` is held out of the vocabulary and keeps its 1/14, as in
``randaugment.py``. The vocabulary is vocab's ``trivialaug_wide`` family: the same
ops as RandAugment's, at TrivialAugment's wide ranges.
"""

from typing import Iterable, Optional

import numpy as np

from .base import AugHost, draw_flips
from .randaugment import OP_ORDER, SIGNED_OPS, family_ops, split_ops, split_weights

__all__ = ["SIGNED_OPS", "TrivialAugmentDiffHost", "TrivialAugmentHost", "split_ops"]


class TrivialAugmentHost(AugHost):
    name = "trivialaugment"
    family = "trivialaug_wide"

    #: TrivialAugment applies exactly one op per image. Not a knob.
    num_ops = 1

    def __init__(
        self,
        *,
        num_magnitude_bins: int = 31,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        if num_magnitude_bins < 2:
            raise ValueError(f"num_magnitude_bins must be >= 2, got {num_magnitude_bins}")
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
        """A uniform bin per image, as TrivialAugment draws it, signed ops negated at
        random. One bin per image for every op (common random numbers, like
        ``sa``'s ``sampled_shared``): the ops are compared at the same draw, and
        independent draws would attenuate R."""
        shared = rng.integers(0, self.num_magnitude_bins, size=batch_size)
        bins = {name: shared.copy() for name in self._names}
        return bins, draw_flips(self._names, batch_size, rng)


class TrivialAugmentDiffHost(TrivialAugmentHost):
    """TrivialAugmentWide over vocab's ``trivialaug_diff``: the same policy, with the
    random sign made explicit. A signed op's 1/13 (given that some op is drawn) is
    split evenly between its ``_pos`` and ``_neg``, so the base distribution is 1/26 per
    direction and 1/13 for Posterize, Solarize, AutoContrast and Equalize; Identity
    keeps its 1/14 as ``none_prob``. Uniform q is TrivialAugmentWide exactly. A q can
    weight the two directions of one op differently, which the method itself cannot.

    ``draw_bins`` is TrivialAugment's (one uniform bin per image, shared across ops).
    The bins are unsigned, so the flips it returns are ignored."""

    name = "trivialaugment_diff"
    family = "trivialaug_diff"

    def __init__(
        self,
        *,
        num_magnitude_bins: int = 31,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        if num_magnitude_bins < 2:
            raise ValueError(f"num_magnitude_bins must be >= 2, got {num_magnitude_bins}")
        self.num_magnitude_bins = int(num_magnitude_bins)

        names = split_ops(self.family)
        AugHost.__init__(
            self,
            names,
            split_weights(names),
            none_prob=1.0 / len(OP_ORDER),
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )
