"""AugMix (Hendrycks et al. 2020) as a host, as Sensaug trained it.

Sensaug's AugMix arm (``AugMixTransform`` in laurayuzheng/SensAug,
sensaug/dataset/augmentations.py) is torchvision's ``AugMix`` with the arguments below,
applied to each training image and not to its mask. They are Sensaug's settings, kept
here rather than in a config: severity 3, width 3 and alpha 1 are also the AugMix
paper's CIFAR settings (google-research/augmix ``cifar.py``), but ``all_ops`` is not
(the paper's default leaves out Color, Contrast, Brightness and Sharpness).

Per image, as torchvision draws it: ``(m0, m1) ~ Dirichlet(alpha, alpha)`` splits the
output between the clean image and the chains, and the chain weights are
``Dirichlet(alpha, ..., alpha) * m1``. Each of the ``mixture_width`` chains applies
``chain_depth`` ops in sequence (uniform on 1..3 when negative), each op drawn uniformly
from the 13, each at a bin drawn uniformly from the first ``severity`` of AugMix's 10,
a signed op negated with p = 0.5. The output is ``m0 * x + sum_i w_i * chain_i(x)``.

The mask is left as it is: torchvision's AugMix takes the image alone and Sensaug did
not replay it on the label, so a chain that rotates, shears or translates leaves the
image and its mask misaligned in the mix. That is Sensaug's behaviour, kept on purpose
(tasks/augment.py applies it). No single warp of the mask could match a mix of chains
warped differently.

The vocabulary is vocab's ``augmix_diff``: the 13 ops, the 9 signed ones split into
``_pos`` / ``_neg`` with half the op's share each, so uniform q is AugMix exactly. q
reweights the op drawn at every step of every chain. AugMix has no Identity, so
``none_prob`` is 0 (bin 0 of a signed op is the identity, though, so a step can still
leave its chain as it was).
"""

from typing import Iterable, Optional, Tuple

import numpy as np

from .base import AugHost, draw_flips
from .randaugment import split_ops, split_weights

__all__ = ["ALL_OPS", "ALPHA", "AugMixDiffHost", "CHAIN_DEPTH", "MIXTURE_WIDTH", "PARAMETER_MAX", "SEVERITY"]

#: Sensaug's AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=True).
SEVERITY = 3
MIXTURE_WIDTH = 3
CHAIN_DEPTH = -1  # negative: each chain's depth uniform on 1..3
ALPHA = 1.0
ALL_OPS = True

#: torchvision AugMix's ``_PARAMETER_MAX``: the bins of each op's table.
PARAMETER_MAX = 10


class AugMixDiffHost(AugHost):
    """AugMix with Sensaug's settings over vocab's ``augmix_diff``."""

    name = "augmix_diff"
    family = "augmix_diff"

    severity = SEVERITY
    mixture_width = MIXTURE_WIDTH
    chain_depth = CHAIN_DEPTH
    alpha = ALPHA
    num_magnitude_bins = PARAMETER_MAX

    def __init__(
        self,
        *,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        names = split_ops(self.family)
        super().__init__(
            names,
            split_weights(names),
            none_prob=0.0,
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    def draw_bins(self, batch_size, rng):
        """A bin per image, uniform on the first ``severity``, as AugMix draws one per op
        application. One bin per image for every op (common random numbers, as in
        ``TrivialAugmentHost.draw_bins``). The bins are unsigned, so the flips are
        ignored. Training calls this once per chain step, so each application gets its
        own draw."""
        shared = rng.integers(0, self.severity, size=batch_size)
        bins = {name: shared.copy() for name in self._names}
        return bins, draw_flips(self._names, batch_size, rng)

    def draw_mix(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(depths, clean, chains)`` per image: each chain's number of ops
        ``(B, mixture_width)``, the clean image's weight ``(B,)`` and each chain's weight
        ``(B, mixture_width)``, drawn as torchvision's AugMix draws them. ``clean +
        chains.sum(1)`` is 1."""
        m = rng.dirichlet([self.alpha, self.alpha], size=batch_size)
        chains = rng.dirichlet([self.alpha] * self.mixture_width, size=batch_size) * m[:, 1:]
        if self.chain_depth > 0:
            depths = np.full((batch_size, self.mixture_width), self.chain_depth)
        else:
            depths = rng.integers(1, 4, size=(batch_size, self.mixture_width))
        return depths, m[:, 0], chains
