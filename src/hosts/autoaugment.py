"""AutoAugment (Cubuk et al. 2019) as a host, with torchvision's ImageNet policy.

AutoAugment does not draw ops. It draws one of 25 sub-policies uniformly, and
each sub-policy applies two ops, each with its own probability and magnitude
bin. The vocabulary is therefore the ops the policy can apply, and the base
distribution is each op's share of the expected applications.

**How q reaches the sub-policies.** A per-op q cannot be hit exactly by
reweighting 25 fixed pairs, so the host applies q as a tilt on sub-policies:

    w(s) ∝ prod over (a, p) in s of (q(a) / base(a)) ** p

With q = base · exp(-λ·red), that is ``exp(-λ · Σ p·red(a))``. It is Lever 3's
max-entropy tilt with a sub-policy scored by the redundancy it is expected to
apply. q = base leaves the policy uniform. q(a) = 0 removes every sub-policy that
can apply ``a``, so a dropped op is never applied. ``sampling_probs()`` reports
the op distribution the tilted policy actually realizes, which in general is
close to q but not equal to it.

**Names.** RandAugment took its op ranges from AutoAugment's Table 6, so vocab's
``randaug_default`` ops (the same torchvision dispatcher, in the same units) ARE
AutoAugment's ops, and the vocabulary uses their names. Only the bins differ (10
here vs RandAugment's 31), which is the host's business, not the op's. ``Invert``
is the exception: RandAugment dropped it, so it is vocab's ``autoaugment`` family,
``autoaugment.Invert``.

There is no no-op draw (``none_prob`` is 0). Every draw is a sub-policy, and how
often an image goes untouched depends on which sub-policy it drew. Reweighting
sub-policies therefore shifts that a little. The fixed pair structure makes this
unavoidable, and it is the one place a host does not hold "how often" exactly.
"""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .base import AugHost
from .randaugment import IDENTITY, OP_ORDER

__all__ = ["AutoAugmentHost", "IMAGENET_POLICY", "AUTOAUGMENT_OPS", "vocab_name"]

#: The ops an AutoAugment policy may use, as torchvision ids: RandAugment's 13
#: (in vocab's order), then Invert. The vocabulary is the subset a given policy
#: actually applies, in this order.
AUTOAUGMENT_OPS = tuple(op for op in OP_ORDER if op != IDENTITY) + ("Invert",)


def vocab_name(op: str) -> str:
    """The vocab op name for a torchvision AutoAugment op id."""
    return f"randaug_default.{op}" if op in OP_ORDER else f"autoaugment.{op}"


#: ``AutoAugment._get_policies(AutoAugmentPolicy.IMAGENET)``, verbatim from
#: torchvision 0.24: ((op, probability, magnitude bin or None), (op, ...)).
#: Five rows repeat earlier ones (e.g. row 25 is row 3, 1-indexed). That is the
#: published policy, so those sub-policies carry double weight; they are kept.
IMAGENET_POLICY = (
    (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
    (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
    (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
    (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
    (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
    (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
    (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
    (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
    (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
    (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
    (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
    (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
    (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
    (("Invert", 0.6, None), ("Equalize", 1.0, None)),
    (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
    (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
    (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
    (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
    (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
    (("Color", 0.4, 0), ("Equalize", 0.6, None)),
    (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
    (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
    (("Invert", 0.6, None), ("Equalize", 1.0, None)),
    (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
    (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
)

SubPolicy = Tuple[Tuple[str, float, Optional[int]], ...]


class AutoAugmentHost(AugHost):
    name = "autoaugment"

    #: torchvision's bin count for AutoAugment magnitudes.
    num_magnitude_bins = 10

    def __init__(
        self,
        *,
        policy: Sequence[SubPolicy] = IMAGENET_POLICY,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        self.policy = tuple(tuple(tuple(step) for step in sub) for sub in policy)
        if not self.policy:
            raise ValueError("policy has no sub-policies")

        # applications[s, a]: expected number of times sub-policy s applies op a.
        # It is also the exponent op a's tilt ratio gets in w(s).
        index = {op: i for i, op in enumerate(AUTOAUGMENT_OPS)}
        full = np.zeros((len(self.policy), len(AUTOAUGMENT_OPS)))
        for s, sub in enumerate(self.policy):
            for op, prob, _ in sub:
                if op not in index:
                    raise ValueError(
                        f"sub-policy {s} uses {op!r}, which is not an AutoAugment op"
                    )
                if not 0.0 <= prob <= 1.0:
                    raise ValueError(f"sub-policy {s}: {op} has probability {prob}")
                full[s, index[op]] += prob

        # An op the policy never applies (or only at p=0) has no mass to tilt and
        # nothing for the probe to measure against, so it is not in the vocabulary.
        applied = full.sum(axis=0) > 0
        self._applications = full[:, applied]
        self._subpolicy_probs = np.full(len(self.policy), 1.0 / len(self.policy))

        super().__init__(
            tuple(vocab_name(op) for op, keep in zip(AUTOAUGMENT_OPS, applied) if keep),
            self._subpolicy_probs @ self._applications,
            none_prob=0.0,
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    def subpolicy_probs(self) -> np.ndarray:
        """The probability of drawing each row of ``policy``. The sampler reads this."""
        return self._subpolicy_probs.copy()

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        ratio = q / self._base  # base > 0 for every vocabulary op, by construction
        # 0.0 ** 0.0 == 1.0, so a p=0 step on a dropped op does not zero its row.
        w = np.prod(ratio[None, :] ** self._applications, axis=1)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0:
            dropped = [n for n, v in zip(self._names, q) if v == 0]
            raise ValueError(
                f"q drops {dropped}, and every sub-policy applies at least one of "
                f"them, so no sub-policy is left to draw"
            )
        self._subpolicy_probs = w / total
        realized = self._subpolicy_probs @ self._applications
        return realized / realized.sum()
