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

**A static prune** (``AugHost.prune``) removes every sub-policy that applies a pruned
op (at p > 0), for the whole run, as a q of 0 would. Its partner ops lose those
applications too, and an op whose sub-policies are all removed is pruned with it
(``prune`` returns it as implied). ``hold_none_prob`` has nothing to hold here.

**``autoaugment_diff``** (``AutoAugmentDiffHost``) is the same policy over vocab's
``randaug_diff`` (the ops made differentiable), and trains. torchvision's AutoAugment
and RandAugment share their ``_AUGMENTATION_SPACE`` formulas, so ``randaug_diff`` read
at 10 bins is AutoAugment's magnitudes; Invert stays ``autoaugment.Invert``. A signed
op is split into ``_pos`` / ``_neg``, so a step on it is a choice between two vocab ops
(its columns), each with share 0.5, as AutoAugment negates with p = 0.5. The tilt
above, per step: the step's factor is ``Σ share · q / base`` over its columns, raised
to p, and the step picks a column ∝ ``share · q / base``. One column per op (the
``autoaugment`` host) is the formula above exactly. So a q of 0 on ``Rotate_neg``
alone makes every Rotate step positive and removes no sub-policy; 0 on both
directions removes them, as for an unsigned op. A prune is the same with the
directions renormalized for the run.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import AugHost, draw_flips
from .randaugment import IDENTITY, OP_ORDER, SIGNED_OPS
from .sensaug import NONE_KEY

__all__ = ["AutoAugmentDiffHost", "AutoAugmentHost", "IMAGENET_POLICY", "AUTOAUGMENT_OPS", "vocab_name"]

#: The ops an AutoAugment policy may use, as torchvision ids: RandAugment's 13
#: (in vocab's order), then Invert. The vocabulary is the subset a given policy
#: actually applies, in this order.
AUTOAUGMENT_OPS = tuple(op for op in OP_ORDER if op != IDENTITY) + ("Invert",)

#: The "leave this image alone" entry of a ``draw_slots`` slot.
NO_OP = NONE_KEY[0]


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

#: One step of a sub-policy as the host holds it: (probability, the vocabulary columns
#: it chooses between, each column's share, magnitude bin or None).
Step = Tuple[float, np.ndarray, np.ndarray, Optional[int]]


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
        for s, sub in enumerate(self.policy):
            for op, prob, _ in sub:
                if op not in AUTOAUGMENT_OPS:
                    raise ValueError(
                        f"sub-policy {s} uses {op!r}, which is not an AutoAugment op"
                    )
                if not 0.0 <= prob <= 1.0:
                    raise ValueError(f"sub-policy {s}: {op} has probability {prob}")

        # An op the policy never applies (or only at p=0) has no mass to tilt and
        # nothing for the probe to measure against, so it is not in the vocabulary.
        applied = {op for sub in self.policy for op, prob, _ in sub if prob > 0}
        names = [name for op in AUTOAUGMENT_OPS if op in applied for name, _ in self._columns(op)]
        index = {name: i for i, name in enumerate(names)}

        # A step on an op the vocabulary leaves out has p = 0 and no columns.
        self._steps: List[List[Step]] = [
            [
                (
                    float(prob),
                    np.array([index[n] for n, _ in self._columns(op) if n in index], dtype=int),
                    np.array([share for n, share in self._columns(op) if n in index]),
                    bin_,
                )
                for op, prob, bin_ in sub
            ]
            for sub in self.policy
        ]
        # The sub-policies a static prune leaves (prune); all of them until then.
        self._allowed = np.ones(len(self.policy), dtype=bool)
        self._names = tuple(names)
        weights, realized, self._within = self._realize(np.ones(len(names)))
        self._subpolicy_probs = weights / weights.sum()

        # Per op, the bins the policy applies it at, each weighted by its step's
        # probability. Sub-policies are equally likely, so their 1/25 cancels.
        bin_weights: dict = {}
        for s, sub in enumerate(self.policy):
            for op, prob, bin_ in sub:
                if bin_ is None or prob == 0:
                    continue
                if not 0 <= bin_ < self.num_magnitude_bins:
                    raise ValueError(
                        f"sub-policy {s}: {op} at bin {bin_}, outside [0, {self.num_magnitude_bins})"
                    )
                for name, _ in self._columns(op):
                    per_op = bin_weights.setdefault(name, {})
                    per_op[int(bin_)] = per_op.get(int(bin_), 0.0) + prob
        self._op_bins = {
            name: (np.array(sorted(b)), np.array([b[k] for k in sorted(b)]) / sum(b.values()))
            for name, b in bin_weights.items()
        }

        super().__init__(
            names,
            self._subpolicy_probs @ realized,
            none_prob=0.0,
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    def _columns(self, op: str) -> Tuple[Tuple[str, float], ...]:
        """The vocab ops a step on torchvision op ``op`` chooses between, each with its
        share of the step: one, here (a signed op flips inside the torchvision op)."""
        return ((vocab_name(op), 1.0),)

    def _realize(self, ratio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
        """For a per-op tilt ``ratio`` (q / base): each sub-policy's weight (unnormalized,
        0 outside ``_allowed``), the expected applications of each op per sub-policy, and
        per step the probability of each of its columns (None for a step that never
        applies). A step contributes ``(Σ share · ratio) ** p`` to its sub-policy's weight;
        0.0 ** 0.0 == 1.0, so a p=0 step on a dropped op does not zero its row."""
        weights = self._allowed.astype(np.float64)
        realized = np.zeros((len(self.policy), len(ratio)))
        within = []
        for s, steps in enumerate(self._steps):
            row = []
            for prob, cols, shares, _ in steps:
                mass = shares * ratio[cols]
                total = float(mass.sum())
                if prob == 0:
                    row.append(None)
                    continue
                weights[s] *= total**prob
                if total > 0:
                    realized[s, cols] += prob * mass / total
                    row.append(mass / total)
                else:
                    row.append(None)
            within.append(row)
        return weights, realized, within

    def draw_bins(self, batch_size, rng):
        """Per op and image, a bin drawn from the bins the policy applies that op at,
        weighted by each step's probability over equally likely sub-policies: the
        published policy, not the q-tilted one. Ops the policy applies without a bin
        (AutoContrast, Equalize, Invert) have no magnitude and are left out. Signed
        ops negated at random."""
        bins = {
            name: values[rng.choice(len(values), size=batch_size, p=probs)]
            for name, (values, probs) in self._op_bins.items()
        }
        return bins, draw_flips(list(bins), batch_size, rng)

    def subpolicy_probs(self) -> np.ndarray:
        """The probability of drawing each row of ``policy``. The sampler reads this."""
        return self._subpolicy_probs.copy()

    def _prune_base(self, keep: np.ndarray) -> np.ndarray:
        """Remove every sub-policy with a step (p > 0) all of whose columns are pruned,
        and spread each remaining step over its kept columns; the base is what the rest
        realize."""
        allowed = np.array(
            [all(prob == 0 or keep[cols].any() for prob, cols, _, _ in steps) for steps in self._steps]
        )
        if not allowed.any():
            pruned = [n for n, k in zip(self._names, keep) if not k]
            raise ValueError(f"every sub-policy applies one of {pruned}, so the prune leaves nothing to draw")
        self._allowed = allowed
        for steps in self._steps:
            for k, (prob, cols, shares, bin_) in enumerate(steps):
                kept = shares * keep[cols]
                if kept.sum() > 0:
                    steps[k] = (prob, cols, kept / kept.sum(), bin_)
        weights, realized, self._within = self._realize(np.ones(len(self._names)))
        self._subpolicy_probs = weights / weights.sum()
        realized = self._subpolicy_probs @ realized
        return realized / realized.sum()

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        # base > 0 for every op but a pruned one, whose q is 0 (set_weights) and whose
        # share is 0 in every step (_prune_base).
        ratio = np.divide(q, self._base, out=np.ones_like(q), where=self._base > 0)
        weights, realized, within = self._realize(ratio)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            dropped = [n for n, v in zip(self._names, q) if v == 0]
            raise ValueError(
                f"q drops {dropped}, and every sub-policy applies at least one of "
                f"them, so no sub-policy is left to draw"
            )
        self._subpolicy_probs = weights / total
        self._within = within
        realized = self._subpolicy_probs @ realized
        return realized / realized.sum()


class AutoAugmentDiffHost(AutoAugmentHost):
    """AutoAugment's policy over vocab's ``randaug_diff`` (module docstring): each signed
    op split into ``_pos`` / ``_neg``, each half of its step. Trains: ``draw_slots`` is
    the policy's draw, q applied. ``draw_bins`` gives both directions of an op the op's
    bins; the bins are unsigned, so the flips it returns are ignored."""

    name = "autoaugment_diff"
    family = "randaug_diff"

    def _columns(self, op: str) -> Tuple[Tuple[str, float], ...]:
        if op in SIGNED_OPS:
            return ((f"{self.family}.{op}_pos", 0.5), (f"{self.family}.{op}_neg", 0.5))
        return ((f"{self.family}.{op}" if op in OP_ORDER else vocab_name(op), 1.0),)

    def draw_slots(self, batch_size: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
        """The method's draw for one batch: per step of a sub-policy, ``(names, bins)``,
        each image's op for that step and its bin (0 for an op with no magnitude),
        ``NO_OP`` where the step did not fire. Per image a sub-policy from
        ``subpolicy_probs()``, then each step with its probability, at a column drawn by
        the step's (tilted) shares. q is applied (module docstring)."""
        width = max(len(sub) for sub in self.policy)
        slots = [(np.full(batch_size, NO_OP, dtype=object), np.zeros(batch_size, dtype=int)) for _ in range(width)]
        rows = rng.choice(len(self.policy), size=batch_size, p=self._subpolicy_probs)
        for i, s in enumerate(rows):
            for k, ((prob, cols, _, bin_), within) in enumerate(zip(self._steps[s], self._within[s])):
                if not rng.random() < prob:
                    continue
                col = cols[int(rng.choice(len(cols), p=within))]
                slots[k][0][i] = self._names[col]
                slots[k][1][i] = 0 if bin_ is None else int(bin_)
        return slots
