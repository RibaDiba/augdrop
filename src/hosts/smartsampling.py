"""SmartSamplingAugment (Negassi, Wagner & Reiterer 2022) as a host.

"Smart(Sampling)Augment: Optimal and Efficient Data Augmentation for Semantic
Segmentation" (arXiv 2111.00487; Algorithms 15(5):165, 2022), Algorithm 2, the
parameter-free method (SmartAugment, the same paper's Bayesian-optimized one, is not
ported: it needs 50 trainings per dataset and publishes no configuration). Per image:

- with probability P it is augmented, else left alone ("Sample var from [0, 1]; if
  var > P: do not augment"). P is "linearly anneal[ed] ... from 0 to 1 ... over the whole
  training epochs"; here ``min(1, step / max_iters)`` for the 1-based step being trained,
  Sensaug's ``(iter + 1) / max_iters`` progress, per iteration, since dropaug trains by
  iteration (the paper updates P once per epoch; the two differ by under one epoch's
  worth of P)
- 2 ops are drawn from A by the weights W, distinct ("random sample 2 ops a_i, a_j"; its
  SmartAugment samples "without replacement", which the paper sets against RandAugment).
  Drawn in turn, each by W among the ops not yet drawn (numpy's ``replace=False``), and
  applied in the order drawn
- both at one level M, uniform on the integers 5..30 of the paper's 0-30 scale ("the
  magnitude is sampled randomly from the interval [5, 30]", Sec. 3.2, in both versions;
  the journal's Algorithm 2 prints [5,28])

A and W are not printed. The paper says how they were made: from RandAugment's CIFAR-10
ablation, "We selected the augmentations with a positive average improvement and
transformed this value into probabilities". That ablation is RandAugment's Table 5
(arXiv:1909.13719v2): rotate 1.3, shear-x 0.9, shear-y 0.9, translate-y 0.4,
translate-x 0.4, autoContrast 0.1, sharpness 0.1, identity 0.1, everything else <= 0.
Identity is left out: SmartAugment's Table 2 lists it "only for the RandAugment and
TrivialAugment approaches", P taking its place. W is Delta / sum(Delta), the plainest
reading of "transformed into probabilities"; it matches the paper's "rotation is strongly
weighted" (Sec. 4.4). No code was released (github.com/mvg-inatech/SmartAugment holds
only a README), so all of this is a reconstruction; the level-to-op map is in
vocab/families/smartaug_diff.py.

The vocabulary is vocab's ``smartaug_diff`` restricted to A, the signed ops split into
``_pos`` / ``_neg`` (13 ops). A geometric op's sign is drawn with p = 0.5, so each
direction has half its W. Sharpness's direction is not drawn: it follows the level
(factor 0.1 + 1.8 M / 30 blurs below M = 15 and sharpens above), so ``_neg`` has W x 10/26
and ``_pos`` W x 16/26 (M = 15, the identity factor, counts as ``_pos`` at m = 0). That is
``base_probs()``, which also becomes the selector's prior: not uniform.

q. A q reweights each op by q / base (``sampling_probs() / base_probs()``) inside the
method's own draw: a transform's weight at level M is W times the tilt of its directions
available at M, and the direction is then picked by the tilted masses. Uniform q is
SmartSamplingAugment exactly. q never moves P: ``none_prob`` is 1 - P whatever q is. A
level at which fewer than 2 transforms have weight (only after q or a prune zeroes
most of them) gets the ones there are, possibly none.

The host needs the training step: ``set_step(runner.iter + 1)`` before each one
(tasks/mmseg/hooks.py's ``HostProgressHook``). A fresh host is at step 1. Pure numpy,
like every host.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .base import AugHost, draw_flips
from .sensaug import NONE_KEY

__all__ = [
    "DELTA",
    "FAMILY",
    "IDENTITY_LEVEL",
    "LEVELS",
    "NUM_OPS",
    "PARAMETER_MAX",
    "SmartSamplingDiffHost",
    "TRANSFORMS",
    "WEIGHTS",
]

FAMILY = "smartaug_diff"

#: RandAugment's Table 5, the ops with a positive average gain (%), Identity left out,
#: in smartaug_diff's (SmartAugment Table 2's) order.
DELTA: Dict[str, float] = {
    "Sharpness": 0.1,
    "AutoContrast": 0.1,
    "Rotate": 1.3,
    "ShearX": 0.9,
    "ShearY": 0.9,
    "TranslateX": 0.4,
    "TranslateY": 0.4,
}
TRANSFORMS = tuple(DELTA)

#: W: Delta / sum(Delta).
WEIGHTS = np.array(list(DELTA.values())) / sum(DELTA.values())

#: The paper's level scale, and the levels SmartSamplingAugment draws (inclusive).
PARAMETER_MAX = 30
LEVELS = (5, 30)

#: The level whose enhance factor is 1.0; Sharpness is _neg below it and _pos from it.
IDENTITY_LEVEL = 15

#: Ops per augmented image.
NUM_OPS = 2

#: The ops whose sign is drawn with p = 0.5 (smartaug_diff splits them by direction).
SIGNED = frozenset({"Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"})

#: The ops whose direction follows the level.
LEVEL_SIGNED = frozenset({"Sharpness"})

#: What an image that isn't augmented draws: tasks.augment's NO_OP.
NO_OP = NONE_KEY[0]

#: ``(names (B,), bins (B,))`` per slot, ``NO_OP`` for an image left alone in it.
Slot = Tuple[np.ndarray, np.ndarray]


def _names(transform_id: str) -> Tuple[str, ...]:
    if transform_id in SIGNED or transform_id in LEVEL_SIGNED:
        return f"{FAMILY}.{transform_id}_pos", f"{FAMILY}.{transform_id}_neg"
    return (f"{FAMILY}.{transform_id}",)


def _direction_share(name: str, level: int) -> float:
    """P(this direction | level) under the method: 1/2 for a signed op's either way, 1
    or 0 for a level-signed op's, 1 for an unsigned op."""
    if name.endswith(("_pos", "_neg")):
        transform_id = name.split(".", 1)[1].rsplit("_", 1)[0]
        if transform_id in LEVEL_SIGNED:
            return float((level >= IDENTITY_LEVEL) == name.endswith("_pos"))
        return 0.5
    return 1.0


class SmartSamplingDiffHost(AugHost):
    """SmartSamplingAugment over vocab's ``smartaug_diff`` (module docstring)."""

    name = "smartsampling_diff"
    family = FAMILY

    num_ops = NUM_OPS
    num_magnitude_bins = PARAMETER_MAX + 1  # bin b is level b

    def __init__(
        self,
        *,
        max_iters: int,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        if int(max_iters) < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        self.max_iters = int(max_iters)
        self._step = 1

        lo, hi = LEVELS
        self._levels = np.arange(lo, hi + 1)
        self._groups: List[Tuple[int, ...]] = []
        names, base = [], []
        for transform_id, w in zip(TRANSFORMS, WEIGHTS):
            group = []
            for name in _names(transform_id):
                group.append(len(names))
                names.append(name)
                base.append(w * np.mean([_direction_share(name, m) for m in self._levels]))
            self._groups.append(tuple(group))
        # b_n(M): W_t times P(direction n | M), per level and op.
        self._conditional = np.array(
            [
                [WEIGHTS[t] * _direction_share(names[i], m) for t, group in enumerate(self._groups) for i in group]
                for m in self._levels
            ]
        )
        super().__init__(
            names,
            base,
            none_prob=0.0,
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    # --- the annealed P ----------------------------------------------------------

    def set_step(self, step: int) -> None:
        """The 1-based training step about to run (mmengine's ``runner.iter + 1`` in
        ``before_train_iter``): sets P for the draws that follow."""
        if int(step) < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        self._step = int(step)

    @property
    def augment_prob(self) -> float:
        """P: the probability an image is augmented now, ``min(1, step / max_iters)``."""
        return min(1.0, self._step / self.max_iters)

    @property
    def none_prob(self) -> float:
        """1 - P: the probability an image is left alone now. Annealed, never moved by q."""
        return 1.0 - self.augment_prob

    def prune(self, names: Iterable[str], *, hold_none_prob: bool = True) -> Tuple[str, ...]:
        """``AugHost.prune``. P is a schedule, not a draw in the bank, so there is no
        no-op draw to renormalize with the survivors: ``hold_none_prob=False`` raises."""
        names = tuple(names)
        if names and not hold_none_prob:
            raise ValueError(
                f"{type(self).__name__}: P(augment) is annealed by the step, not drawn "
                f"against the ops, so hold_none_prob=False has nothing to renormalize"
            )
        return super().prune(names, hold_none_prob=True)

    # --- the draws -----------------------------------------------------------------

    def draw_slots(self, batch_size: int, rng: np.random.Generator) -> List[Slot]:
        """The method's draw for one batch: ``NUM_OPS`` slots of ``(names, bins)``, slot k
        holding each image's k-th op and its bin (= its level), ``NO_OP`` where the image
        is not augmented or ran out of ops. q is applied (module docstring)."""
        base = self.base_probs()
        probs = self.sampling_probs()
        tilt = np.divide(probs, base, out=np.zeros_like(probs), where=base > 0)
        p_aug = self.augment_prob
        names = np.array(self._names, dtype=object)
        slots = [(np.full(batch_size, NO_OP, dtype=object), np.zeros(batch_size, dtype=int)) for _ in range(NUM_OPS)]
        for i in range(batch_size):
            if not rng.random() < p_aug:
                continue
            row = int(rng.integers(len(self._levels)))
            level = int(self._levels[row])
            weights = tilt * self._conditional[row]
            per_transform = np.array([weights[list(group)].sum() for group in self._groups])
            for k in range(NUM_OPS):
                total = per_transform.sum()
                if total <= 0:
                    break
                t = int(rng.choice(len(self._groups), p=per_transform / total))
                per_transform[t] = 0.0
                group = list(self._groups[t])
                within = weights[group]
                op = group[int(rng.choice(len(group), p=within / within.sum()))]
                slots[k][0][i] = names[op]
                slots[k][1][i] = level
        return slots

    def draw_bins(self, batch_size, rng):
        """The level SmartSamplingAugment would apply each op at, per image, for the probe:
        one uniform draw u per image shared across ops (common random numbers, as in
        ``TrivialAugmentHost.draw_bins``), mapped onto each op's own levels. A geometric op
        and AutoContrast take M = 5 + floor(26 u); Sharpness ``_neg`` the levels below 15
        and ``_pos`` the levels from 15, by the same u. The bins are unsigned, so the
        flips are ignored."""
        u = rng.random(batch_size)
        lo, hi = LEVELS

        def on(first, last):
            return first + np.minimum((u * (last - first + 1)).astype(int), last - first)

        bins = {}
        for name in self._names:
            transform_id = name.split(".", 1)[1].rsplit("_", 1)[0] if name.endswith(("_pos", "_neg")) else None
            if transform_id in LEVEL_SIGNED:
                bins[name] = on(IDENTITY_LEVEL, hi) if name.endswith("_pos") else on(lo, IDENTITY_LEVEL - 1)
            else:
                bins[name] = on(lo, hi)
        return bins, draw_flips(self._names, batch_size, rng)
