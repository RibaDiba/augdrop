"""Adversarial AutoAugment (Zhang et al., ICLR 2020) as a host: the policies its controller
samples each epoch, drawn per image.

A policy is 5 sub-policies of 2 operations, each operation a type (one of the paper's
16) and a magnitude (one of 10 bins); there are no probabilities: "we just slightly
modify the search space with removing the probability of each operation" (§3.2). Each
epoch the controller (an LSTM, ``src/advaa/``) samples M policies, and every training
image is repeated M times, copy m augmented by policy m (Algorithm 1, Eq. 4-6; M = 8). "For
each image in a mini-batch, only one sub-policy will be randomly selected to be applied":
per image, a sub-policy of its copy's policy, uniform, and its two operations in order.
``AdvAAControllerHook`` (tasks/mmseg/advaa_hooks.py) trains the controller and installs
each epoch's policies here (``set_policies``); this host draws from them.

**Magnitudes** (the user's choice, 2026-11-03, the paper's text): "The range of the
magnitude is also discretized uniformly into 10 values", over AutoAugment's Table 6
ranges (footnote 1), so bin b of an op with range [lo, hi] is ``linspace(lo, hi, 10)[b]``.
The signed ranges run through 0 ([-30, 30], [0.1, 1.9] about 1), so the bin carries the
sign: bins 0-4 are the ``_neg`` op and 5-9 the ``_pos`` one, at |value|, and no bin is a
signed op's identity. AutoAugment's TF code instead uses levels 0-9 of [0, max] with a
random sign; this does not. Solarize at bin 9 (256), Posterize at bin 9 (8 bits), Cutout
and SamplePairing at bin 0 are their ops' identities, as the paper's grid has them.
Posterize applies ``int(value)`` bits. AutoContrast, Invert and Equalize ignore the bin
(the controller still emits one, as the paper's 20 parameters count it).

**base_probs** is each op's share of the current policies' applications (every
sub-policy equally likely, both operations applied): it moves with the policies, as
MADAug's prior moves with its policy (``set_policies`` recomputes it). Before the first
policies (``policies() is None``) it is the share of the 160 (type, bin) atoms, the
controller's uniform start. ``none_prob`` is 0: every image applies two operations.

**q and the prune.** A q tilts each policy's sub-policy weights by ``prod r(op)``, ``r =
q / base`` (AutoAugment's rule, hosts/autoaugment.py, with every p = 1), and keeps the
ratio across epochs. A copy whose policy has no sub-policy left applies nothing. The
static prune is that tilt at r = 0, and ``action_mask()`` hands it to the controller, which
then never samples a pruned (type, bin): every policy it samples is free of pruned ops,
so ``hold_none_prob`` has nothing to hold. An op the installed policies don't apply has
base 0 and takes a neutral ratio (1), so a later epoch's policies apply it untilted.

**The online prune** (the probe-and-prune recipe's online kind, the user's choice,
2026-11-05: probe and prune every max_iters/8 with selector nms): PruneHook's q zeroes the
pruned ops at once, and ``AdvAAControllerHook`` then prunes the controller to
``action_mask(extra=<the latched ops>)`` and samples new policies over what is left.

**Batch augmentation.** ``num_copies`` is M: the mmseg preprocessor repeats the batch M
times copy-major before ``draw_stages``, which reads image i as copy i // B.

The probe: the config's ``fade:`` block, at a fixed magnitude in the recipe (no
``draw_magnitudes``). SamplePairing couples images and stays unprobed.

Pure numpy, like every host; the op names and ranges are hardcoded so the host stays
free of torch (vocab/families/advaa.py is the implementation).
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .base import AugHost
from .sensaug import NONE_KEY

__all__ = [
    "ATOM_COLUMNS",
    "AdversarialAutoAugmentHost",
    "FAMILY",
    "NUM_BINS",
    "TYPES",
    "VOCAB_NAMES",
    "bin_atom",
    "bin_value",
]

FAMILY = "advaa"

#: The "leave this image alone" draw (``tasks.augment.NO_OP``).
NO_OP = NONE_KEY[0]

AA_IMAGE_SIZE = 331.0

#: The paper's 16 operation types, in its §3.2 order, with AutoAugment's Table 6 range in
#: the vocab op's units (translates and Cutout as a fraction of the 331 px image), None
#: for an op with no magnitude.
TYPE_RANGES: Dict[str, Optional[Tuple[float, float]]] = {
    "ShearX": (-0.3, 0.3),
    "ShearY": (-0.3, 0.3),
    "TranslateX": (-150.0 / AA_IMAGE_SIZE, 150.0 / AA_IMAGE_SIZE),
    "TranslateY": (-150.0 / AA_IMAGE_SIZE, 150.0 / AA_IMAGE_SIZE),
    "Rotate": (-30.0, 30.0),
    "AutoContrast": None,
    "Invert": None,
    "Equalize": None,
    "Solarize": (0.0, 256.0),
    "Posterize": (4.0, 8.0),
    "Contrast": (0.1, 1.9),
    "Color": (0.1, 1.9),
    "Brightness": (0.1, 1.9),
    "Sharpness": (0.1, 1.9),
    "Cutout": (0.0, 60.0 / AA_IMAGE_SIZE),
    "SamplePairing": (0.0, 0.4),
}
TYPES: Tuple[str, ...] = tuple(TYPE_RANGES)

NUM_BINS = 10

SIGNED = frozenset({"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"})
ENHANCE = frozenset({"Contrast", "Color", "Brightness", "Sharpness"})

#: vocab ``advaa``'s ops, in its order, with each op's mag_range (None: no magnitude).
_RANGES: Dict[str, Optional[Tuple[float, float]]] = {}
for _t in TYPES:
    if _t in SIGNED or _t in ENHANCE:
        _hi = 0.9 if _t in ENHANCE else TYPE_RANGES[_t][1]
        _RANGES[f"{FAMILY}.{_t}_pos"] = _RANGES[f"{FAMILY}.{_t}_neg"] = (0.0, _hi)
    else:
        _RANGES[f"{FAMILY}.{_t}"] = TYPE_RANGES[_t]

VOCAB_NAMES: Tuple[str, ...] = tuple(_RANGES)

#: Each stage's draw for a batch: ``(names (B,), magnitudes (B,))``.
Stage = Tuple[np.ndarray, np.ndarray]


def bin_value(type_name: str, b: int) -> float:
    """Bin ``b`` of an op type: ``linspace(lo, hi, 10)[b]`` in its range (0 for an op
    with no magnitude)."""
    bounds = TYPE_RANGES[type_name]
    if bounds is None:
        return 0.0
    return float(np.linspace(bounds[0], bounds[1], NUM_BINS)[int(b)])


def _clip(name: str, value: float) -> float:
    lo, hi = _RANGES[name]
    return float(min(max(value, lo), hi))


def bin_atom(type_name: str, b: int) -> Tuple[str, float]:
    """The vocab op and magnitude (in its units) that type ``type_name`` at bin ``b``
    applies (module docstring)."""
    value = bin_value(type_name, b)
    if type_name in SIGNED:
        name = f"{FAMILY}.{type_name}_{'pos' if value > 0 else 'neg'}"
        return name, _clip(name, abs(value))
    if type_name in ENHANCE:
        name = f"{FAMILY}.{type_name}_{'pos' if value > 1.0 else 'neg'}"
        return name, _clip(name, abs(value - 1.0))
    name = f"{FAMILY}.{type_name}"
    if TYPE_RANGES[type_name] is None:
        return name, 0.0
    if type_name == "Posterize":
        return name, float(int(value))
    return name, _clip(name, value)


_ATOMS = [[bin_atom(t, b) for b in range(NUM_BINS)] for t in TYPES]
#: (16, 10): the vocab column, the op name and the magnitude of each (type, bin).
ATOM_COLUMNS = np.array([[VOCAB_NAMES.index(n) for n, _ in row] for row in _ATOMS], dtype=np.int64)
ATOM_NAMES = np.array([[n for n, _ in row] for row in _ATOMS], dtype=object)
ATOM_MAGNITUDES = np.array([[m for _, m in row] for row in _ATOMS], dtype=np.float64)


class AdversarialAutoAugmentHost(AugHost):
    """Adversarial AutoAugment's epoch policies over vocab ``advaa`` (module docstring)."""

    name = "adversarial_autoaugment"
    family = FAMILY

    def __init__(
        self,
        *,
        num_copies: int = 8,
        num_sub_policies: int = 5,
        ops_per_sub_policy: int = 2,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        self.num_copies = int(num_copies)
        self.num_sub_policies = int(num_sub_policies)
        self.ops_per_sub_policy = int(ops_per_sub_policy)
        if min(self.num_copies, self.num_sub_policies, self.ops_per_sub_policy) < 1:
            raise ValueError("num_copies, num_sub_policies and ops_per_sub_policy must be >= 1")
        self._onehot = np.eye(len(VOCAB_NAMES))
        self._ratio = np.ones(len(VOCAB_NAMES))
        self._keep = np.ones(len(VOCAB_NAMES))
        self._policies: Optional[np.ndarray] = None
        self._columns: Optional[np.ndarray] = None
        self._magnitudes: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        super().__init__(
            VOCAB_NAMES,
            self._realize(self._keep),
            none_prob=0.0,
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    # --- the realized distribution --------------------------------------------------------

    def _sub_policy_weights(self, ratio: np.ndarray) -> np.ndarray:
        """(M, S): each policy's sub-policy weights under ``ratio``, normalized per policy
        (a row of zeros where nothing is left)."""
        weights = ratio[self._columns].prod(axis=-1)
        sums = weights.sum(axis=1, keepdims=True)
        return np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0)

    def _realize(self, ratio: np.ndarray) -> np.ndarray:
        """Each op's share of the applications under ``ratio`` (A,), unnormalized: over the
        current policies, or the 160 atoms before the first."""
        if self._columns is None:
            return np.bincount(ATOM_COLUMNS.reshape(-1), minlength=len(VOCAB_NAMES)) * ratio
        weights = self._sub_policy_weights(ratio)
        counts = self._onehot[self._columns].sum(axis=2)  # (M, S, A)
        return (weights[..., None] * counts).sum(axis=(0, 1))

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        base = self._base
        self._ratio = np.divide(q, base, out=np.ones_like(q), where=base > 0) * self._keep
        return self._normalized(self._realize(self._ratio))

    def _normalized(self, realized: np.ndarray) -> np.ndarray:
        total = float(realized.sum())
        if total <= 0:
            raise ValueError(f"{type(self).__name__}: q zeroes every sub-policy of every policy")
        if self._columns is not None:
            self._weights = self._sub_policy_weights(self._ratio)
        return realized / total

    def _prune_base(self, keep: np.ndarray) -> np.ndarray:
        self._keep = keep.astype(np.float64)
        self._ratio = self._keep.copy()
        if self._columns is not None and self._keep[self._columns].min() == 0:
            raise ValueError(f"{type(self).__name__}: prune before the controller samples policies")
        return self._normalized(self._realize(self._ratio))

    def action_mask(self, extra: Iterable[str] = ()) -> np.ndarray:
        """(16, 10) bool: the (type, bin) pairs the controller may sample, i.e. those whose
        op is not pruned, statically or in ``extra`` (an online prune's ops)."""
        keep = self._keep.copy()
        for name in extra:
            keep[self._names.index(name)] = 0.0
        return keep[ATOM_COLUMNS] > 0

    # --- the policies -------------------------------------------------------------------

    def set_policies(self, actions) -> None:
        """Install an epoch's M policies: ``actions`` (M, S, J, 2) ints, ``[..., 0]`` the
        type, ``[..., 1]`` the bin. The base moves to their share; a q in effect is kept as
        its ratio."""
        actions = np.asarray(actions, dtype=np.int64)
        shape = (self.num_copies, self.num_sub_policies, self.ops_per_sub_policy, 2)
        if actions.shape != shape:
            raise ValueError(f"policies of shape {actions.shape}, want {shape}")
        types, bins = actions[..., 0], actions[..., 1]
        if types.min() < 0 or types.max() >= len(TYPES) or bins.min() < 0 or bins.max() >= NUM_BINS:
            raise ValueError("a policy's type or bin is out of range")
        columns = ATOM_COLUMNS[types, bins]
        if self._keep[columns].min() == 0:
            raise ValueError(f"a policy applies a pruned op ({sorted(self._pruned)}); mask the controller")
        self._policies = actions.copy()
        self._columns = columns
        self._magnitudes = ATOM_MAGNITUDES[types, bins]
        self._base = self._normalized(self._realize(self._keep))
        self._probs = self._normalized(self._realize(self._ratio))

    def policies(self) -> Optional[np.ndarray]:
        """The installed (M, S, J, 2) policies, or None before the first."""
        return None if self._policies is None else self._policies.copy()

    def describe(self) -> List[str]:
        """One line per installed policy: its sub-policies as ``op@bin`` pairs."""
        if self._policies is None:
            return []
        return [
            " | ".join(", ".join(f"{TYPES[t]}@{b}" for t, b in sub) for sub in policy)
            for policy in self._policies.tolist()
        ]

    # --- the draws -----------------------------------------------------------------------

    def draw_stages(self, batch_size: int, rng: np.random.Generator) -> List[Stage]:
        """The method's draw for one batch of ``batch_size = B * M`` images, copy-major:
        image i, of copy ``m = i // B``, takes a sub-policy of policy m (uniform, or by the
        tilted weights) and stage j holds its j-th operation at its bin's magnitude. A copy
        with no sub-policy left applies nothing (``NO_OP``)."""
        if self._policies is None:
            raise RuntimeError(
                f"host {self.name!r} has no policies yet: AdvAAControllerHook samples and installs "
                f"them before training (config.py adds it for a host with a controller: block)"
            )
        copies = self.num_copies
        if batch_size % copies:
            raise ValueError(f"a batch of {batch_size} is not {copies} copies of one batch")
        per_copy = batch_size // copies
        names = [np.full(batch_size, NO_OP, dtype=object) for _ in range(self.ops_per_sub_policy)]
        mags = [np.zeros(batch_size) for _ in range(self.ops_per_sub_policy)]
        vocab = np.array(VOCAB_NAMES, dtype=object)
        for m in range(copies):
            weights = self._weights[m]
            if weights.sum() <= 0:
                continue
            rows = rng.choice(self.num_sub_policies, size=per_copy, p=weights)
            span = slice(m * per_copy, (m + 1) * per_copy)
            for j in range(self.ops_per_sub_policy):
                names[j][span] = vocab[self._columns[m, rows, j]]
                mags[j][span] = self._magnitudes[m, rows, j]
        return list(zip(names, mags))
