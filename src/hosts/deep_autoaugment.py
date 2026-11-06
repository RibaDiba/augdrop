"""Deep AutoAugment (Zheng et al., ICLR 2022) as a host: its published policy, per image.

A DeepAA policy is K layers (5 in the published ImageNet policies). Layer k is a
categorical ``p_k`` over T = 139 transforms, each an (op, magnitude level) pair of the
ImageNet search space (``DeepAA_utils.aug_op_imagenet_list``, 18 ops, ``l_mags = 13``
levels with each op's neutral level removed, ``policy._get_unique_policy``). Training
(``DeepAA_evaluate/deep_autoaugment.py``, ``Augmentation_DeepAA``) runs the layers in
turn on every image: in layer k it draws one transform from ``p_k`` and applies it at
``level = mag / (l_mags - 1)``, mapped to the op's range ``[low, high]`` by
``level * (high - low) + low`` (``apply_augment``). Each image draws afresh each time it is
loaded. This is that draw, per image, over vocab's ``deepaa`` ops (the module docstring
of vocab/families/deepaa.py says how each op is ported).

The policy is the published one (``third_party/DeepAA/policy_port/policy_DeepAA_
imagenet_1.npz``, the config's ``policy_file``): DeepAA's own ``.npz`` (``save_policy``:
``policy_probs`` (K, T), ``l_ops``, ``l_mags``, ``ops`` / ``mags`` (T, 1), ``op_names``).
``load_policy`` checks its op names and ranges against the ImageNet list and its
(op, level) table against ``unique_policy``, this module's port of
``_get_unique_policy``. DeepAA's search is not ported (the user's choice, 2026-11-03).

**Transforms -> ops.** Each transform becomes one or more atoms ``(vocab op or NO_OP,
magnitude in the op's units, share)``:

- Identity: ``NO_OP``.
- ShearX/Y, TranslateX/Y, Rotate at signed v: the trainer flips v's sign with p = 0.5
  (``random_mirror = True`` in ``deep_autoaugment.py``; the search's ops do not, and the
  paper does not say), so ``_pos`` and ``_neg`` at |v|, half each. That is the user's
  choice (2026-11-03): train as the released trainer ran. ``random_mirror=False`` applies
  the searched sign instead.
- Contrast, Color, Brightness, Sharpness at factor v: ``_pos`` at v - 1 above 1,
  ``_neg`` at 1 - v below (the trainer does not mirror these).
- Solarize at threshold v; Posterize at ``int(v)`` bits, as DeepAA's ``Posterize`` takes.
- AutoContrast, Invert, Equalize: the op.
- RandFlip: Flip half the time, ``NO_OP`` the other half (its coin).
- RandCutout: Cutout. RandResizeCrop_imagenet: RandomResizedCrop.

**base_probs, q and the prune** are Faster AutoAugment's rule (hosts/faster_autoaugment)
with a layer as a stage. Layer k applies op a with mass ``m_k(a)``, the sum of its atoms'
``p_k(t) share``, and leaves the image alone with its ``NO_OP`` atoms' mass.
``base_probs()`` is ``sum_k m_k(a)`` normalized, and ``none_prob`` the mean ``NO_OP`` mass
of a layer. A q multiplies each atom by ``r = q / base`` of its op and rescales the layer's
op atoms to their mass, so each layer's ``NO_OP`` mass is held: q changes what a layer
applies, never how often. Within a mirrored transform it can favour ``_pos`` over
``_neg``. The static prune zeroes the pruned ops' atoms; ``hold_none_prob`` (the default)
keeps each layer's op mass, False lets the pruned mass become ``NO_OP``.

**Batch augmentation.** ``num_copies`` (default 1) is how many times the mmseg
preprocessor repeats each image in the batch, each copy drawn independently. DeepAA's
ImageNet config (``resnet50_imagenet_DeepAA_8x256_1.yaml``) trains 8 copies, every worker
on the same batch (``all_workers_use_the_same_batches``); ``num_copies: 8`` is that.

**The search in the run** (``policy_file: search``, ``SEARCH``; the probe-and-prune recipe,
the user's choice, 2026-11-05: train, pause, prune, search, resume). The host starts with
no policy and draws nothing, as DeepAA's own ``pretrain()`` trains on its default
augmentation only (here the recipe's pipeline). Its ``base_probs()`` meanwhile is the
search's starting point's op shares (``initial_policy``), the prior NMS reads at the probe.
At the pause, ``DeepAASearchHook`` (tasks/mmseg/deepaa_hooks.py) searches the K layers on
the frozen model over the transforms the prune left (``src/deepaa/``) and installs them
(``install_policy``): the static prune is applied to them and a q in effect carries over as
its ratio (PruneHook's M is a ratio of 1 or 0), each layer's no-op mass held.

No ``draw_magnitudes``: the probe-and-prune recipe probes at a fixed magnitude.

Pure numpy, like every host. The op names and ranges are hardcoded here so the host
stays free of torch; vocab/families/deepaa.py is the implementation.
"""

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import AugHost
from .sensaug import NONE_KEY

__all__ = [
    "DEEPAA_OPS",
    "DeepAutoAugmentHost",
    "FAMILY",
    "SEARCH",
    "VOCAB_NAMES",
    "initial_policy",
    "load_policy",
    "transform_atoms",
    "unique_policy",
]

FAMILY = "deepaa"

#: The "leave this image alone" draw (``tasks.augment.NO_OP``).
NO_OP = NONE_KEY[0]

#: ``DeepAA_utils.aug_op_imagenet_list``'s names and ranges, as the published ImageNet
#: policies write them (``op_names``). Their op 16 is ``RandCutout``, where the repo's
#: list now says ``RandCutout60``: the policies are the ground truth.
DEEPAA_OPS: Tuple[Tuple[str, float, float], ...] = (
    ("Identity", 0.0, 1.0),
    ("ShearX", -0.3, 0.3),
    ("ShearY", -0.3, 0.3),
    ("TranslateX", -0.45, 0.45),
    ("TranslateY", -0.45, 0.45),
    ("Rotate", -30.0, 30.0),
    ("AutoContrast", 0.0, 1.0),
    ("Invert", 0.0, 1.0),
    ("Equalize", 0.0, 1.0),
    ("Solarize", 0.0, 256.0),
    ("Posterize", 4.0, 8.0),
    ("Contrast", 0.1, 1.9),
    ("Color", 0.1, 1.9),
    ("Brightness", 0.1, 1.9),
    ("Sharpness", 0.1, 1.9),
    ("RandFlip", 0.0, 1.0),
    ("RandCutout", 0.0, 1.0),
    ("RandResizeCrop_imagenet", 0.0, 1.0),
)

#: DeepAA's magnitude levels (``--l_mags``, every README command).
L_MAGS = 13

#: ``policy_file`` for the search in the run: no policy until ``install_policy``.
SEARCH = "search"

MIRRORED = frozenset({"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"})
ENHANCE = frozenset({"Contrast", "Color", "Brightness", "Sharpness"})
UNSIGNED = {"AutoContrast": "AutoContrast", "Invert": "Invert", "Equalize": "Equalize"}
RANDOM = {"RandFlip": "Flip", "RandCutout": "Cutout", "RandResizeCrop_imagenet": "RandomResizedCrop"}

#: vocab ``deepaa``'s ops, in its order, with each op's mag_range upper end (None: no
#: magnitude; Solarize and Posterize are (lower, upper)).
_RANGES: Dict[str, object] = {}
for _t, _hi in (("ShearX", 0.3), ("ShearY", 0.3), ("TranslateX", 0.45), ("TranslateY", 0.45), ("Rotate", 30.0)):
    _RANGES[f"{FAMILY}.{_t}_pos"] = _RANGES[f"{FAMILY}.{_t}_neg"] = (0.0, _hi)
for _t in ("AutoContrast", "Invert", "Equalize"):
    _RANGES[f"{FAMILY}.{_t}"] = None
_RANGES[f"{FAMILY}.Solarize"] = (0.0, 256.0)
_RANGES[f"{FAMILY}.Posterize"] = (4.0, 8.0)
for _t in ("Contrast", "Color", "Brightness", "Sharpness"):
    _RANGES[f"{FAMILY}.{_t}_pos"] = _RANGES[f"{FAMILY}.{_t}_neg"] = (0.0, 0.9)
for _t in ("Flip", "Cutout", "RandomResizedCrop"):
    _RANGES[f"{FAMILY}.{_t}"] = None

VOCAB_NAMES: Tuple[str, ...] = tuple(_RANGES)

#: ``(vocab op or NO_OP, magnitude, share)``.
Atom = Tuple[str, float, float]

#: Each layer's draw for a batch: ``(names (B,), magnitudes (B,))``.
Stage = Tuple[np.ndarray, np.ndarray]


def mid_magnitude(name: str, l_mags: int):
    """``augmentation.get_mid_magnitude``: the level removed from an op's grid (its
    neutral level), ``None`` for an op with no magnitude, ``"random"`` for a random op."""
    if name in RANDOM:
        return "random"
    if name in ("Identity",) or name in UNSIGNED:
        return None
    if name in ("Solarize", "Posterize"):
        return l_mags - 1
    return (l_mags - 1) // 2


def unique_policy(names: Sequence[str], l_mags: int) -> Tuple[np.ndarray, np.ndarray]:
    """``DA_Policy_logits._get_unique_policy``: the (op index, level) of each transform,
    (T,) each. An op with a magnitude has every level but its neutral one; an op without
    one, and a random op, has one transform at level ``(l_mags - 1) // 2``."""
    ops, mags = [], []
    for k, name in enumerate(names):
        mid = mid_magnitude(name, l_mags)
        if mid == "random" or mid is None:
            ops.append(k)
            mags.append((l_mags - 1) // 2)
        else:
            for level in range(l_mags):
                if level != mid:
                    ops.append(k)
                    mags.append(level)
    return np.array(ops, dtype=np.int64), np.array(mags, dtype=np.int64)


def _clip(name: str, value: float) -> float:
    lo, hi = _RANGES[name]
    return float(min(max(value, lo), hi))


def transform_atoms(name: str, value: float, random_mirror: bool = True) -> List[Atom]:
    """The atoms DeepAA's op ``name`` at value ``value`` (in its own range) becomes
    (module docstring). Magnitudes are clipped to the op's range against float error."""
    if name == "Identity":
        return [(NO_OP, 0.0, 1.0)]
    if name in MIRRORED:
        pos, neg = f"{FAMILY}.{name}_pos", f"{FAMILY}.{name}_neg"
        m = abs(value)
        if random_mirror:
            return [(pos, _clip(pos, m), 0.5), (neg, _clip(neg, m), 0.5)]
        op = pos if value > 0 else neg
        return [(op, _clip(op, m), 1.0)]
    if name in ENHANCE:
        op = f"{FAMILY}.{name}_pos" if value > 1.0 else f"{FAMILY}.{name}_neg"
        return [(op, _clip(op, abs(value - 1.0)), 1.0)]
    if name == "Solarize":
        return [(f"{FAMILY}.Solarize", _clip(f"{FAMILY}.Solarize", value), 1.0)]
    if name == "Posterize":
        return [(f"{FAMILY}.Posterize", _clip(f"{FAMILY}.Posterize", float(max(1, int(value)))), 1.0)]
    if name in UNSIGNED:
        return [(f"{FAMILY}.{UNSIGNED[name]}", 0.0, 1.0)]
    if name == "RandFlip":
        return [(f"{FAMILY}.Flip", 0.0, 0.5), (NO_OP, 0.0, 0.5)]
    if name in RANDOM:
        return [(f"{FAMILY}.{RANDOM[name]}", 0.0, 1.0)]
    raise KeyError(f"DeepAA op {name!r} is not in the ImageNet search space")


def _value(op: int, level: int, l_mags: int) -> float:
    """``apply_augment``'s magnitude: ``level / (l_mags - 1)`` mapped into the op's range,
    in float32 as the trainer computes it."""
    _, low, high = DEEPAA_OPS[op]
    unit = np.float32(level) / np.float32(l_mags - 1)
    return float(unit * np.float32(high - low) + np.float32(low))


def initial_policy(num_layers: int = 5) -> dict:
    """DeepAA's starting point (``policy_init='identity'``): every layer's logits
    ``[8, 0, ..., 0]`` minus their mean, so Identity has about 0.956. A stand-in that lets
    the host be built without a policy file, never trained on."""
    names = [n for n, _, _ in DEEPAA_OPS]
    ops, mags = unique_policy(names, L_MAGS)
    logits = np.zeros(len(ops))
    logits[0] = 8.0
    logits -= logits.mean()
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return {
        "policy_probs": np.tile(probs, (int(num_layers), 1)),
        "l_mags": L_MAGS,
        "ops": ops,
        "mags": mags,
    }


_NAME = re.compile(r"^(?P<op>[A-Za-z_0-9]+):\((?P<low>[-0-9.e]+),,(?P<high>[-0-9.e]+)\)$")


def load_policy(path: str) -> dict:
    """A DeepAA policy file (``DeepAA_utils.save_policy``), checked: its ops and ranges are
    the ImageNet list's, its (op, level) table is ``unique_policy``'s, and each layer is a
    distribution (renormalized against float32 rounding)."""
    try:
        data = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no DeepAA policy at {path!r}; the published ones are in the third_party/DeepAA "
            f"submodule (git submodule update --init third_party/DeepAA)"
        ) from None
    with data:
        missing = {"policy_probs", "l_ops", "l_mags", "ops", "mags", "op_names"} - set(data.files)
        if missing:
            raise ValueError(f"{path}: not a DeepAA policy file (missing {sorted(missing)})")
        probs = np.asarray(data["policy_probs"], dtype=np.float64)
        l_mags = int(data["l_mags"])
        ops = np.asarray(data["ops"]).reshape(-1).astype(np.int64)
        mags = np.asarray(data["mags"]).reshape(-1).astype(np.int64)
        names = [str(n) for n in data["op_names"]]
        l_ops = int(data["l_ops"])

    parsed = []
    for raw in names:
        match = _NAME.match(raw)
        if match is None:
            raise ValueError(f"{path}: op name {raw!r} is not DeepAA's 'Op:(low,,high)'")
        parsed.append((match["op"], float(match["low"]), float(match["high"])))
    if len(parsed) != l_ops or [p[0] for p in parsed] != [n for n, _, _ in DEEPAA_OPS]:
        raise ValueError(
            f"{path}: ops {[p[0] for p in parsed]} are not the ImageNet search space "
            f"{[n for n, _, _ in DEEPAA_OPS]} (a CIFAR policy's RandCrop is not ported)"
        )
    for (name, low, high), (_, want_low, want_high) in zip(parsed, DEEPAA_OPS):
        if not (np.isclose(low, want_low) and np.isclose(high, want_high)):
            raise ValueError(f"{path}: {name}'s range is ({low}, {high}), not ({want_low}, {want_high})")
    want_ops, want_mags = unique_policy([n for n, _, _ in DEEPAA_OPS], l_mags)
    if not (np.array_equal(ops, want_ops) and np.array_equal(mags, want_mags)):
        raise ValueError(f"{path}: its (op, level) table is not _get_unique_policy's at l_mags={l_mags}")
    if probs.ndim != 2 or probs.shape[1] != len(ops):
        raise ValueError(f"{path}: policy_probs has shape {probs.shape}, want (K, {len(ops)})")
    if not np.isfinite(probs).all() or probs.min() < 0:
        raise ValueError(f"{path}: policy_probs holds negative or non-finite values")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise ValueError(f"{path}: layers sum to {sums.tolist()}, not 1")
    return {"policy_probs": probs / sums[:, None], "l_mags": l_mags, "ops": ops, "mags": mags}


class DeepAutoAugmentHost(AugHost):
    """Deep AutoAugment's policy over vocab ``deepaa`` (module docstring)."""

    name = "deep_autoaugment"
    family = FAMILY

    def __init__(
        self,
        *,
        policy_file: Optional[str] = None,
        random_mirror: bool = True,
        num_copies: int = 1,
        num_layers: int = 5,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        self.policy_file = None if policy_file is None else str(policy_file)
        #: True until ``install_policy``: the search in the run hasn't happened yet, and
        #: the host draws nothing.
        self.pending = self.policy_file == SEARCH
        policy = initial_policy(num_layers) if self.policy_file in (None, SEARCH) else load_policy(self.policy_file)
        # Trainable: a policy file, or the search in the run (which installs one). The
        # starting point alone (null) is probe-only (check_trainable).
        self.searched = self.policy_file is not None
        self.random_mirror = bool(random_mirror)
        self.num_copies = int(num_copies)
        if self.num_copies < 1:
            raise ValueError(f"num_copies must be >= 1, got {num_copies}")
        self._hold_layer_mass = True
        self._build(policy)

        total = self._columns(self._atoms).sum(axis=0)
        if total.sum() <= 0:
            raise ValueError(f"{self.policy_file or 'the policy'} never applies any op")
        super().__init__(
            VOCAB_NAMES,
            total,
            none_prob=self._layer_none_prob(self._atoms),
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )
        #: q / base per op, the last q's (ones before any); ``install_policy`` carries it.
        self._ratio = np.ones(len(VOCAB_NAMES))

    def _build(self, policy: dict) -> None:
        """The atoms of ``policy`` (``load_policy``'s dict): every transform's atoms per
        layer, untilted and unpruned."""
        probs = np.asarray(policy["policy_probs"], dtype=np.float64)
        self._policy = {
            "policy_probs": probs.copy(),
            "l_mags": int(policy["l_mags"]),
            "ops": np.asarray(policy["ops"]).reshape(-1).copy(),
            "mags": np.asarray(policy["mags"]).reshape(-1).copy(),
        }
        self.num_layers = int(probs.shape[0])
        column = {name: a for a, name in enumerate(VOCAB_NAMES)}
        atom_transform, atom_names, atom_mags, atom_share = [], [], [], []
        for t, (op, level) in enumerate(zip(self._policy["ops"], self._policy["mags"])):
            name = DEEPAA_OPS[int(op)][0]
            for vocab_name, magnitude, share in transform_atoms(
                name, _value(int(op), int(level), self._policy["l_mags"]), self.random_mirror
            ):
                atom_transform.append(t)
                atom_names.append(vocab_name)
                atom_mags.append(magnitude)
                atom_share.append(share)
        self._atom_names = np.array(atom_names, dtype=object)
        self._atom_mags = np.array(atom_mags, dtype=np.float64)
        self._atom_col = np.array([column.get(n, -1) for n in atom_names], dtype=np.int64)
        self._is_op = self._atom_col >= 0
        # (K, N): layer k draws atom n with this probability.
        self._atoms = probs[:, atom_transform] * np.array(atom_share)[None, :]
        self._onehot = np.zeros((len(atom_names), len(VOCAB_NAMES)))
        self._onehot[np.flatnonzero(self._is_op), self._atom_col[self._is_op]] = 1.0
        self._atom_base = self._atoms.copy()
        self._atom_layer = self._atoms.copy()

    def _layer_none_prob(self, atoms: np.ndarray) -> float:
        return float(np.clip(1.0 - self._op_mass(atoms).mean(), 0.0, 1.0 - 1e-12))

    def policy(self) -> dict:
        """The policy the host draws from (``load_policy``'s dict, untilted, unpruned): the
        published or searched one, or the starting point while ``pending``."""
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._policy.items()}

    def install_policy(self, policy: dict) -> None:
        """Install a policy searched in the run (``DeepAASearchHook``, at the pause):
        ``load_policy``'s dict, its (op, level) table the ImageNet space's. Its layers
        replace the current ones; the static prune is applied to them (each layer's op
        mass held, or not, as the prune says) and a q in effect carries over as its ratio,
        so PruneHook's M keeps its ops at 0. The host then draws from it."""
        want_ops, want_mags = unique_policy([n for n, _, _ in DEEPAA_OPS], int(policy["l_mags"]))
        if not (
            np.array_equal(np.asarray(policy["ops"]).reshape(-1), want_ops)
            and np.array_equal(np.asarray(policy["mags"]).reshape(-1), want_mags)
        ):
            raise ValueError("the policy's (op, level) table is not the ImageNet space's")
        ratio = self._ratio.copy()
        self._build(policy)
        if self._columns(self._atoms).sum() <= 0:
            raise ValueError("the policy never applies any op")
        if self._pruned:
            keep = np.array([n not in self._pruned for n in self._names])
            base = self._prune_base(keep)
        else:
            total = self._columns(self._atoms).sum(axis=0)
            base = total / total.sum()
        self._base = base
        self._none_prob = self._layer_none_prob(self._atom_base)
        self._ratio = ratio
        self._probs = self._tilted(ratio)
        self.pending = False

    # --- the layers' op masses ---------------------------------------------------------

    def _columns(self, atoms: np.ndarray) -> np.ndarray:
        """(K, A): each layer's mass on each vocab op."""
        return atoms @ self._onehot

    def _op_mass(self, atoms: np.ndarray) -> np.ndarray:
        """(K,): each layer's mass on any op (1 minus its ``NO_OP`` mass)."""
        return atoms[:, self._is_op].sum(axis=1)

    def _tilt(self, atoms: np.ndarray, ratio: np.ndarray, mass: np.ndarray) -> np.ndarray:
        """Each layer's op atoms times their op's ``ratio`` (A,), rescaled to the layer's
        op mass ``mass`` (K,); ``NO_OP`` atoms untouched. A layer left with no op mass
        applies nothing."""
        scale = np.where(self._is_op, ratio[np.clip(self._atom_col, 0, None)], 0.0)
        tilted = atoms * scale[None, :]
        sums = tilted.sum(axis=1, keepdims=True)
        factor = np.divide(mass[:, None], sums, out=np.zeros_like(sums), where=sums > 0)
        return np.where(self._is_op[None, :], tilted * factor, atoms)

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        base = self._base
        # An op the base never applies has no ratio to take: neutral, so a policy
        # installed later applies it as searched.
        self._ratio = np.divide(q, base, out=np.ones_like(q), where=base > 0)
        return self._tilted(self._ratio)

    def _tilted(self, ratio: np.ndarray) -> np.ndarray:
        """The realized distribution under ``ratio`` (A,), each layer's op mass held."""
        self._atom_layer = self._tilt(self._atom_base, ratio, self._op_mass(self._atom_base))
        realized = self._columns(self._atom_layer).sum(axis=0)
        total = float(realized.sum())
        if total <= 0:
            raise ValueError(f"{type(self).__name__}: q zeroes every op of every layer; nothing would be applied")
        return realized / total

    def prune(self, names: Iterable[str], *, hold_none_prob: bool = True) -> Tuple[str, ...]:
        """``AugHost.prune``, per layer (Faster AutoAugment's rule): ``hold_none_prob``
        keeps each layer's op mass, False lets the pruned mass become ``NO_OP``."""
        self._hold_layer_mass = bool(hold_none_prob)
        implied = super().prune(names, hold_none_prob=True)
        if self._pruned and not self._hold_layer_mass:
            self._none_prob = self._layer_none_prob(self._atom_base)
        return implied

    def _prune_base(self, keep: np.ndarray) -> np.ndarray:
        keep = keep.astype(np.float64)
        if self._hold_layer_mass:
            mass = self._op_mass(self._atoms)
        else:
            # The kept ops' own mass: the pruned ops' share becomes NO_OP.
            mass = (self._atoms * np.where(self._is_op, keep[np.clip(self._atom_col, 0, None)], 0.0)).sum(axis=1)
        atoms = self._tilt(self._atoms, keep, mass)
        total = self._columns(atoms).sum(axis=0)
        if total.sum() <= 0:
            raise ValueError(f"{type(self).__name__}: the prune leaves no op any layer applies")
        self._atom_base = atoms
        self._atom_layer = atoms.copy()
        self._ratio = np.ones(len(self._names))  # AugHost.prune clears any q
        return total / total.sum()

    def layer_probs(self) -> np.ndarray:
        """(K, A): the probability each layer applies each op now, q and the prune
        included. Each layer's no-op is 1 minus its row sum."""
        return self._columns(self._atom_layer)

    def prune_report(self, flag_at: float = 0.5) -> Dict[str, object]:
        """What the static prune did to each layer, for ``prune.json`` and train.py's
        summary: the Faster AutoAugment host's report and keys, a layer as a stage (its
        policy's op mass, the share of it the pruned ops held, the op mass it applies now;
        ``flagged`` the layers that lost more than ``flag_at``). While ``pending`` it
        describes the starting point."""
        before = self._op_mass(self._atoms)
        pruned = np.array([n in self._pruned for n in self._names])
        removed = self._columns(self._atoms)[:, pruned].sum(axis=1)
        share = np.divide(removed, before, out=np.zeros_like(before), where=before > 0)
        after = self._op_mass(self._atom_base)
        return {
            "hold_stage_mass": bool(self._hold_layer_mass),
            "stage_mass_before": before.round(6).tolist(),
            "stage_share_pruned": share.round(6).tolist(),
            "stage_mass_after": after.round(6).tolist(),
            "flagged": [
                {"layer": k, "stage": k, "share_pruned": round(float(share[k]), 6)}
                for k in range(self.num_layers)
                if share[k] > flag_at
            ],
        }

    # --- the draws -----------------------------------------------------------------------

    def draw_stages(self, batch_size: int, rng: np.random.Generator) -> List[Stage]:
        """The method's draw for one batch: K stages of ``(names, magnitudes)``, stage k
        holding each image's draw from layer k (an op at its magnitude in the op's units,
        or ``NO_OP``, 0), q and the prune applied. Every image draws independently. While
        ``pending`` (the search in the run hasn't installed a policy) every stage is
        ``NO_OP``."""
        if self.pending:
            return [(np.full(batch_size, NO_OP, dtype=object), np.zeros(batch_size)) for _ in range(self.num_layers)]
        names = np.append(self._atom_names, NO_OP)
        mags = np.append(self._atom_mags, 0.0)
        out: List[Stage] = []
        for k in range(self.num_layers):
            probs = np.append(self._atom_layer[k], max(0.0, 1.0 - float(self._atom_layer[k].sum())))
            picks = rng.choice(len(probs), size=batch_size, p=probs / probs.sum())
            magnitudes = mags[picks].copy()
            magnitudes[~np.append(self._is_op, False)[picks]] = 0.0
            out.append((names[picks].copy(), magnitudes))
        return out
