"""Greedy AutoAugment (Naghizadeh, Abavisani & Metaxas, 2020) as a host.

Paper: arXiv 1908.00704 (Pattern Recognition Letters 2020). Code:
github.com/arnaghizadeh/greedy_auto_augment (no license file), ``Search/main.py`` for the
search and ``Train/importer/custom_augment.py`` / ``found_policies.py`` for training.

**The search** (``src/greedyaa/``, run before training by ``scripts/gaa_search.py``) tries
every (op, level) pair at probability 1 on a child network, keeps the best, appends every
pair to it, and so on for ``levels`` levels (Algorithm 1). The policy is the ``top`` (25)
best trials over all levels, ranked, each a chain of up to ``levels`` (op, level) steps.

**Training** (``func_gautoaug``). The ranked chains are cut into sets of ``set_size`` (5):
S_1 the best five, and so on. Per image the code draws one r uniform on [0, 1] and scans
``v = (1, (1/2)^a, (1/3)^a, ...)`` (a = ``alpha`` = 2) from the right, taking the first
index with r <= v_i: index 0 is the image as it is, index j the set S_j. So

    P(original) = 1 - (1/2)^a,   P(S_j) = (1/(j+1))^a - (1/(j+2))^a,   P(S_n) = (1/(n+1))^a

(0.75, then 0.139, 0.049, 0.023, 0.012, 0.028 at a = 2 over five sets: the last set gets
the scan's remainder, so it outweighs the fourth). Within the set, a chain uniformly
(``found_policies``' tail draws one per block and ``func_gautoaug`` keeps the block it
picks), and every step of it applied, at probability 1. The paper's Eq. 2 is the same v.
``draw_stages`` draws all of that as one categorical over the original-data draw and the
chains, each chain's share ``P(S_j) / |S_j|``.

**The ops.** The vocabulary is vocab ``greedy_aa`` (29 ops, the TF AutoAugment CIFAR
code's, the signed ones split). The search runs over ``SEARCH_OPS`` (25): the 5 signed
geometric ops split into ``_pos`` / ``_neg`` (the user's choice, 2026-11-07: a direction is
its own search op), the 4 enhance ops whole, their direction the level (the factor 0.1 +
0.18 L is below 1 at L <= 4, 1 at L = 5, above after), and the 11 others. A level maps to
a vocab op and its magnitude by the TF code's formulas (``level_to_draw``).

**base_probs and q.** ``base_probs()`` is each op's share of the expected applications,
sum over chains of the chain's share times its count of the op. A q tilts each chain by
``prod over its steps of q(a) / base(a)`` (AutoAugment's tilt, every step at p = 1),
rescaled so the chains hold their total mass: the original-data share is held, as
``none_prob`` is for the other hosts. q = base is the searched policy exactly.

**The static prune** drops every chain that applies a pruned op; its share goes to the
other chains by share (``hold_none_prob``, the default), or to the original-data draw
(False). An op no chain applies after that is pruned by implication. A policy searched
over a subset never applies the ops it left out, which its meta lists (``pruned_ops``).

**``policy_file``.** A search's ``final.json`` / ``level_<c>.json``; ``initial``
(``INITIAL``, train.sbatch ``GAA_POLICY=initial``) is the search's starting point, no
augmentation: the trunk of the search-then-train recipe trains on it (the recipe's own
pipeline only), is probed at 25%, and its prune decides the ops the search runs over.
It draws nothing; its ``base_probs()`` is the level-1 share of the (op, level) pairs, the
selector's prior. ``null`` is the same, probe-only (``check_trainable`` refuses it).
``chain`` (a list of ``[search op, level]``) is one search candidate: every image gets the
whole chain (the search's p = 1), for a child network's training.

Pure numpy, like every host.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import AugHost
from .sensaug import NONE_KEY

__all__ = [
    "FAMILY",
    "GreedyAutoAugmentHost",
    "INITIAL",
    "NUM_LEVELS",
    "POLICY_FORMAT",
    "SEARCH_OPS",
    "VOCAB_NAMES",
    "candidates",
    "column_of",
    "level_to_draw",
    "load_policy",
    "save_policy",
    "set_probs",
]

FAMILY = "greedy_aa"

#: ``policy_file`` naming the search's starting point (no augmentation) as one to train on.
INITIAL = "initial"

POLICY_FORMAT = "dropaug.greedy_autoaugment.policy"
POLICY_VERSION = 1

#: The "leave this image alone" draw (``tasks.augment.NO_OP``).
NO_OP = NONE_KEY[0]

#: Levels 0 .. NUM_LEVELS - 1 (the paper's m_n = 10; the TF code's PARAMETER_MAX is 10).
NUM_LEVELS = 10
PARAMETER_MAX = 10.0

#: The TF code's images are 32 x 32; its pixel ranges become fractions of the image.
IMAGE_SIZE = 32.0

_SIGNED = ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY")
_ENHANCE = ("Color", "Contrast", "Brightness", "Sharpness")
_NO_MAGNITUDE = ("FlipLR", "FlipUD", "AutoContrast", "Equalize", "Invert", "Blur", "Smooth")

#: The TF code's ``ALL_TRANSFORMS`` order, as vocab ``greedy_aa`` lists it.
_ORDER = (
    "FlipLR", "FlipUD", "AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "CropBilinear",
    "Solarize", "Color", "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateX",
    "TranslateY", "Cutout", "Blur", "Smooth",
)


def _split(transform_id: str) -> Tuple[str, str]:
    return f"{transform_id}_pos", f"{transform_id}_neg"


#: vocab ``greedy_aa``'s 29 op names, in its order (hardcoded so the host stays free of
#: torch; ``check`` compares them with the vocab).
VOCAB_NAMES: Tuple[str, ...] = tuple(
    f"{FAMILY}.{name}"
    for t in _ORDER
    for name in (_split(t) if t in _SIGNED or t in _ENHANCE else (t,))
)

#: The search's 25 ops: the signed geometric ones split, the enhance ones whole.
SEARCH_OPS: Tuple[str, ...] = tuple(name for t in _ORDER for name in (_split(t) if t in _SIGNED else (t,)))


def _int_parameter(level: int, maxval: float) -> int:
    return int(level * maxval / PARAMETER_MAX)


def _float_parameter(level: int, maxval: float) -> float:
    return float(level) * maxval / PARAMETER_MAX


def level_to_draw(search_op: str, level: int) -> Tuple[str, float]:
    """``(vocab op, magnitude in its units)`` for one search step: the TF code's level
    formulas (``augmentation_transforms.py``), its 32 px ones as fractions of the image.
    An enhance op's level picks its direction: ``_neg`` at 1 - v below the identity, ``_pos``
    at v - 1 from it (L = 5, v = 1, is ``_pos`` at 0)."""
    level = int(level)
    if not 0 <= level < NUM_LEVELS:
        raise ValueError(f"level {level} is not in 0..{NUM_LEVELS - 1}")
    if search_op not in SEARCH_OPS:
        raise ValueError(f"{search_op!r} is not a Greedy AutoAugment search op ({list(SEARCH_OPS)[:3]}...)")
    base = search_op.rsplit("_", 1)[0] if search_op.endswith(("_pos", "_neg")) else search_op
    name = f"{FAMILY}.{search_op}"
    if base in _NO_MAGNITUDE:
        return name, 0.0
    if base == "Rotate":
        return name, float(_int_parameter(level, 30))
    if base in ("ShearX", "ShearY"):
        return name, _float_parameter(level, 0.3)
    if base in ("TranslateX", "TranslateY"):
        return name, _int_parameter(level, 10) / IMAGE_SIZE
    if base == "Posterize":
        return name, float(4 - _int_parameter(level, 4))
    if base == "Solarize":
        return name, float(256 - _int_parameter(level, 256))
    if base == "CropBilinear":
        # _crop_impl takes the raw level as pixels.
        return name, level / IMAGE_SIZE
    if base == "Cutout":
        return name, _int_parameter(level, 20) / IMAGE_SIZE
    if base in _ENHANCE:
        v = _float_parameter(level, 1.8) + 0.1
        direction = "neg" if level < NUM_LEVELS // 2 else "pos"
        return f"{FAMILY}.{base}_{direction}", round(abs(v - 1.0), 12)
    raise AssertionError(search_op)


def column_of(search_op: str, level: int) -> str:
    """The vocab op a search step applies (``level_to_draw``'s name)."""
    return level_to_draw(search_op, level)[0]


def candidates(pruned: Iterable[str] = ()) -> List[Tuple[str, int]]:
    """Every ``(search op, level)`` pair whose vocab op is not in ``pruned``, in
    ``SEARCH_OPS`` then level order: one greedy level's trials (250 with nothing pruned)."""
    out = set(pruned)
    return [(op, level) for op in SEARCH_OPS for level in range(NUM_LEVELS) if column_of(op, level) not in out]


def set_probs(num_sets: int, alpha: float = 2.0) -> Tuple[float, np.ndarray]:
    """``(P(original), P(S_1 .. S_n))`` of ``func_gautoaug``'s scan over n sets (module
    docstring). They sum to 1."""
    if num_sets < 0:
        raise ValueError(f"num_sets must be >= 0, got {num_sets}")
    v = np.array([1.0] + [(1.0 / (i + 1)) ** float(alpha) for i in range(1, num_sets + 1)])
    probs = v - np.append(v[1:], 0.0)
    return float(probs[0]), probs[1:]


# --- the policy file ---------------------------------------------------------------------


def _check_chain(chain, where: str) -> Tuple[Tuple[str, int], ...]:
    steps = []
    for step in chain:
        if len(step) != 2:
            raise ValueError(f"{where}: a chain step is [search op, level], got {step!r}")
        op, level = str(step[0]), int(step[1])
        level_to_draw(op, level)  # raises on an unknown op or level
        steps.append((op, level))
    if not steps:
        raise ValueError(f"{where}: an empty chain")
    return tuple(steps)


def save_policy(path: str, sub_policies: Sequence[dict], *, set_size: int, alpha: float, meta: dict) -> None:
    """Write a policy file atomically: ``sub_policies`` ranked best first, each
    ``{"chain": [[search op, level], ...], "score": ..., ...}``."""
    record = {
        "format": POLICY_FORMAT,
        "version": POLICY_VERSION,
        "ops": list(VOCAB_NAMES),
        "search_ops": list(SEARCH_OPS),
        "set_size": int(set_size),
        "alpha": float(alpha),
        "sub_policies": [
            {**sp, "chain": [list(step) for step in _check_chain(sp["chain"], path)]} for sp in sub_policies
        ],
        "meta": dict(meta),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, path)


def load_policy(path: str) -> dict:
    """A policy file, checked: format, version, the ops against vocab ``greedy_aa``, every
    chain's ops and levels, ``set_size`` >= 1 and ``alpha`` > 0."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no Greedy AutoAugment policy at {path!r} (scripts/gaa_search.py writes one)")
    with open(path) as f:
        policy = json.load(f)
    if policy.get("format") != POLICY_FORMAT:
        raise ValueError(f"{path}: format {policy.get('format')!r}, expected {POLICY_FORMAT!r}")
    if int(policy.get("version", -1)) != POLICY_VERSION:
        raise ValueError(f"{path}: policy version {policy.get('version')}, this code reads {POLICY_VERSION}")
    if tuple(policy.get("ops", ())) != VOCAB_NAMES:
        raise ValueError(f"{path}: its ops are not vocab greedy_aa's")
    chains = [_check_chain(sp["chain"], path) for sp in policy.get("sub_policies", [])]
    if not chains:
        raise ValueError(f"{path}: no sub-policies")
    set_size, alpha = int(policy.get("set_size", 0)), float(policy.get("alpha", 0))
    if set_size < 1 or not alpha > 0:
        raise ValueError(f"{path}: set_size {set_size} must be >= 1 and alpha {alpha} > 0")
    return {"chains": chains, "set_size": set_size, "alpha": alpha, "meta": dict(policy.get("meta") or {})}


# --- the host ------------------------------------------------------------------------------


class GreedyAutoAugmentHost(AugHost):
    """Greedy AutoAugment's searched policy over vocab ``greedy_aa`` (module docstring)."""

    name = "greedy_autoaugment"
    family = FAMILY

    def __init__(
        self,
        *,
        policy_file: Optional[str] = None,
        chain: Optional[Sequence[Sequence]] = None,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        if chain is not None and policy_file is not None:
            raise ValueError("give a policy_file or a chain (one search candidate), not both")
        self.policy_file = None if policy_file is None else str(policy_file)
        self.policy_meta: dict = {}
        self._index = {name: a for a, name in enumerate(VOCAB_NAMES)}
        if chain is not None:
            # One search candidate: every image, the whole chain.
            self.mode = "chain"
            chains = [_check_chain(chain, "chain")]
            none_mass, masses = 0.0, np.ones(1)
        elif self.policy_file in (None, INITIAL):
            self.mode = "initial"
            chains, none_mass, masses = [], 1.0, np.zeros(0)
        else:
            self.mode = "policy"
            policy = load_policy(self.policy_file)
            self.policy_meta = policy["meta"]
            chains = list(policy["chains"])
            sets = [range(s, min(s + policy["set_size"], len(chains))) for s in range(0, len(chains), policy["set_size"])]
            none_mass, set_mass = set_probs(len(sets), policy["alpha"])
            masses = np.zeros(len(chains))
            for j, members in enumerate(sets):
                masses[list(members)] = set_mass[j] / len(members)
        # May be trained on (tasks.augment.check_trainable): a bare null is probe-only.
        self.searched = not (self.mode == "initial" and self.policy_file is None)
        self.chain_steps = [tuple(level_to_draw(op, level) for op, level in c) for c in chains]
        self.num_stages = max((len(c) for c in self.chain_steps), default=1)
        self._mass = masses.astype(np.float64)
        self._mass_base = self._mass.copy()
        self._mass_now = self._mass.copy()
        self._held = float(self._mass.sum())

        if self.mode == "initial":
            # The selector's prior: each op's share of the level-1 (op, level) pairs.
            counts = np.zeros(len(VOCAB_NAMES))
            for op, level in candidates():
                counts[self._index[column_of(op, level)]] += 1.0
            base = counts
            none_prob = 1.0 - 1e-12
        else:
            base = self._applications(self._mass)
            none_prob = float(np.clip(none_mass, 0.0, 1.0 - 1e-12))
        super().__init__(
            VOCAB_NAMES,
            base,
            none_prob=none_prob,
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    # --- shares ---------------------------------------------------------------------------

    def _counts(self) -> np.ndarray:
        """(chains, A): how many times each chain applies each op."""
        counts = np.zeros((len(self.chain_steps), len(VOCAB_NAMES)))
        for c, steps in enumerate(self.chain_steps):
            for name, _ in steps:
                counts[c, self._index[name]] += 1.0
        return counts

    def _applications(self, masses: np.ndarray) -> np.ndarray:
        total = masses @ self._counts() if len(masses) else np.zeros(len(VOCAB_NAMES))
        if total.sum() <= 0:
            raise ValueError(f"{self.policy_file or 'the policy'} never applies any op")
        return total

    def chain_probs(self) -> np.ndarray:
        """Each chain's probability of being drawn now (q and the prune included); the
        original-data draw is 1 minus their sum."""
        return self._mass_now.copy()

    # --- q and the prune --------------------------------------------------------------------

    def _tilt(self, masses: np.ndarray, ratio: np.ndarray) -> np.ndarray:
        """Each chain's mass times the product of ``ratio`` over its steps, rescaled to the
        held total."""
        if not len(masses):
            return masses
        weights = masses * np.prod(ratio[None, :] ** self._counts(), axis=1)
        total = float(weights.sum())
        if total <= 0:
            raise ValueError(f"{type(self).__name__}: q drops every chain; nothing would be applied")
        return weights * (float(masses.sum()) / total)

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        if self.mode == "initial":
            return q
        base = self._base
        ratio = np.divide(q, base, out=np.zeros_like(q), where=base > 0)
        self._mass_now = self._tilt(self._mass_base, ratio)
        realized = self._applications(self._mass_now)
        return realized / realized.sum()

    def prune(self, names: Iterable[str], *, hold_none_prob: bool = True) -> Tuple[str, ...]:
        """``AugHost.prune``: the chains that apply a pruned op are dropped (module
        docstring). ``hold_none_prob`` gives their share to the other chains; False to the
        original-data draw."""
        self._hold_mass = bool(hold_none_prob)
        implied = super().prune(names, hold_none_prob=True)
        if self.mode != "initial" and self._pruned and not self._hold_mass:
            self._none_prob = float(np.clip(1.0 - self._mass_base.sum(), 0.0, 1.0 - 1e-12))
        return implied

    def _prune_base(self, keep: np.ndarray) -> np.ndarray:
        if self.mode == "initial":
            return super()._prune_base(keep)
        applies_pruned = (self._counts()[:, ~keep] > 0).any(axis=1)
        masses = np.where(applies_pruned, 0.0, self._mass)
        if masses.sum() <= 0:
            raise ValueError(f"{type(self).__name__}: the prune drops every chain of the policy")
        if getattr(self, "_hold_mass", True):
            masses = masses * (self._held / masses.sum())
        self._mass_base = masses
        self._mass_now = masses.copy()
        total = self._applications(masses)
        return total / total.sum()

    # --- the draws -------------------------------------------------------------------------

    def draw_stages(self, batch_size: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
        """The method's draw for one batch: ``num_stages`` stages of ``(names, magnitudes)``,
        stage k holding each image's k-th step (``NO_OP``, magnitude 0, past its chain's
        end, and in every stage for an image drawn as it is). ``initial`` draws nothing; a
        ``chain`` host gives every image its chain."""
        names = np.full((self.num_stages, batch_size), NO_OP, dtype=object)
        magnitudes = np.zeros((self.num_stages, batch_size))
        if self.mode != "initial" and len(self.chain_steps):
            p = np.append(self._mass_now, max(0.0, 1.0 - float(self._mass_now.sum())))
            picks = rng.choice(len(p), size=batch_size, p=p / p.sum())
            for i, c in enumerate(picks):
                if c == len(self.chain_steps):
                    continue
                for k, (name, magnitude) in enumerate(self.chain_steps[c]):
                    names[k, i] = name
                    magnitudes[k, i] = magnitude
        return [(names[k], magnitudes[k]) for k in range(self.num_stages)]
