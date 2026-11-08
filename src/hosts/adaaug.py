"""AdaAug (Cheung & Yeung, ICLR 2022) as a host: the numpy half of its policy.

"AdaAug: Learning Class- and Instance-adaptive Data Augmentation Policies", code
github.com/jamestszhim/adaptive_augment (MIT). A policy network h reads each image's
feature from a model gf and gives, per image, a probability p over 17 outputs (16 ops and
Identity) and a level λ in [0, 1] for each (``AdaAug.predict_aug_params``). The network
and its search are torch and live in ``src/adaaug/``; the hook that runs it in training is
``tasks/mmseg/adaaug_hooks.py``. This host holds the sampling (``AdaAug.exploit``,
``get_training_aug_images``):

- ``k_ops`` outputs drawn from the image's p without replacement (``torch.multinomial``
  there), applied in the order drawn; Identity applies nothing
- each drawn op's λ perturbed by up to ``delta`` (``perturb_param``: an amount uniform on
  [0, delta], down or up with p = 0.5, clipped to [0, 1]), then applied at
  ``λ (high - low) + low`` in ``AUGMENT_LIST``'s range, which MADAug's code keeps as it is
  (``hosts.madaug.RANGES``); the five geometric ops' value is negated with p = 0.5
  (``random_mirror``)

That value is put on the vocab's split ops (``to_columns``, vocab ``adaaug``): a geometric
op at ``|v|`` on ``_pos`` or ``_neg`` (the mirror coin picks the direction whatever λ's
side of 0.5, since a fair coin times the sign is a fair coin), an enhance op at
``|f - 1|`` on the side its factor f falls (no mirror: below λ = 0.5 it is ``_neg``),
Solarize, Posterize and Cutout at their value, the others with no magnitude.

**Two settings.** ``policy_file: initial`` is the search's starting policy (the trunk of
the search-then-train recipe): an untrained h, never updated, reading the live model's
features, drawn as the search's exploit draws (``search.py`` hard-codes k = 1, δ = 0,
T = 1, ``SEARCH_EXPLOIT``). A search's directory (``scripts/adaaug_search.py``) is the
searched policy, h and gf frozen, read from ``final/``, drawn as ``train.py`` draws
(``k_ops``, ``delta``, ``temperature``; ``train.sh`` on reduced CIFAR-10: 2, 0.3, 3). Its
outputs are the ones the search kept (``final/meta.json``), so p and λ are that wide.

**The prune.** A search over the ops left by a prune (``--prune-ops`` / ``--random-prune``)
drops an output from h when every vocab op it maps to is pruned (``searched_outputs``):
the policy is smaller and the explore makes fewer copies. Identity and Flip are never
dropped (Flip is out of probing and pruning). An op with one direction pruned keeps its
output, and its pruned direction is mirrored onto the kept one at the same magnitude, in
the search's explore and in training alike (a choice of mine, flagged in CLAUDE.md): a
geometric op's coin always lands on the kept side, an enhance op's level on the pruned
side applies the kept direction at ``|f - 1|``. ``tasks.augment``'s guard checks that no
pruned op is ever drawn.

**base_probs and q.** AdaAug's p is per image and moves with the model (in the trunk) or
with the image (frozen). ``observe(p, levels)`` keeps an exponential average of each vocab
op's expected share over training batches (``prior_momentum``; an output's p split
between its two directions by the coin, or by each image's level for an enhance op), and
``base_probs()`` is that, renormalized (uniform before the first batch). A q is held as
the ratio ``q / base`` and tilts each image's p over the outputs, ``Σ q / Σ base`` over an
output's live columns, renormalized to ``1 - p(Identity)``; a signed op's direction is
then drawn in proportion to its two ratios (the coin, under q = base). p(Identity) is
never moved by q. ``none_prob`` is the running mean p(Identity).

Pure numpy, like every host.
"""

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import AugHost
from .madaug import IDENTITY, NO_OP, OPS_NAMES, RANGES, SIGNED, Slot, perturb_level

__all__ = [
    "AdaAugHost",
    "ENHANCE",
    "INITIAL",
    "OPS_NAMES",
    "SEARCH_EXPLOIT",
    "VOCAB_NAMES",
    "columns_of",
    "load_meta",
    "searched_outputs",
    "to_columns",
]

#: The vocab family the host's ops are in (vocab/families/adaaug.py).
FAMILY = "adaaug"

#: The enhance ops: their direction follows the level (no ``random_mirror``).
ENHANCE = frozenset({"Contrast", "Color", "Brightness", "Sharpness"})

#: ``policy_file`` for the search's starting policy (module docstring).
INITIAL = "initial"

#: The search's exploit draw, hard-coded in AdaAug's ``search.py`` (``adaaug_config``).
SEARCH_EXPLOIT = {"k_ops": 1, "delta": 0.0, "temperature": 1.0}

#: Where a search's result is, under its directory (``src/adaaug/search.py``).
FINAL_DIR = "final"
META_FILE = "meta.json"


def columns_of(op: str) -> Tuple[str, ...]:
    """The vocab ops AdaAug output ``op`` maps to: ``(_pos, _neg)`` for a signed or
    enhance op, one op otherwise, none for Identity."""
    if op == IDENTITY:
        return ()
    if op in SIGNED or op in ENHANCE:
        return (f"{FAMILY}.{op}_pos", f"{FAMILY}.{op}_neg")
    return (f"{FAMILY}.{op}",)


#: The host's vocabulary: every output's columns, in AdaAug's order (vocab adaaug's).
VOCAB_NAMES: Tuple[str, ...] = tuple(c for op in OPS_NAMES for c in columns_of(op))


def searched_outputs(pruned: Iterable[str] = ()) -> Tuple[str, ...]:
    """The outputs a search over the ops left by ``pruned`` keeps, in AdaAug's order: each
    with a column left, Identity always (module docstring)."""
    pruned = set(pruned)
    unknown = pruned - set(VOCAB_NAMES)
    if unknown:
        raise KeyError(f"{sorted(unknown)} are not adaaug ops")
    return tuple(op for op in OPS_NAMES if op == IDENTITY or any(c not in pruned for c in columns_of(op)))


def to_columns(op: str, level, positive, live=None) -> Tuple[List[str], np.ndarray]:
    """Per image, output ``op`` at ``level`` (B,) in [0, 1] as ``(vocab op, magnitude)``:
    ``operation.py``'s ``level * (high - low) + low``, put on the split ops (module
    docstring). ``positive`` (B,) bool is a geometric op's direction after the mirror
    coin (ignored for the others). ``live``, a set of vocab ops, mirrors a direction not
    in it onto the other (``None``: every op live). Identity gives ``NO_OP``."""
    level = np.asarray(level, dtype=np.float64).reshape(-1)
    n = level.shape[0]
    if op == IDENTITY:
        return [NO_OP] * n, np.zeros(n)
    cols = columns_of(op)
    if RANGES[op] is None:
        return [cols[0]] * n, np.zeros(n)
    low, high = RANGES[op]
    value = low + level * (high - low)
    if len(cols) == 1:
        return [cols[0]] * n, value
    if op in SIGNED:
        pos = np.asarray(positive, dtype=bool).reshape(-1)
        magnitude = np.abs(value)
    else:  # an enhance factor: _pos above 1, _neg below
        pos = value >= 1.0
        magnitude = np.abs(value - 1.0)
    if live is not None:
        alive = (cols[0] in live, cols[1] in live)
        if not any(alive):
            raise ValueError(f"{op}: both directions are pruned; the output should not be drawn")
        if not alive[0]:
            pos = np.zeros(n, dtype=bool)
        elif not alive[1]:
            pos = np.ones(n, dtype=bool)
    return [cols[0] if p else cols[1] for p in pos], magnitude


def load_meta(policy_file: str) -> dict:
    """A search's ``final/meta.json`` (``src/adaaug/search.py``): its outputs, the ops it
    left out, and what built it. ``policy_file`` is the search's directory (or ``final/``
    itself)."""
    directory = os.path.abspath(str(policy_file))
    if os.path.basename(directory.rstrip(os.sep)) != FINAL_DIR:
        directory = os.path.join(directory, FINAL_DIR)
    path = os.path.join(directory, META_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no AdaAug search at {policy_file}: {path} is missing (scripts/adaaug_search.py writes it "
            f"when the search finishes; policy_file: {INITIAL} trains on the starting policy)"
        )
    with open(path) as f:
        meta = json.load(f)
    outputs = tuple(meta.get("outputs") or ())
    if not outputs or IDENTITY not in outputs or any(op not in OPS_NAMES for op in outputs):
        raise ValueError(f"{path}: outputs {list(outputs)} are not a subset of AdaAug's with Identity")
    if list(outputs) != [op for op in OPS_NAMES if op in outputs]:
        raise ValueError(f"{path}: outputs {list(outputs)} are not in AdaAug's order")
    expected = searched_outputs(meta.get("pruned_ops") or ())
    if outputs != expected:
        raise ValueError(f"{path}: outputs {list(outputs)} are not the ones its pruned_ops leave ({list(expected)})")
    meta["directory"] = directory
    return meta


class AdaAugHost(AugHost):
    """AdaAug's sampling over its 16 ops, split into 25 vocab ops (module docstring)."""

    name = "adaaug"

    #: AdaAug's full policy width: the 16 ops and Identity.
    num_policy_outputs = len(OPS_NAMES)

    def __init__(
        self,
        *,
        policy_file: Optional[str] = INITIAL,
        k_ops: int = 2,
        delta: float = 0.3,
        temperature: float = 3.0,
        prior_momentum: float = 0.99,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        if policy_file is None:
            raise ValueError(f"host adaaug: policy_file is '{INITIAL}' (the starting policy) or a search's directory")
        if not 0.0 <= float(prior_momentum) < 1.0:
            raise ValueError(f"prior_momentum must be in [0, 1), got {prior_momentum}")
        self.policy_file = str(policy_file)
        self.searched = self.policy_file != INITIAL
        if self.searched:
            self.policy_meta: Optional[dict] = load_meta(self.policy_file)
            # config.check reads the ops a search left out as one space-separated string.
            self.policy_meta["pruned_ops"] = " ".join(self.policy_meta.get("pruned_ops") or ())
            self._outputs = tuple(self.policy_meta["outputs"])
            settings = {"k_ops": k_ops, "delta": delta, "temperature": temperature}
        else:
            self.policy_meta = None
            self._outputs = OPS_NAMES
            settings = dict(SEARCH_EXPLOIT)
        if not 1 <= int(settings["k_ops"]) <= len(self._outputs):
            raise ValueError(f"k_ops must be in [1, {len(self._outputs)}], got {settings['k_ops']}")
        if not 0.0 <= float(settings["delta"]) <= 1.0:
            raise ValueError(f"delta must be in [0, 1], got {settings['delta']}")
        if float(settings["temperature"]) <= 0:
            raise ValueError(f"temperature must be positive, got {settings['temperature']}")
        self.k_ops = int(settings["k_ops"])
        self.delta = float(settings["delta"])
        self.temperature = float(settings["temperature"])
        self.prior_momentum = float(prior_momentum)
        super().__init__(
            VOCAB_NAMES,
            np.full(len(VOCAB_NAMES), 1.0 / len(VOCAB_NAMES)),
            none_prob=0.0,
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )
        self._index = {n: i for i, n in enumerate(self._names)}
        self._tilt = np.ones(len(self._names))
        self._shares: Optional[np.ndarray] = None
        self._identity: Optional[float] = None

    # --- the policy's shape ------------------------------------------------------------

    @property
    def outputs(self) -> Tuple[str, ...]:
        """The policy's outputs, in AdaAug's order: all 17 for the starting policy, the
        search's kept ones for a searched policy. p and levels are this wide."""
        return self._outputs

    @property
    def settings(self) -> Dict[str, float]:
        """The draw's ``k_ops``, ``delta`` and ``temperature`` (module docstring)."""
        return {"k_ops": self.k_ops, "delta": self.delta, "temperature": self.temperature}

    # --- the prior and q -----------------------------------------------------------

    @property
    def none_prob(self) -> float:
        """The policy's running mean p(Identity), 1 / (its outputs) before the first
        batch. Never moved by q."""
        return 1.0 / len(self._outputs) if self._identity is None else self._identity

    def observe(self, p, levels) -> None:
        """Fold one training batch's policy outputs (p and levels, (B, len(outputs)))
        into the running prior ``base_probs()``."""
        p, levels = self._check(p, levels)
        shares = np.zeros(len(self._names))
        for j, op in enumerate(self._outputs):
            cols = columns_of(op)
            if not cols:
                continue
            if op in ENHANCE:
                low, high = RANGES[op]
                pos = low + levels[:, j] * (high - low) >= 1.0
                shares[self._index[cols[0]]] += float((p[:, j] * pos).mean())
                shares[self._index[cols[1]]] += float((p[:, j] * ~pos).mean())
            else:
                for c in cols:
                    shares[self._index[c]] += float(p[:, j].mean()) / len(cols)
        identity = float(p[:, self._outputs.index(IDENTITY)].mean())
        m = self.prior_momentum
        if self._shares is None:
            self._shares, self._identity = shares, identity
        else:
            self._shares = m * self._shares + (1.0 - m) * shares
            self._identity = m * self._identity + (1.0 - m) * identity
        # An unpruned direction stays drawable (e.g. an enhance op whose level never fell on
        # one side yet), so floor its share above zero; pruned ones stay at 0.
        floor = 1e-12
        base = np.where([n in self._pruned for n in self._names], 0.0, np.maximum(self._shares, floor))
        if base.sum() > 0:
            self._base = base / base.sum()

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        """Hold q as its ratio to the base it was derived from."""
        self._tilt = np.divide(q, self._base, out=np.zeros_like(q), where=self._base > 0)
        return q

    def clear_weights(self) -> None:
        """Back to the policy's own p (a static prune still holds)."""
        self._tilt = np.array([0.0 if n in self._pruned else 1.0 for n in self._names])

    def sampling_probs(self) -> np.ndarray:
        """The prior times q's ratio, renormalized: the mean of what is drawn."""
        probs = self._base * self._tilt
        total = probs.sum()
        return probs / total if total > 0 else self._base.copy()

    def prune(self, names: Iterable[str], *, hold_none_prob: bool = True) -> Tuple[str, ...]:
        """``AugHost.prune``, with the pruned ops' ratio set to 0. The no-op mass is each
        image's p(Identity), the policy's, so ``hold_none_prob=False`` has nothing to
        renormalize and raises. The starting policy's outputs become the ones the prune
        leaves (``searched_outputs``: the search's h); a searched policy keeps its own,
        and an output whose ops the run prunes besides is given 0 (``tilted``)."""
        names = tuple(names)
        if names and not hold_none_prob:
            raise ValueError(
                f"{type(self).__name__}: the no-op is the policy's per-image p(Identity), "
                f"so hold_none_prob=False has nothing to renormalize"
            )
        implied = super().prune(names, hold_none_prob=True)
        self.clear_weights()
        if not self.searched:
            # The starting policy over what the prune leaves: the search's own h.
            self._outputs = searched_outputs(self._pruned)
        return implied

    def _live(self) -> set:
        """The vocab ops a draw may land on now: not pruned, and q gives them weight."""
        return {n for n, t in zip(self._names, self._tilt) if t > 0 and n not in self._pruned}

    def _output_ratio(self, op: str, live: set) -> float:
        """q's ratio for output ``op``: Σ q / Σ base over its live columns (1 under q =
        base; 0 when none is live)."""
        cols = [c for c in columns_of(op) if c in live]
        if not cols:
            return 0.0
        idx = [self._index[c] for c in cols]
        base = float(self._base[idx].sum())
        if base <= 0:
            return float(np.mean(self._tilt[idx]))
        return float((self._base[idx] * self._tilt[idx]).sum()) / base

    # --- the draws -------------------------------------------------------------------

    def _check(self, p, levels) -> Tuple[np.ndarray, np.ndarray]:
        p = np.asarray(p, dtype=np.float64)
        levels = np.asarray(levels, dtype=np.float64)
        width = len(self._outputs)
        if p.ndim != 2 or p.shape[1] != width or levels.shape != p.shape:
            raise ValueError(f"p {p.shape} and levels {levels.shape} must both be (B, {width}), the policy's outputs")
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(levels))):
            raise ValueError("the policy's p and levels must be finite")
        return p, levels

    def tilted(self, p) -> np.ndarray:
        """Per image (rows of ``p``, the policy's outputs), p with q's ratio applied to
        the ops and p(Identity) held: the distribution ``draw`` samples."""
        p = np.asarray(p, dtype=np.float64)
        live = self._live()
        ident = self._outputs.index(IDENTITY)
        ratio = np.array([1.0 if op == IDENTITY else self._output_ratio(op, live) for op in self._outputs])
        ops = p * ratio
        ops[:, ident] = 0.0
        total = ops.sum(axis=1, keepdims=True)
        keep = 1.0 - p[:, ident : ident + 1]
        ops = np.divide(ops * keep, total, out=np.zeros_like(ops), where=total > 0)
        ops[:, ident] = np.where(total[:, 0] > 0, p[:, ident], 1.0)
        return ops / ops.sum(axis=1, keepdims=True)

    def direction(self, op: str, size: int, rng: np.random.Generator, live: Optional[set] = None) -> np.ndarray:
        """A geometric op's direction after ``random_mirror``, per image: ``_pos`` with
        probability ratio(_pos) / (ratio(_pos) + ratio(_neg)), 1/2 under q = base."""
        live = self._live() if live is None else live
        pos_col, neg_col = columns_of(op)
        t_pos = float(self._tilt[self._index[pos_col]]) if pos_col in live else 0.0
        t_neg = float(self._tilt[self._index[neg_col]]) if neg_col in live else 0.0
        if t_pos + t_neg <= 0:
            raise ValueError(f"{op}: both directions are pruned; the output should not be drawn")
        return rng.random(size) < t_pos / (t_pos + t_neg)

    def draw(self, p, levels, rng: np.random.Generator) -> List[Slot]:
        """One batch's draw: ``k_ops`` slots, slot k holding each image's k-th
        ``(vocab op, magnitude)``, ``NO_OP`` where the image drew Identity (or has fewer
        than k drawable outputs). ``p`` and ``levels`` are the policy's (B, len(outputs))
        outputs for the batch, ``p`` at the host's temperature."""
        p, levels = self._check(p, levels)
        p = self.tilted(p)
        live = self._live()
        batch_size = p.shape[0]
        slots: List[Slot] = [[(NO_OP, 0.0)] * batch_size for _ in range(self.k_ops)]
        for i in range(batch_size):
            k = min(self.k_ops, int(np.count_nonzero(p[i])))
            picks = rng.choice(len(self._outputs), size=k, replace=False, p=p[i])
            for slot, j in enumerate(picks):
                op = self._outputs[int(j)]
                if op == IDENTITY:
                    continue
                level = perturb_level(levels[i, j : j + 1], self.delta, rng)
                positive = self.direction(op, 1, rng, live) if op in SIGNED else None
                names, magnitudes = to_columns(op, level, positive, live)
                slots[slot][i] = (names[0], float(magnitudes[0]))
        return slots
