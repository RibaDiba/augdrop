"""Faster AutoAugment (Hataya et al., ECCV 2020) as a host: the numpy half of its policy.

Faster AutoAugment searches a policy (``search.py``, torch, run before training by
``scripts/faa_search.py``) and then trains on it. A policy is L sub-policies (10) of K
stages (2). Stage s holds, for every op a of the vocabulary, a weight ``w_s(a)`` (its
softmax at temperature 0.05; the weights sum to 1), a probability ``p_s(a)`` and a
magnitude ``mu_s(a)``. An image draws one sub-policy uniformly, then runs its stages in
turn. In each stage it draws op a with probability ``w_s(a)`` and applies it, at
``mu_s(a)``, with probability ``p_s(a)`` (dda's ``SubPolicyStage`` /
``_Operation.forward`` in eval mode). So stage s applies op a with probability
``m_s(a) = w_s(a) p_s(a)`` and leaves the image alone with probability
``1 - M_s``, where ``M_s = sum_a m_s(a)``.

dda draws the sub-policy and each stage's op once per chunk of a 16-way split of the
batch, and the Bernoulli per image. At mmseg's batches (at most 16 images a GPU) every
chunk is one image, so this draws all of it per image.

The vocabulary is vocab's ``faster_aa``, all 26 ops in its order. The 9 signed ops are
split into ``_pos`` / ``_neg``, in the search too, so each direction has its own w, p and
mu (a departure from the paper's random sign; vocab/families/faster_aa.py).

**base_probs and q.** ``base_probs()`` is each op's share of the expected applications,
``sum_s m_s(a)`` normalized. A q tilts every stage by ``r = q / base``:

    m'_s(a) = M_s * m_s(a) r(a) / sum_b m_s(b) r(b)

so q changes which op a stage applies and never how often it applies one: each stage's
no-op mass ``1 - M_s`` is held, which is the rule ``none_prob`` keeps for the other hosts.
q = base is the searched policy exactly. A stage whose ops q zeroes entirely applies
nothing. ``sampling_probs()`` is the realized share ``sum_s m'_s(a)`` normalized, which is
q only when every stage has the same shape. ``none_prob`` is the mean of ``1 - M_s`` over
the stages: one draw is one stage.

**The static prune** zeroes the pruned ops in every stage. ``hold_none_prob`` (the
default) keeps each ``M_s``, so the mass goes to the stage's other ops; False leaves the
other ops' ``m_s`` as they were, so the stage applies nothing that much more often.

**The probe's magnitudes** (``draw_magnitudes``): per image and op, one of the op's
searched magnitudes, ``mu_s(a)`` with probability proportional to ``m_s(a)``: the
magnitudes the searched policy applies the op at, weighted by how often it does. The
searched policy's, never the tilted one.

The policy file is ``search.py``'s ``.npz`` (``save_policy`` there, ``load_policy`` here).
Without one (``policy_file=None``) the host is dda's starting policy, every weight equal,
every p and mu at the middle of its range: a stand-in that lets a model be probed with
these ops, never trained on (``dropaug.config.check`` refuses it for training).

Pure numpy, like every host. The torch half (``policy.py``, ``search.py``) is imported
only by the search.
"""

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..base import AugHost
from ..sensaug import NONE_KEY

__all__ = [
    "FAMILY",
    "FasterAutoAugmentHost",
    "POLICY_FORMAT",
    "SCALES",
    "Stage",
    "VOCAB_NAMES",
    "initial_policy",
    "load_policy",
]

FAMILY = "faster_aa"

#: The ``format`` field of a policy file.
POLICY_FORMAT = "dropaug.faster_autoaugment.policy"

#: The policy file's version. Bump it when a field's meaning changes.
POLICY_VERSION = 1

#: The "leave this image alone" draw (``tasks.augment.NO_OP``).
NO_OP = NONE_KEY[0]

#: vocab ``faster_aa``'s ops, in its order, each with dda's ``magnitude_scale``: the op-unit
#: magnitude at mu = 1, so a policy's magnitudes lie in [0, scale] (None for an op with no
#: magnitude). Hardcoded so the host stays free of torch; the search writes the names
#: into every policy file and ``load_policy`` checks them against these
#: (``vocab.families.faster_aa.POLICY_SCALES``).
_SPLIT = (("ShearX", 0.3), ("ShearY", 0.3), ("TranslateX", 0.45), ("TranslateY", 0.45), ("Rotate", 30.0))
_ENHANCE = (("Contrast", 1.0), ("Color", 1.0), ("Brightness", 1.0), ("Sharpness", 1.0))


def _split(pairs):
    return [(f"{FAMILY}.{t}_{d}", hi) for t, hi in pairs for d in ("pos", "neg")]


SCALES: Dict[str, Optional[float]] = dict(
    _split(_SPLIT)
    + [
        (f"{FAMILY}.HorizontalFlip", None),
        (f"{FAMILY}.Solarize", 1.0),
        (f"{FAMILY}.Posterize", 1.0),
        (f"{FAMILY}.Invert", None),
    ]
    + _split(_ENHANCE)
    + [
        (f"{FAMILY}.AutoContrast", None),
        (f"{FAMILY}.Equalize", None),
        (f"{FAMILY}.Cutout", 0.2),
        (f"{FAMILY}.SamplePairing", 1.0),
    ]
)

VOCAB_NAMES: Tuple[str, ...] = tuple(SCALES)

#: Each stage's draw for a batch: ``(names (B,), magnitudes (B,))``, ``NO_OP`` (magnitude
#: 0) where the image is left alone.
Stage = Tuple[np.ndarray, np.ndarray]


def initial_policy(num_sub_policies: int = 10, operation_count: int = 2) -> dict:
    """dda's starting point, made deterministic: dda draws every parameter uniform on
    [0, 1]; this sets each to the middle, 0.5, so every weight is equal and every p and
    mu (on dda's [0, 1] scale) is 0.5."""
    shape = (int(num_sub_policies), int(operation_count), len(VOCAB_NAMES))
    magnitudes = np.full(shape, np.nan)
    for a, name in enumerate(VOCAB_NAMES):
        if SCALES[name] is not None:
            magnitudes[..., a] = 0.5 * SCALES[name]
    return {
        "ops": VOCAB_NAMES,
        "weights": np.full(shape, 1.0 / len(VOCAB_NAMES)),
        "probabilities": np.full(shape, 0.5),
        "magnitudes": magnitudes,
        "meta": {"source": "initial"},
    }


def load_policy(path: str) -> dict:
    """A policy file written by ``search.save_policy``, checked: the format and version,
    the op names against vocab ``faster_aa``, the shapes, each stage's weights a
    distribution, each p in [0, 1], and each magnitude within [0, its op's scale]."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no Faster AutoAugment policy at {path!r} (scripts/faa_search.py writes one)")
    with np.load(path, allow_pickle=False) as data:
        policy = {key: data[key] for key in data.files}
    fmt = str(policy.get("format", ""))
    if fmt != POLICY_FORMAT:
        raise ValueError(f"{path}: format {fmt!r}, expected {POLICY_FORMAT!r}")
    version = int(policy.get("version", -1))
    if version != POLICY_VERSION:
        raise ValueError(f"{path}: policy version {version}, this code reads {POLICY_VERSION}")
    ops = tuple(str(n) for n in policy["ops"])
    if ops != VOCAB_NAMES:
        raise ValueError(f"{path}: its ops {list(ops)[:3]}... are not vocab faster_aa's {list(VOCAB_NAMES)[:3]}...")
    meta = {key[len("meta_"):]: policy[key].item() for key in policy if key.startswith("meta_")}
    return {
        "ops": ops,
        "weights": np.asarray(policy["weights"], dtype=np.float64),
        "probabilities": np.asarray(policy["probabilities"], dtype=np.float64),
        "magnitudes": np.asarray(policy["magnitudes"], dtype=np.float64),
        "meta": meta,
    }


def _check_policy(policy: dict, where: str) -> None:
    weights, probs, mags = policy["weights"], policy["probabilities"], policy["magnitudes"]
    shape = weights.shape
    if len(shape) != 3 or shape[-1] != len(VOCAB_NAMES) or probs.shape != shape or mags.shape != shape:
        raise ValueError(
            f"{where}: weights {weights.shape}, probabilities {probs.shape} and magnitudes {mags.shape} "
            f"must all be (sub-policies, stages, {len(VOCAB_NAMES)})"
        )
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(-1), 1.0, atol=1e-5):
        raise ValueError(f"{where}: each stage's weights must be a distribution over the ops")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError(f"{where}: probabilities must be in [0, 1]")
    for a, name in enumerate(VOCAB_NAMES):
        if SCALES[name] is None:
            continue
        lo, hi = 0.0, SCALES[name]
        m = mags[..., a]
        slack = 1e-6 * (hi - lo)
        if not np.isfinite(m).all() or m.min() < lo - slack or m.max() > hi + slack:
            raise ValueError(f"{where}: {name}'s magnitudes span [{m.min()}, {m.max()}], outside [{lo}, {hi}]")


class FasterAutoAugmentHost(AugHost):
    """Faster AutoAugment's searched policy over vocab ``faster_aa`` (module docstring)."""

    name = "faster_autoaugment"
    family = FAMILY

    def __init__(
        self,
        *,
        policy_file: Optional[str] = None,
        num_sub_policies: int = 10,
        operation_count: int = 2,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        self.policy_file = None if policy_file is None else str(policy_file)
        if self.policy_file is None:
            policy = initial_policy(num_sub_policies, operation_count)
        else:
            policy = load_policy(self.policy_file)
        _check_policy(policy, self.policy_file or "the initial policy")
        self.policy_meta = dict(policy["meta"])
        self.searched = self.policy_file is not None

        weights = policy["weights"]
        self.num_sub_policies, self.operation_count = int(weights.shape[0]), int(weights.shape[1])
        num_ops = len(VOCAB_NAMES)
        # (S, A), S = L * K stages in sub-policy-major order.
        self._applied = (weights * policy["probabilities"]).reshape(-1, num_ops)
        self._magnitudes = np.nan_to_num(policy["magnitudes"].reshape(-1, num_ops), nan=0.0)
        self._hold_stage_mass = True
        # The base the tilt is taken against: the searched masses, or the pruned ones.
        self._stage_base = self._applied.copy()
        self._stage = self._applied.copy()

        total = self._applied.sum(axis=0)
        if total.sum() <= 0:
            raise ValueError(f"{self.policy_file or 'the policy'} never applies any op (every p is 0)")
        super().__init__(
            VOCAB_NAMES,
            total,
            none_prob=float(np.clip(1.0 - self._applied.sum(axis=1).mean(), 0.0, 1.0 - 1e-12)),
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    # --- q and the prune ---------------------------------------------------------------

    @staticmethod
    def _tilt(stages: np.ndarray, ratio: np.ndarray, mass: np.ndarray) -> np.ndarray:
        """Each stage's applied masses times ``ratio``, rescaled to the stage mass ``mass``
        (S,). A stage left with no mass applies nothing."""
        tilted = stages * ratio[None, :]
        sums = tilted.sum(axis=1, keepdims=True)
        scale = np.divide(mass[:, None], sums, out=np.zeros_like(sums), where=sums > 0)
        return tilted * scale

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        base = self._base
        ratio = np.divide(q, base, out=np.zeros_like(q), where=base > 0)
        self._stage = self._tilt(self._stage_base, ratio, self._stage_base.sum(axis=1))
        realized = self._stage.sum(axis=0)
        total = float(realized.sum())
        if total <= 0:
            raise ValueError(f"{type(self).__name__}: q zeroes every op of every stage; nothing would be applied")
        return realized / total

    def prune(self, names: Iterable[str], *, hold_none_prob: bool = True) -> Tuple[str, ...]:
        """``AugHost.prune``: the ops are zeroed in every stage. ``hold_none_prob`` keeps
        each stage's mass ``M_s`` (module docstring); False lets it shrink, and
        ``none_prob`` grows with it. ``AugHost``'s single-draw renormalization of the no-op
        does not apply: the no-op here is per stage."""
        self._hold_stage_mass = bool(hold_none_prob)
        implied = super().prune(names, hold_none_prob=True)
        if self._pruned and not self._hold_stage_mass:
            self._none_prob = float(np.clip(1.0 - self._stage_base.sum(axis=1).mean(), 0.0, 1.0 - 1e-12))
        return implied

    def _prune_base(self, keep: np.ndarray) -> np.ndarray:
        mass = self._applied.sum(axis=1) if self._hold_stage_mass else (self._applied * keep).sum(axis=1)
        stages = self._tilt(self._applied, keep.astype(np.float64), mass)
        total = stages.sum(axis=0)
        if total.sum() <= 0:
            raise ValueError(f"{type(self).__name__}: the prune leaves no op any stage applies")
        self._stage_base = stages
        self._stage = stages.copy()
        return total / total.sum()

    def stage_probs(self) -> np.ndarray:
        """(L, K, A): the probability each stage applies each op now, q and the prune
        included. Each stage's no-op is 1 minus its row sum."""
        return self._stage.reshape(self.num_sub_policies, self.operation_count, -1).copy()

    # --- the draws -----------------------------------------------------------------------

    def draw_stages(self, batch_size: int, rng: np.random.Generator) -> List[Stage]:
        """The method's draw for one batch: K stages of ``(names, magnitudes)``, stage k
        holding each image's op in its k-th stage and that op's searched magnitude in the
        op's units, ``NO_OP`` (0) where the stage leaves the image alone. One sub-policy per
        image, uniform; q and the prune applied."""
        names = np.array(VOCAB_NAMES + (NO_OP,), dtype=object)
        num_ops = len(VOCAB_NAMES)
        rows = rng.integers(self.num_sub_policies, size=batch_size)
        out: List[Stage] = []
        for k in range(self.operation_count):
            picked = np.full(batch_size, NO_OP, dtype=object)
            magnitudes = np.zeros(batch_size)
            for i, row in enumerate(rows):
                s = int(row) * self.operation_count + k
                probs = np.append(self._stage[s], max(0.0, 1.0 - float(self._stage[s].sum())))
                a = int(rng.choice(num_ops + 1, p=probs / probs.sum()))
                picked[i] = names[a]
                if a < num_ops:
                    magnitudes[i] = self._magnitudes[s, a]
            out.append((picked, magnitudes))
        return out

    def draw_magnitudes(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """For the probe: per op with a magnitude, ``(B,)`` magnitudes in the op's units,
        each one of the op's searched magnitudes ``mu_s(a)`` drawn with probability
        proportional to ``m_s(a)`` (the searched policy, not the tilted or pruned one). An
        op the policy never applies gets its stages' mean magnitude."""
        out = {}
        for a, name in enumerate(VOCAB_NAMES):
            if SCALES[name] is None:
                continue
            weights = self._applied[:, a]
            if weights.sum() > 0:
                stages = rng.choice(len(weights), size=batch_size, p=weights / weights.sum())
                out[name] = self._magnitudes[stages, a].copy()
            else:
                out[name] = np.full(batch_size, float(self._magnitudes[:, a].mean()))
        return out
