"""AugHost: the adapter an augmentation method implements so train.py can swap it.

A host is a method's sampling POLICY, not its image code. It answers four
questions for the training loop:

- ``vocabulary()``      which ops the method samples between (the axes of R and q)
- ``sampling_probs()``  how often each of them is drawn right now
- ``set_weights(q)``    install a new per-op distribution, e.g. a redundancy tilt
- ``should_probe(it)``  whether the gradient probe runs after training iteration ``it``

and, for a method with a magnitude policy of its own (the torchvision methods), one
more, for the probe:

- ``draw_bins(B, rng)`` which magnitude bin the method would apply each op at, per image

Applying the ops to images is not the host's job, and neither is the probe. Both
key off ``vocabulary()`` names. Pure numpy, no torch, so every host is testable
without a GPU.

Two conventions every host shares:

- **The no-op draw is not in the vocabulary.** torchvision's ``Identity`` and
  SensAug's ``("none", 0)`` have no magnitude to probe. Their probability,
  ``none_prob``, is held fixed by ``set_weights``, so q changes WHICH op is drawn
  and never HOW OFTEN augmentation happens. This is the rule Sensaug's Lever 3
  holds.
- **q replaces, it does not compound.** Derive q from ``base_probs()`` (the
  method's own distribution), not from ``sampling_probs()``. Otherwise each probe
  tilts the previous tilt.
"""

from collections.abc import Mapping
from typing import ClassVar, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["AugHost", "draw_flips", "resolve_gate"]

Weights = Union[Mapping, Sequence[float], np.ndarray]

#: ``({op: (B,) bin index}, {op: (B,) negate?})``, what ``draw_bins`` returns.
Bins = Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]


def draw_flips(names: Sequence[str], batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Per op and image, whether the method negates a signed op's magnitude: p = 0.5,
    as torchvision's auto-augment methods do. Drawn for every op; an unsigned op
    ignores it."""
    return {name: rng.random(batch_size) < 0.5 for name in names}


def resolve_gate(
    interval: Optional[int], iters: Optional[Iterable[int]]
) -> Tuple[Optional[int], Optional[Tuple[int, ...]]]:
    """Normalize the two ways of saying when the probe fires into ``(interval, iters)``.

    Ported from Sensaug's ``grad_hook.resolve_gate``, with one difference: passing
    neither is allowed and means the host never probes during training (the
    probe can still run post hoc on a checkpoint). An explicit but EMPTY
    ``iters`` still raises. That is what a schedule derivation produces when it
    goes wrong, and it would otherwise surface weeks later as an empty
    ``corr_matrix_log.json``.
    """
    if interval is not None and iters is not None:
        raise ValueError(
            f"probe_interval={interval} and probe_iters={tuple(iters)!r} were both "
            f"given; they are two different clocks and only one can be in effect"
        )
    if iters is not None:
        out = tuple(sorted({int(i) for i in iters}))
        if not out:
            raise ValueError("probe_iters is empty, so the probe would never fire")
        if out[0] < 1:
            raise ValueError(f"probe_iters must be positive iteration counts, got {out[0]}")
        return None, out
    if interval is None:
        return None, None
    if interval < 1:
        raise ValueError(f"probe_interval must be a positive iteration count, got {interval}")
    return int(interval), None


class AugHost:
    """Base host. Subclasses supply the vocabulary, the method's own distribution
    over it, and the no-op probability; everything else is shared.

    ``iteration`` everywhere means the number of training iterations COMPLETED,
    i.e. mmengine's ``runner.iter + 1`` inside ``after_train_iter``. The gate is a
    pure function of it, so every DDP rank agrees on when to probe and the probe's
    gather cannot deadlock.
    """

    #: The key train.py selects this host by. Set on every subclass.
    name: ClassVar[str] = ""

    #: How many magnitude bins ``draw_bins`` indexes, or None for a host with no
    #: magnitude policy of its own (``draw_bins`` then returns None).
    num_magnitude_bins: Optional[int] = None

    def __init__(
        self,
        vocabulary: Sequence[str],
        base_probs: Weights,
        *,
        none_prob: float = 0.0,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        names = tuple(str(n) for n in vocabulary)
        if not names:
            raise ValueError(f"{type(self).__name__}: the vocabulary is empty")
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"{type(self).__name__}: duplicate op names {dupes}")
        if not 0.0 <= float(none_prob) < 1.0:
            raise ValueError(f"none_prob must be in [0, 1), got {none_prob}")

        self._names = names
        self._none_prob = float(none_prob)
        self._base = self._as_distribution(base_probs)
        self._probs = self._base.copy()

        self._probe_interval, iters = resolve_gate(probe_interval, probe_iters)
        self._probe_iters = None if iters is None else frozenset(iters)
        self._control_iters = frozenset(int(i) for i in control_iters)
        stray = sorted(i for i in self._control_iters if not self.should_probe(i))
        if stray:
            raise ValueError(
                f"control_iters {stray} are not iterations the probe fires on. A "
                f"baseline that never fires is a silently missing baseline."
            )

    # --- the policy ------------------------------------------------------------

    def vocabulary(self) -> Tuple[str, ...]:
        """The ops this method samples between, in a fixed order. R's rows and
        columns and every q vector are indexed by this order."""
        return self._names

    @property
    def none_prob(self) -> float:
        """Probability that one draw applies no op. ``set_weights`` never moves it.

        A draw picks op ``a`` with probability
        ``(1 - none_prob) * sampling_probs()[a]``.
        """
        return self._none_prob

    def base_probs(self) -> np.ndarray:
        """The method's own distribution over the vocabulary, before any q.

        This is the prior a redundancy tilt is taken against.
        """
        return self._base.copy()

    def sampling_probs(self) -> np.ndarray:
        """What is actually being drawn now, given that some op is drawn. Sums to 1.

        Equal to the last q for most hosts. A host whose method cannot hit an
        arbitrary per-op distribution (AutoAugment samples whole sub-policies)
        reports the distribution it actually realizes instead.
        """
        return self._probs.copy()

    def set_weights(self, q: Weights) -> None:
        """Install q, either a mapping of op name to weight or a sequence aligned to
        ``vocabulary()``. q is normalized, so only ratios matter. A zero weight
        drops an op.

        A non-finite q raises instead of being coerced. Missing data upstream is
        NaN on purpose, and it has to be resolved into a weight (typically "leave
        it at its base probability") by the caller who knows why it is missing.
        """
        self._probs = self._apply_weights(self._as_distribution(q))

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        """Turn a validated, normalized q into the distribution actually sampled.

        Identity here. Hosts whose sampling unit is not a single op override it.
        """
        return q

    def draw_bins(self, batch_size: int, rng: np.random.Generator) -> Optional[Bins]:
        """The magnitudes the method itself would apply, for the probe: per image, a bin
        index into ``num_magnitude_bins`` for each op, and whether a signed op is
        negated. An op with no magnitude may be listed and is ignored. None for a host
        with no magnitude policy of its own (the probe uses its configured magnitude).

        The method's distribution, not the current q: q changes which op is drawn, and
        the probe's magnitudes must not follow the last q around.
        """
        return None

    # --- the probe schedule ----------------------------------------------------

    def should_probe(self, iteration: int) -> bool:
        """Whether the gradient probe runs after this many completed iterations."""
        if self._probe_iters is not None:
            return iteration in self._probe_iters
        if self._probe_interval is not None:
            return iteration > 0 and iteration % self._probe_interval == 0
        return False

    def probe_role(self, iteration: int) -> Optional[str]:
        """``"control"``, ``"active"``, or None when the probe does not fire.

        A control R is a baseline. It is measured and logged as usual (the
        ``role`` field of ``corr_matrix_log.json``), but it must not be turned
        into a q and handed to ``set_weights``.
        """
        if not self.should_probe(iteration):
            return None
        return "control" if iteration in self._control_iters else "active"

    # --- helpers ---------------------------------------------------------------

    def _as_distribution(self, q: Weights) -> np.ndarray:
        """q as a float64 vector aligned to the vocabulary, validated and summing to 1."""
        if isinstance(q, Mapping):
            unknown = sorted(set(q) - set(self._names))
            missing = [n for n in self._names if n not in q]
            if unknown or missing:
                raise ValueError(
                    f"{type(self).__name__}: weights must cover the vocabulary exactly; "
                    f"unknown {unknown}, missing {missing}"
                )
            arr = np.array([q[n] for n in self._names], dtype=np.float64)
        else:
            arr = np.asarray(q, dtype=np.float64)
            if arr.shape != (len(self._names),):
                raise ValueError(
                    f"{type(self).__name__}: weights have shape {arr.shape}, expected "
                    f"({len(self._names)},) aligned to vocabulary()"
                )
        if not np.all(np.isfinite(arr)):
            bad = [n for n, v in zip(self._names, arr) if not np.isfinite(v)]
            raise ValueError(f"{type(self).__name__}: non-finite weights for {bad}")
        if np.any(arr < 0):
            bad = [n for n, v in zip(self._names, arr) if v < 0]
            raise ValueError(f"{type(self).__name__}: negative weights for {bad}")
        total = float(arr.sum())
        if total <= 0:
            raise ValueError(f"{type(self).__name__}: weights sum to {total}; nothing to sample")
        return arr / total
