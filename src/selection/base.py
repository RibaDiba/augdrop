"""The selector contract, the R it reads, and what every hard selector shares.

A selector is one function::

    Selector.select(R, rel, budget) -> Selection        # .q is the new weights

R       a ``Redundancy``: op names, the (A, A) correlation matrix ``stats`` builds
        (NaN where nothing was measured), and the two masks Sensaug applies to it,
        the BH-FDR survivors and the within-op pairs.
rel     ``{op: relevance}``, in the order q comes back in. Pass the host's own
        distribution, ``dict(zip(host.vocabulary(), host.base_probs()))``. It is
        both the prior q is taken against and mRMR's relevance term. The same is
        true in Sensaug, where both were the SA pdf's per-op mass. An op whose rel
        is 0 (one the host excludes) always gets q = 0.
budget  how many of the measured ops a hard selector keeps. A soft selector
        ignores it. A hard selector built with ``lam`` derives it as Sensaug's mRMR
        arm did, ``ceil(A_measured / (1 + lam))``, when budget is None. ``nms`` takes
        none: a threshold on R decides, and it also needs a per-op score (nms.py).

Rules every selector keeps. They are Sensaug's ``DOWNWEIGHT_METHODS`` contract
(``sensaug/loops/grad_corr_loop.py``):

- **q changes which op is drawn, never how often.** It is over ops only, and the
  host holds its no-op probability fixed.
- **Decline, don't degrade.** When the input can't support a decision (no usable R,
  flat relevance, a budget that keeps everything), ``applied`` is False, ``reason``
  says why, and ``q`` is the prior. Leave the host alone then rather than handing it
  that q: skipping ``set_weights`` keeps a control arm bit-identical. Malformed input
  (a shape mismatch, a negative relevance) raises.
- **Soft unless declared hard.** Only a selector with ``hard = True`` may drive an op
  with a positive prior to exactly 0, and it drops the whole op.
- **An op with no usable row in R is exempt.** A hard selector keeps it, outside the
  budget, and the soft one leaves it at its prior ratio. Pruning an op on the
  strength of a measurement that doesn't exist can't be argued for.
- **Deterministic, and independent of rel's order.** Hard selectors rank the
  measured ops in name order, so ties break the same way on every DDP rank and for
  every host's vocabulary order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from stats.redundancy import MODES, within_op_pairs

__all__ = [
    "Decline",
    "HardSelector",
    "MODES",
    "Redundancy",
    "Selection",
    "Selector",
    "as_relevance",
    "budget_from_lambda",
    "declined",
    "finite_mean",
    "pairwise",
    "psd_similarity",
    "ramp_lambda",
    "tie_break",
]

#: Guards a standardization whose spread is exactly 0. A uniform prior (every host
#: but SensAug's) makes the relevance spread 0. That is not an error: the ranking
#: then falls through to pure minimum redundancy. Sensaug's ``_MRMR_STD_FLOOR``.
STD_FLOOR = 1e-8

#: The smallest eigenvalue ``psd_similarity`` leaves in the kernel.
_PSD_FLOOR = 1e-6


class Decline(Exception):
    """Raised inside a selector when the input can't support a decision.
    ``HardSelector.select`` turns it into ``applied=False`` with this reason."""


# --- R ------------------------------------------------------------------------


def _as_matrix(value) -> np.ndarray:
    """A float matrix, tolerating the JSON round trip. ``corr_matrix_log.json``
    writes every non-finite cell as null, which numpy refuses to read as a float.
    Ported from Sensaug's ``grad_corr_loop._as_matrix``."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    try:
        return np.asarray(
            [[np.nan if v is None else float(v) for v in row] for row in value],
            dtype=np.float64,
        )
    except TypeError as exc:
        raise ValueError(f"R could not be read as a matrix ({exc})") from None


@dataclass(frozen=True)
class Redundancy:
    """R, labelled, with the masks that say which of its cells count as evidence.

    names:           the axis order of ``r`` and ``survives``
    r:               (A, A) correlation matrix. NaN marks a cell with no
                     measurement, and a whole NaN row an op that was dropped or
                     never probed. Nested lists with None for NaN are accepted.
    survives:        (A, A) bool from ``stats.fdr_correct``, scattered back to A.
                     A cell that did not survive the FDR gate is not evidence. None
                     means no gate was run, and every measured cell counts.
    mask_within_op:  exclude the two halves of one op (``lighter_X``/``darker_X``,
                     ``X_pos``/``X_neg``). They measure a parameterization
                     convention, not redundancy between two separate choices.
    """

    names: Tuple[str, ...]
    r: np.ndarray
    survives: Optional[np.ndarray] = None
    mask_within_op: bool = True

    def __post_init__(self):
        names = tuple(str(n) for n in self.names)
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Redundancy: duplicate op names {dupes}")
        r = _as_matrix(self.r)
        if r.shape != (len(names), len(names)):
            raise ValueError(
                f"Redundancy: r is {r.shape} but {len(names)} op names came with it; "
                f"refusing to guess the axis order"
            )
        survives = None
        if self.survives is not None:
            survives = np.asarray(self.survives, dtype=bool)
            if survives.shape != r.shape:
                raise ValueError(f"Redundancy: survives is {survives.shape}, expected {r.shape}")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "survives", survives)
        object.__setattr__(self, "mask_within_op", bool(self.mask_within_op))


def pairwise(R: Redundancy, mode: str = "squared") -> np.ndarray:
    """(A, A) pairwise redundancy: R reduced cell by cell under ``mode``, and NaN
    wherever the pair carries no usable evidence. That is the diagonal, a within-op
    pair, a cell that failed the FDR gate, and a cell R never measured.

    Ported from Sensaug's ``_pairwise_redundancy``. These are the cells
    ``stats.compute_red`` sums into red(a), under the same masks and reductions, so
    the soft and hard selectors differ in what they do with R, not in what they
    think it says. One difference: a masked cell is NaN here where compute_red
    zeroes it. To a selector that ranks pairs, "no evidence" is not "no
    redundancy".
    """
    if mode not in MODES:
        raise ValueError(f"unknown red mode {mode!r}, expected one of {MODES}")
    if mode == "squared":
        work = R.r**2
    elif mode == "abs":
        work = np.abs(R.r)
    else:
        # Under 'signed' an anti-correlated pair scores negative, so it PROTECTS
        # both ops rather than pruning either. That is why 'squared' is the default.
        work = R.r.copy()
    np.fill_diagonal(work, np.nan)
    if R.mask_within_op:
        for i, j in within_op_pairs(R.names):
            work[i, j] = np.nan
            work[j, i] = np.nan
    if R.survives is not None:
        work = np.where(R.survives, work, np.nan)
    return work


# --- shared numerics ------------------------------------------------------------


def finite_mean(values) -> float:
    """Mean over the finite entries, NaN when there are none. np.nanmean would do it,
    but it warns on an all-NaN slice, and those are expected here: an op whose
    partners all failed the FDR gate has no measured redundancy against them."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def tie_break(sub: np.ndarray) -> np.ndarray:
    """Each op's mean redundancy against the whole measured bank, used to break ties
    (lower first). An op with no finite cell gets the bank's mean.

    Ties are the norm when the prior is uniform, which it is for every host but
    SensAug's. Breaking them on the least redundant op keeps the ranking meaningful
    instead of falling back to name order. Sensaug's mRMR ``global_red``.
    """
    cells = sub[np.isfinite(sub)]
    mu = float(cells.mean()) if cells.size else 0.0
    per_op = np.array([finite_mean(row) for row in sub], dtype=np.float64)
    return np.where(np.isnan(per_op), mu, per_op)


def psd_similarity(sub: np.ndarray) -> np.ndarray:
    """A similarity kernel over the measured ops, for the selectors that need a PSD
    matrix (DPP, QPFS): the pairwise redundancy with a unit diagonal and 0 wherever
    there is no evidence.

    A full correlation matrix is PSD, and so is its elementwise square (Schur), but
    masking cells to 0 can break that. When it does, the negative eigenvalues are
    clipped to a small floor and the result is rescaled back to a unit diagonal.
    """
    s = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
    s = (s + s.T) / 2.0
    np.fill_diagonal(s, 1.0)
    eigvals, eigvecs = np.linalg.eigh(s)
    if eigvals.min() < _PSD_FLOOR:
        s = (eigvecs * np.maximum(eigvals, _PSD_FLOOR)) @ eigvecs.T
        d = np.sqrt(np.diag(s))
        s = s / np.outer(d, d)
        s = (s + s.T) / 2.0
    return s


def budget_from_lambda(n_measured: int, lam: float) -> int:
    """Sensaug's mRMR budget, ``ceil(A / (1 + lam))`` of the A measured ops, at least 1.

    lam=0 keeps everything, so it is continuous with the lam=0 control instead of
    jumping at it. 0.25 keeps ~80%, 0.5 ~67%, 1.0 half, 2.0 a third.
    """
    if not (math.isfinite(lam) and lam >= 0):
        raise ValueError(f"lam must be a finite number >= 0, got {lam}")
    return max(1, math.ceil(n_measured / (1.0 + float(lam))))


def ramp_lambda(target: float, progress: float, mode: str = "linear") -> float:
    """Scale lambda by training progress, the fraction of training elapsed.

    R measured early describes a model that barely tells augmentations apart yet, so
    full strength from step 0 acts hardest on the least trustworthy measurement.
    Ported from Sensaug's ``redundancy.ramp_lambda``.
    """
    if mode == "constant":
        return float(target)
    if mode == "linear":
        return float(target) * float(np.clip(progress, 0.0, 1.0))
    raise ValueError(f"unknown lambda ramp {mode!r}, expected 'linear' or 'constant'")


def as_relevance(rel) -> Tuple[Tuple[str, ...], np.ndarray]:
    """rel as ``(names, values)``: validated, not normalized."""
    if not isinstance(rel, Mapping):
        raise TypeError(
            f"rel must map op name -> relevance, e.g. "
            f"dict(zip(host.vocabulary(), host.base_probs())); got {type(rel).__name__}"
        )
    names = tuple(str(k) for k in rel)
    if not names:
        raise ValueError("rel is empty")
    values = np.array([float(v) for v in rel.values()], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        bad = [n for n, v in zip(names, values) if not np.isfinite(v)]
        raise ValueError(f"non-finite relevance for {bad}")
    if np.any(values < 0):
        bad = [n for n, v in zip(names, values) if v < 0]
        raise ValueError(f"negative relevance for {bad}")
    if values.sum() <= 0:
        raise ValueError("rel sums to 0; there is no op to weight")
    return names, values


# --- the contract ---------------------------------------------------------------


class Selection(NamedTuple):
    """What ``Selector.select`` returns.

    names:    rel's ops, in rel's order. q is aligned to them, so for a host
              ``host.set_weights(selection.q)`` works directly.
    q:        the new weights, summing to 1. When ``applied`` is False this is the
              normalized prior, and the host should be left alone instead.
    applied:  whether q was moved on purpose
    reason:   None when applied, otherwise why the selector declined
    kept:     hard selectors only: the ops left with q > 0, in the selector's rank
              order, followed by the exempt ops. None for a soft selector.
    exempt:   ops kept because R has no usable row for them
    pruned:   ops with a positive prior that q sets to exactly 0
    """

    names: Tuple[str, ...]
    q: np.ndarray
    applied: bool
    reason: Optional[str]
    kept: Optional[Tuple[str, ...]] = None
    exempt: Tuple[str, ...] = ()
    pruned: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, float]:
        return {name: float(v) for name, v in zip(self.names, self.q)}

    @property
    def spread(self) -> float:
        """max/min over the ops q keeps, as in Sensaug's log line. 1.0 when not
        applied."""
        if not self.applied:
            return 1.0
        positive = self.q[self.q > 0]
        return float(positive.max() / positive.min()) if positive.size else 1.0


def declined(names: Tuple[str, ...], prior: np.ndarray, reason: str) -> Selection:
    """The Selection a selector returns when it leaves the prior alone."""
    return Selection(names, prior / prior.sum(), False, reason)


class Selector:
    """Base selector. Subclasses set ``name`` and ``hard`` and implement ``select``."""

    #: The key ``build_selector`` finds this selector by.
    name: ClassVar[str] = ""
    #: Whether this selector may set an op with positive prior to exactly 0.
    hard: ClassVar[bool] = False

    def select(self, R: Redundancy, rel: Mapping, budget: Optional[int] = None) -> Selection:
        raise NotImplementedError


class HardSelector(Selector):
    """A selector that keeps ``budget`` of the measured ops and zeroes the rest.

    Subclasses implement ``_rank``. Everything around it is shared, so the hard
    selectors differ in exactly one thing: which ops they keep. The pool is the ops
    with a positive prior. The measured ones (a finite cell anywhere in their row
    of ``pairwise(R)``) are ranked, and the rest are exempt and always kept. The
    ops kept keep their prior ratios; only the pruned mass moves, to them in
    proportion to what they already held.
    """

    hard = True

    def __init__(self, *, mode: str = "squared", lam: Optional[float] = None):
        if mode not in MODES:
            raise ValueError(f"unknown red mode {mode!r}, expected one of {MODES}")
        if lam is not None and not (math.isfinite(lam) and lam >= 0):
            raise ValueError(f"lam must be a finite number >= 0, got {lam}")
        self.mode = mode
        self.lam = lam

    def _rank(self, sub: np.ndarray, rel: np.ndarray, budget: int, names: List[str]) -> Sequence[int]:
        """Indices into ``names`` of the ops to keep, best first; the first
        ``budget`` are kept. ``names`` is sorted, ``sub`` is ``pairwise(R)`` over
        them, and ``rel`` their prior. Raise ``Decline`` to leave the prior alone."""
        raise NotImplementedError

    def _budget(self, budget: Optional[int], n_measured: int) -> int:
        if budget is None:
            if self.lam is None:
                raise ValueError(
                    f"{self.name} needs a budget: pass budget=k, or build it with lam= "
                    f"to derive one as ceil(A / (1 + lam))"
                )
            return budget_from_lambda(n_measured, self.lam)
        if self.lam is not None:
            raise ValueError(
                f"{self.name} was built with lam={self.lam} and also given budget={budget}; "
                f"they are two ways of setting the same thing"
            )
        if isinstance(budget, bool) or int(budget) != budget or budget < 1:
            raise ValueError(f"budget must be a whole number of ops >= 1, got {budget!r}")
        return int(budget)

    def select(self, R: Redundancy, rel: Mapping, budget: Optional[int] = None) -> Selection:
        names, prior = as_relevance(rel)
        prior_of = dict(zip(names, prior))
        work = pairwise(R, self.mode)
        index = {name: i for i, name in enumerate(R.names)}

        pool = sorted(n for n in names if prior_of[n] > 0)
        measured = [n for n in pool if n in index and np.isfinite(work[index[n]]).any()]
        measured_set = set(measured)
        exempt = [n for n in pool if n not in measured_set]
        if not measured:
            return declined(
                names,
                prior,
                "no op in rel has a usable row in R (vocabulary mismatch, or every cell "
                "was masked out); there is nothing to rank",
            )

        k = self._budget(budget, len(measured))
        if k >= len(measured):
            return declined(
                names,
                prior,
                f"a budget of {k} keeps all {len(measured)} measured ops, so nothing is pruned",
            )

        idx = np.array([index[n] for n in measured])
        sub = work[np.ix_(idx, idx)]
        rel_sub = np.array([prior_of[n] for n in measured], dtype=np.float64)
        try:
            order = list(self._rank(sub, rel_sub, k, measured))
        except Decline as exc:
            return declined(names, prior, str(exc))

        chosen = [measured[i] for i in order[:k]]
        keep = set(chosen) | set(exempt)
        q = np.where([n in keep for n in names], prior, 0.0)
        q = q / q.sum()
        pruned = tuple(n for n in names if prior_of[n] > 0 and n not in keep)
        return Selection(
            names,
            q,
            True,
            None,
            kept=tuple(chosen) + tuple(exempt),
            exempt=tuple(exempt),
            pruned=pruned,
        )
