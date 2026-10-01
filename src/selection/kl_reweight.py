"""Soft: ``q(a) proportional to p(a) * exp(-lambda * red(a))``. Sensaug's Lever 3.

Ported from Sensaug's ``redundancy.reweight`` (its ``soft-weighting`` arm), over ops
rather than ``(op, level)`` pdf entries. The host scales every level of an op by the
op's factor and holds the no-op mass fixed (``SensAugHost.pdf``), which is what
``reweight`` did on the whole pdf. red(a) is ``stats.compute_red``.

The rule is the closed-form solution to minimising KL(q || p) subject to
``sum_a q(a) red(a) <= C`` and ``sum_a q(a) = 1``, with lambda the Lagrange
multiplier on the redundancy budget C. So it is a derivation rather than a
heuristic, and lambda=0 is exactly p.

* **red(a) is standardized, and that is load-bearing.** A constant added to red
  cancels in the normaliser, so only deviations from the mean act. Dividing by the
  standard deviation makes one lambda mean the same thing at every checkpoint. On
  Sensaug's logged matrices lambda=0.25 gives a max/min spread of 2.3-3.1x for both
  the 14- and 32-op vocabularies (``scripts/calibrate_lambda.py`` there).
* **Soft, never zero.** exp() is strictly positive, and the exponent is clipped so
  that stays true in floating point. At the correlation sizes Sensaug observed
  (mean |r| 0.11-0.22), deleting an op is not justified.

One deliberate difference from Sensaug, in ``measured_red``: an op whose whole row
of R is NaN is left out of red(a) instead of being scored 0 (see there). On an R
with no NaN row the two agree exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Dict, Optional

import numpy as np

from stats.redundancy import RedundancyScore, compute_red

from .base import MODES, Redundancy, Selection, Selector, as_relevance, declined

__all__ = ["KLReweight", "measured_red", "reweight"]

# Half-width of the window the exponent is confined to, so the widest achievable
# max/min ratio is exp(2 * _EXP_CLIP) ~ 4e260: inside float64's normal range and far
# past any useful setting (lambda=1 on a standardized score spans about 11). exp()
# is strictly positive in exact arithmetic, but a wide enough exponent underflows to
# exactly 0 after normalisation, which would silently turn down-weighting into
# deletion. Past the clip the request saturates instead.
_EXP_CLIP = 300.0


def measured_red(R: Redundancy, mode: str = "squared", min_std: float = 1e-6) -> RedundancyScore:
    """``stats.compute_red`` over the ops R actually measured.

    compute_red (like Sensaug's) keeps an op whose whole row is NaN in the
    standardization with a raw row sum of 0, the lowest a squared or abs sum can be.
    Standardized, that reads as the LEAST redundant op in the bank, and the tilt
    up-weights it on the strength of no measurement. In dropaug that is not rare:
    every op without a magnitude (AutoContrast, Equalize, Invert) gets a NaN row in
    G. So the score is computed on the measured sub-matrix, and a NaN-row op gets no
    score at all, which ``reweight`` reads as "leave it at its prior ratio". It is
    listed in ``dropped``, and ``names`` holds the scored ops only.
    """
    n = len(R.names)
    off_diagonal = ~np.eye(n, dtype=bool)
    all_nan = np.all(np.isnan(np.where(off_diagonal, R.r, np.nan)), axis=1)
    keep = np.flatnonzero(~all_nan)
    dropped = [R.names[i] for i in np.flatnonzero(all_nan)]
    if keep.size == 0:
        return RedundancyScore(
            names=[],
            raw=np.zeros(0),
            std=np.zeros(0),
            dropped=dropped,
            mode=mode,
            reason="every op was dropped upstream; R carries no usable cell",
        )
    survives = None if R.survives is None else R.survives[np.ix_(keep, keep)]
    score = compute_red(
        R.r[np.ix_(keep, keep)],
        [R.names[i] for i in keep],
        mode=mode,
        mask_within_op=R.mask_within_op,
        survivor_mask=survives,
        min_std=min_std,
    )
    return score._replace(dropped=dropped)


def reweight(rel: Mapping, red: Optional[Dict[str, float]], lam: float) -> Selection:
    """Max-entropy tilt of the prior ``rel`` by a per-op score ``red``.

    An op missing from ``red`` (or scored NaN) scores 0, which on a standardized
    score is the mean: no adjustment either way. lam=0 declines rather than
    multiplying by exp(0) and renormalising through float error, so the control
    arm stays bit-identical when the caller skips ``set_weights``.
    """
    names, prior = as_relevance(rel)
    if not (math.isfinite(lam) and lam >= 0):
        raise ValueError(f"lambda must be a finite number >= 0, got {lam}")
    if lam == 0:
        return declined(names, prior, "lambda is 0")
    if not red:
        return declined(names, prior, "no redundancy score available")

    scores = np.array([float(red.get(n, 0.0)) for n in names], dtype=np.float64)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    live = prior > 0
    if not np.any(scores[live]):
        return declined(names, prior, "no op in rel has a redundancy score")

    # Re-centring cancels in the normaliser, but it keeps the exponent centred on 0
    # so the clip bites symmetrically instead of lopping off one tail.
    exponent = np.clip(
        -float(lam) * (scores - scores[live].mean()), -_EXP_CLIP, _EXP_CLIP
    )
    # Shift the top to 0 before exponentiating: exp() lands in (0, 1] and cannot
    # overflow, and the sum is at least its largest term, so the normalisation below
    # cannot underflow the small end either.
    exponent -= exponent[live].max()
    # Only over the live ops: an op with prior 0 can sit above the shifted top, and
    # 0 * exp(600) is NaN.
    weights = np.zeros_like(prior)
    weights[live] = prior[live] * np.exp(exponent[live])
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        return declined(names, prior, "reweighting produced a degenerate distribution")
    return Selection(names, weights / total, True, None)


class KLReweight(Selector):
    """``q(a) proportional to p(a) * exp(-lam * red(a))``, red(a) from ``measured_red``.

    lam:      the tilt strength. Required: the control is lam=0 and has to be
              stated. Scale it by training progress with ``ramp_lambda`` if wanted.
    mode:     how a row of R reduces to red(a): ``squared`` (default), ``abs``,
              ``signed``
    min_std:  below this spread of red(a), every op is equally redundant and the
              selector declines
    budget:   ignored. This selector is soft; its strength is lam.
    """

    name = "kl_reweight"
    hard = False

    def __init__(self, *, lam: float, mode: str = "squared", min_std: float = 1e-6):
        if not (math.isfinite(lam) and lam >= 0):
            raise ValueError(f"lam must be a finite number >= 0, got {lam}")
        if mode not in MODES:
            raise ValueError(f"unknown red mode {mode!r}, expected one of {MODES}")
        self.lam = float(lam)
        self.mode = mode
        self.min_std = float(min_std)

    def select(self, R: Redundancy, rel: Mapping, budget: Optional[int] = None) -> Selection:
        names, prior = as_relevance(rel)
        if self.lam == 0:
            return declined(names, prior, "lambda is 0")
        score = measured_red(R, self.mode, self.min_std)
        if not score.usable:
            return declined(names, prior, score.reason)
        return reweight(rel, score.as_dict(), self.lam)
