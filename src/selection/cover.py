"""Hard: move a covered op's mass off it, either to the op that covers it
(``cover_single``) or spread over every other op (``cover_even``). A first, naive cut
at substitution, checked with the TrivialAugment hosts and SensAug.

New, not in Sensaug. The two selectors are one experiment: they move exactly the same
ops by exactly the same amount and differ only in where the mass goes, so comparing
them isolates "substitute the cover" from "drop and renormalize".

Shared by both:

- **Evidence is an effect size, not significance.** A pair qualifies when the lower
  bound of its bootstrap CI is above ``tau`` (and the cell survived FDR, when a gate
  ran). FDR alone tests r against the closure null, which a cell of r = 0.01 can
  pass.
- **One partner per moved op, pairs strongest first** (by CI lower bound). Of a
  qualifying pair, the op with the smaller prior is moved; on a tie, the one later in
  name order, so the choice is the same at every emission. The partner is pinned: it
  is never moved itself, so there are no chains. An op already moved can't be a
  partner. The two halves of one op are never a pair under ``mask_within_op``.
- **lam is the fraction of the moved op's mass taken off it**, in [0, 1]. Each moved
  op ends at exactly ``(1 - lam) * prior`` in both selectors. ApplyWeightsHook ramps
  lam 0 -> target over training, as it ramps every selector's lam. At 1 the op is
  pruned.
- **Stateless.** Each emission decides on its own R. There is no persistence gate,
  no hysteresis and no rollback.

Where the mass goes:

- ``cover_single``: all of it to the moved op's partner, its cover. Every other op
  keeps exactly its prior. No cap on what one cover absorbs.
- ``cover_even``: over every op that isn't moved (partners included), in proportion
  to its prior, so their ratios to each other are unchanged. What every other hard
  selector does with pruned mass.

The q is per op. On TrivialAugment, where every op is drawn at the same uniform bin,
``cover_single``'s q is exactly "apply the cover where the method drew the moved op".
On SensAug it is per op too: ``set_weights`` scales all of an op's (op, level) entries
by one factor, so every op keeps the level shape its SA pdf gave it, and the mass
``cover_single`` moves is drawn at the cover's levels, not the moved op's. SensAug's
default pdf (``sa.generate_pdf_new``) gives every op an equal allotment, so there the
moved op of a pair is the later one by name (priors within ``_PRIOR_RTOL`` are a tie).
On AutoAugment a per-op q isn't realizable (sub-policies), so ``dropaug.config.check``
accepts these only with a TrivialAugment host or sensaug.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import numpy as np

from stats.redundancy import within_op_pairs

from .base import Redundancy, Selection, Selector, as_relevance, declined

__all__ = ["COVER_SELECTORS", "CoverEven", "CoverSingle", "cover_pairs"]

#: Two priors this close are a tie. SensAug's pdf gives every op the same allotment,
#: and each op's prior is summed over its levels in that op's own mIoU order. On 200
#: random pdfs those sums came out bit-equal, but nothing guarantees it, and round-off
#: must not pick the moved op.
_PRIOR_RTOL = 1e-9


def cover_pairs(R: Redundancy, tau: float) -> List[Tuple[float, str, str]]:
    """``(ci_lo, a, b)`` for every cross-op pair whose CI lower bound is above ``tau``,
    strongest first (ties by name). ``a`` < ``b`` by name."""
    if R.ci_lo is None:
        return []
    masked = set()
    if R.mask_within_op:
        masked = {frozenset((i, j)) for i, j in within_op_pairs(R.names)}
    pairs = []
    n = len(R.names)
    for i in range(n):
        for j in range(i + 1, n):
            lo = R.ci_lo[i, j]
            if not (np.isfinite(lo) and lo > tau) or frozenset((i, j)) in masked:
                continue
            if R.survives is not None and not R.survives[i, j]:
                continue
            a, b = sorted((R.names[i], R.names[j]))
            pairs.append((float(lo), a, b))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    return pairs


class _Cover(Selector):
    """The shared gate and pairing. Subclasses implement ``_redistribute``."""

    hard = True

    def __init__(self, *, lam: Optional[float] = None, tau: float = 0.3):
        if lam is None:
            raise ValueError(f"{self.name} needs lam, the fraction of a moved op's mass taken off it (0-1)")
        if not (np.isfinite(lam) and 0.0 <= lam <= 1.0):
            raise ValueError(f"{self.name}'s lam is a fraction of mass, in [0, 1]; got {lam}")
        if not (np.isfinite(tau) and -1.0 < tau < 1.0):
            raise ValueError(f"tau must be a correlation in (-1, 1), got {tau}")
        self.lam = float(lam)
        self.tau = float(tau)

    def _redistribute(
        self, q: np.ndarray, prior: np.ndarray, moved: Dict[int, int], taken: np.ndarray
    ) -> np.ndarray:
        """q with ``taken[i]`` already removed from each moved op ``i``; add it back
        somewhere. ``moved`` maps a moved op's index to its partner's."""
        raise NotImplementedError

    def select(self, R: Redundancy, rel: Mapping, budget: Optional[int] = None) -> Selection:
        names, prior = as_relevance(rel)
        if self.lam == 0.0:
            return declined(names, prior, "lam is 0, so no mass moves")
        if R.ci_lo is None:
            return declined(names, prior, f"R has no bootstrap CIs, and {self.name} gates on their lower bounds")
        prior_of: Dict[str, float] = dict(zip(names, prior))

        covers: Dict[str, str] = {}  # moved op -> its partner
        pinned = set()
        for _, a, b in cover_pairs(R, self.tau):
            if prior_of.get(a, 0.0) <= 0 or prior_of.get(b, 0.0) <= 0:
                continue
            if a in covers or b in covers:
                continue  # one side is already moved
            if a in pinned and b in pinned:
                continue
            if a in pinned or b in pinned:
                gone, partner = (b, a) if a in pinned else (a, b)
            elif not np.isclose(prior_of[a], prior_of[b], rtol=_PRIOR_RTOL, atol=0.0):
                gone, partner = (a, b) if prior_of[a] < prior_of[b] else (b, a)
            else:
                gone, partner = b, a  # a < b by name: keep a
            covers[gone] = partner
            pinned.add(partner)

        if not covers:
            return declined(
                names,
                prior,
                f"no cross-op pair has a bootstrap CI lower bound above tau={self.tau:g}",
            )

        index = {name: i for i, name in enumerate(names)}
        moved = {index[g]: index[p] for g, p in covers.items()}
        prior = prior / prior.sum()
        q = prior.astype(np.float64).copy()
        taken = np.zeros_like(q)
        for i in moved:
            taken[i] = self.lam * prior[i]
            q[i] = 0.0 if self.lam == 1.0 else prior[i] - taken[i]  # exactly 0 at lam 1
        q = self._redistribute(q, prior, moved, taken)
        q = q / q.sum()
        pruned = tuple(n for n in names if prior_of[n] > 0 and q[index[n]] == 0.0)
        return Selection(
            names,
            q,
            True,
            None,
            kept=tuple(n for n in names if q[index[n]] > 0),
            pruned=pruned,
            covers=tuple(sorted(covers.items())),
        )


class CoverSingle(_Cover):
    """Each moved op's mass goes to its cover; every other op keeps its prior."""

    name = "cover_single"

    def _redistribute(self, q, prior, moved, taken):
        for i, j in moved.items():
            q[j] += taken[i]
        return q


class CoverEven(_Cover):
    """The moved mass is spread over every op that isn't moved, in proportion to its
    prior. Same ops moved as ``cover_single``; ``covers`` is the evidence, not the
    destination."""

    name = "cover_even"

    def _redistribute(self, q, prior, moved, taken):
        rest = np.ones(len(q), dtype=bool)
        rest[list(moved)] = False
        rest &= prior > 0
        q[rest] += taken.sum() * prior[rest] / prior[rest].sum()
        return q


#: Both arms of the substitution experiment.
COVER_SELECTORS = (CoverSingle.name, CoverEven.name)
