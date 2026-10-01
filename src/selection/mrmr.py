"""Hard: greedy minimum-Redundancy Maximum-Relevance. Sensaug's mRMR arm.

Ported from Sensaug's ``_downweight_mrmr`` (``sensaug/loops/grad_corr_loop.py``).
The ranking is unchanged. The pool, the exemption and the pruning around it are
``HardSelector``'s, and the budget is explicit, or ``ceil(A / (1 + lam))`` as in
Sensaug when built with ``lam``.

The ranking is textbook greedy mRMR. Seed with the most relevant op, then repeatedly
take the op maximising ``rel(a) - red(a | S)`` against the already-selected set S,
where red(a | S) is a's mean pairwise redundancy with S. Both terms are standardized
across ops, so the difference is dimensionless and neither dominates by unit choice.
That is what keeps the objective lambda-free. With a uniform prior (every host but
SensAug's) the relevance term is flat, and the ranking becomes pure minimum-
redundancy selection rather than an arbitrary order.

Two things to know before running it. Deletion is a stronger claim than Sensaug's
observed correlations (mean |r| 0.11-0.22) support on their own, so read a result
from it next to ``kl_reweight`` at a matching lambda and against the ``random_k``
null band. And the prune is not latched: it is re-derived from whatever R and prior
it is given, so an op pruned at one probe can return at the next.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import STD_FLOOR, HardSelector, finite_mean, tie_break

__all__ = ["MRMR"]


class MRMR(HardSelector):
    """Greedy mRMR over the measured ops.

    mode:  how a cell of R reduces to redundancy (``squared`` default, ``abs``,
           ``signed``), the same reduction ``kl_reweight`` applies row-wise
    lam:   derive the budget as ``ceil(A_measured / (1 + lam))`` when select is
           given none
    """

    name = "mrmr"

    def __init__(self, *, mode: str = "squared", lam: Optional[float] = None):
        super().__init__(mode=mode, lam=lam)

    def _rank(self, sub: np.ndarray, rel: np.ndarray, budget: int, names: List[str]) -> List[int]:
        rel_z = (rel - rel.mean()) / (rel.std() + STD_FLOOR)

        cells = sub[np.isfinite(sub)]
        red_mu = float(cells.mean()) if cells.size else 0.0
        red_sd = float(cells.std()) if cells.size else 0.0
        global_red = tie_break(sub)

        selected: List[int] = []
        remaining = list(range(len(names)))
        while len(selected) < budget and remaining:
            if selected:
                penalty = np.array(
                    [finite_mean(sub[i, selected]) for i in remaining], dtype=np.float64
                )
                # No measured cell against the selected set is no evidence either
                # way, which is the average, not zero redundancy.
                penalty = np.where(np.isnan(penalty), red_mu, penalty)
                scores = rel_z[remaining] - (penalty - red_mu) / (red_sd + STD_FLOOR)
            else:
                scores = rel_z[remaining]
            # lexsort takes its PRIMARY key last: score descending, then redundancy
            # against the whole bank ascending, then name order.
            pick = int(np.lexsort((global_red[remaining], -scores))[0])
            selected.append(remaining.pop(pick))
        return selected
