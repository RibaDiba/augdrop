"""Hard: the most diverse relevant subset, as the greedy MAP of a DPP over R.

Not in Sensaug; new here. A determinantal point process scores a subset Y of ops by
``det(L_Y)`` with the L-ensemble kernel ``L = diag(rel) S diag(rel)``, S the
similarity between ops (``psd_similarity`` of ``pairwise(R)``, unit diagonal). So::

    det(L_Y) = prod_{a in Y} rel(a)^2  *  det(S_Y)

relevance enters through the product, and redundancy through det(S_Y), which
shrinks toward 0 as the ops in Y become collinear. Unlike mRMR's mean pairwise
penalty, the determinant sees a triple of ops that are jointly redundant even when
no single pair of them is strongly correlated.

The exact MAP is NP-hard, so this takes the standard greedy route: add the op with
the largest marginal gain ``det(L_{Y+a}) / det(L_Y)`` until ``budget`` ops are in,
with the incremental Cholesky update of Chen, Zhang & Zhou (NeurIPS 2018), O(A k^2).
The first pick is the most relevant op, as in mRMR. A global rescale of rel scales
every gain by the same factor, so the ranking doesn't depend on how the prior is
normalized. Ties break as in mRMR.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import Decline, HardSelector, psd_similarity, tie_break

__all__ = ["DPP", "greedy_map"]


def greedy_map(L: np.ndarray, k: int, tie: np.ndarray) -> List[int]:
    """Greedy MAP of the DPP with PSD kernel ``L``: k indices, in the order added.

    ``d2[i]`` holds the gain of adding i to the current set, the Schur complement
    ``L_ii - L_iY L_YY^{-1} L_Yi``, and each step updates it with one new column of
    the Cholesky factor of L_Y. Ties in the gain go to the lower ``tie``, then the
    lower index.
    """
    n = L.shape[0]
    chol = np.zeros((n, k))
    d2 = np.diag(L).astype(np.float64).copy()
    selected: List[int] = []
    remaining = list(range(n))
    for step in range(k):
        pick = int(np.lexsort((tie[remaining], -d2[remaining]))[0])
        j = remaining.pop(pick)
        if not d2[j] > 0:
            raise Decline(
                f"the DPP kernel is singular after {step} ops: every remaining op is "
                f"(numerically) a combination of the ones already kept"
            )
        selected.append(j)
        e = (L[j] - chol[j, :step] @ chol[:, :step].T) / np.sqrt(d2[j])
        chol[:, step] = e
        d2 = d2 - e**2
    return selected


class DPP(HardSelector):
    """Greedy DPP MAP over the measured ops.

    mode:  how a cell of R reduces to similarity (``squared`` default, ``abs``,
           ``signed``)
    lam:   derive the budget as in ``MRMR``
    """

    name = "dpp"

    def __init__(self, *, mode: str = "squared", lam: Optional[float] = None):
        super().__init__(mode=mode, lam=lam)

    def _rank(self, sub: np.ndarray, rel: np.ndarray, budget: int, names: List[str]) -> List[int]:
        S = psd_similarity(sub)
        L = rel[:, None] * S * rel[None, :]
        return greedy_map(L, budget, tie_break(sub))
