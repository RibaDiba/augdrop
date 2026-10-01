"""Hard: Quadratic Programming Feature Selection over R.

Not in Sensaug; new here. QPFS (Rodriguez-Lujan, Huerta, Elkan & Santa Cruz, JMLR
2010) replaces mRMR's greedy search with one convex program over op weights x on the
simplex::

    minimise   1/2 (1 - alpha) x^T Q x  -  alpha F^T x    s.t.  x >= 0, sum(x) = 1

Q is the pairwise redundancy between ops (``psd_similarity`` of ``pairwise(R)``, unit
diagonal), and F the relevance (the prior). The ops kept are the top ``budget`` by x.
Because every op is weighed against every other at once, one op's redundancy with a
cluster counts against it even when greedy mRMR's selection order would have
reached it before the cluster.

alpha trades relevance against redundancy. The paper's default, used when alpha is
None, balances their mean magnitudes, ``alpha = mean(Q) / (mean(Q) + mean(F))``.
That makes the objective invariant to rescaling F, so the ranking doesn't depend on
how the prior is normalized. With a uniform prior F^T x is constant on the simplex,
and QPFS reduces to minimum redundancy, as mRMR does.

x is often sparse, with exact zeros. When the budget reaches into the zeros, the
rest are chosen by the same tie-break mRMR uses (least redundant against the whole
bank first, then name order).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import Decline, HardSelector, psd_similarity, tie_break

__all__ = ["QPFS", "solve_simplex_qp"]


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x) = 1} (Duchi et al., ICML 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, v.size + 1)
    rho = np.flatnonzero(u - css / k > 0)[-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def solve_simplex_qp(
    H: np.ndarray, f: np.ndarray, *, tol: float = 1e-12, max_iter: int = 100_000
) -> np.ndarray:
    """argmin of ``1/2 x^T H x - f^T x`` on the simplex, for a PSD ``H``.

    Accelerated projected gradient (FISTA) with function-value restart, from the
    uniform point. Deterministic. Raises ``Decline`` if it has not converged within
    ``max_iter`` steps: a ranking off a half-solved program would look like a real one.
    """
    n = f.size
    lipschitz = max(float(np.linalg.eigvalsh(H)[-1]), 1e-12)

    def objective(x):
        return 0.5 * x @ H @ x - f @ x

    x = np.full(n, 1.0 / n)
    y = x.copy()
    t = 1.0
    fx = objective(x)
    restarted = False
    for _ in range(max_iter):
        x_new = _project_simplex(y - (H @ y - f) / lipschitz)
        if np.max(np.abs(x_new - x)) < tol:
            return x_new
        f_new = objective(x_new)
        if f_new > fx:
            if restarted:
                # A plain projected-gradient step from x provably descends unless x
                # is optimal, so this "ascent" is rounding: x is optimal to float
                # precision.
                return x
            # Momentum overshot: restart from the last iterate.
            y, t, restarted = x.copy(), 1.0, True
            continue
        restarted = False
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, fx, t = x_new, f_new, t_new
    raise Decline(f"QPFS did not converge in {max_iter} iterations")


class QPFS(HardSelector):
    """QPFS over the measured ops.

    mode:   how a cell of R reduces to redundancy (``squared`` default, ``abs``,
            ``signed``)
    alpha:  the relevance weight in [0, 1]. None uses the paper's
            ``mean(Q) / (mean(Q) + mean(F))``.
    lam:    derive the budget as in ``MRMR``
    """

    name = "qpfs"

    def __init__(
        self,
        *,
        mode: str = "squared",
        alpha: Optional[float] = None,
        lam: Optional[float] = None,
    ):
        super().__init__(mode=mode, lam=lam)
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def weights(self, sub: np.ndarray, rel: np.ndarray) -> np.ndarray:
        """QPFS's x over the ops of ``sub`` (a ``pairwise`` block) with prior ``rel``."""
        Q = psd_similarity(sub)
        if self.alpha is None:
            q_bar, f_bar = float(Q.mean()), float(rel.mean())
            alpha = q_bar / (q_bar + f_bar)
        else:
            alpha = float(self.alpha)
        return solve_simplex_qp((1.0 - alpha) * Q, alpha * rel)

    def _rank(self, sub: np.ndarray, rel: np.ndarray, budget: int, names: List[str]) -> List[int]:
        x = self.weights(sub, rel)
        return list(np.lexsort((tie_break(sub), -x)))
