"""Shrinkage estimates of R, for when N images is not much larger than A ops.

Each cell of the sample R carries noise of order 1/sqrt(N), and the whole matrix
carries more structure than the data supports once A/N is not small: its eigenvalues
spread out even when the ops are independent (``diagnostics.marchenko_pastur_edges``).
Both estimators here pull every off-diagonal cell toward a target by one data-driven
intensity ``lam`` in [0, 1] and keep the unit diagonal::

    r_shrunk[i, j] = lam * target + (1 - lam) * r[i, j]        (i != j)

    ledoit_wolf       Ledoit & Wolf (2004), "A well-conditioned estimator for
                      large-dimensional covariance matrices", on the standardized
                      rows. The target is the identity (target 0), and lam is their
                      closed-form optimum (scikit-learn's ``ledoit_wolf_shrinkage``
                      on the same data).
    schafer_strimmer  Schafer & Strimmer (2005), "A shrinkage approach to large-scale
                      covariance matrix estimation", target D on the standardized
                      rows: lam = sum Var(r_ij) / sum (r_ij - target)^2 over i != j,
                      with Var(r_ij) estimated from the per-image products. The target
                      is a constant off-diagonal value you choose.

For a per-image-normalized G, independent ops sit at ``closure_null(K)``, not at 0
(see ``corr.closure_null``), so pass ``target=closure_null(K)`` to Schafer-Strimmer.
Ledoit-Wolf shrinks toward 0 regardless.

Shrinking every cell by the same factor toward 0 leaves the standardized red(a)
unchanged under the ``squared`` and ``abs`` reductions. It matters for what reads
R's spectrum or its cells nonlinearly: ``diagnostics``, and the QPFS / DPP kernels.

Rows are handled like ``correlate``: a row with no variance is dropped (NaN row and
column, listed in ``dropped``), and the matrix is always (A, A).
"""

from typing import NamedTuple

import numpy as np

from .corr import _STD_EPS

__all__ = ["METHODS", "Shrunk", "ledoit_wolf", "schafer_strimmer", "shrink"]


class Shrunk(NamedTuple):
    """r (A, A) with NaN for dropped rows; dropped row indices; lam in [0, 1]; and
    the off-diagonal value the cells were pulled toward."""

    r: np.ndarray
    dropped: np.ndarray
    intensity: float
    target: float


def _kept(d_grad: np.ndarray):
    d_grad = np.asarray(d_grad, dtype=np.float64)
    if d_grad.ndim != 2:
        raise ValueError(f"G must be (A, N), got {d_grad.shape}")
    stds = d_grad.std(axis=1) if d_grad.shape[1] else np.zeros(d_grad.shape[0])
    return d_grad, np.flatnonzero(stds >= _STD_EPS), np.flatnonzero(stds < _STD_EPS)


def _assemble(n_ops: int, keep: np.ndarray, sub: np.ndarray, dropped, lam: float, target: float) -> Shrunk:
    r = np.full((n_ops, n_ops), np.nan)
    r[np.ix_(keep, keep)] = sub
    return Shrunk(r, dropped, float(lam), float(target))


def _pulled(r: np.ndarray, lam: float, target: float) -> np.ndarray:
    out = lam * target + (1.0 - lam) * r
    np.fill_diagonal(out, 1.0)
    return out


def ledoit_wolf(d_grad: np.ndarray) -> Shrunk:
    """Ledoit-Wolf shrinkage of R toward the identity."""
    d_grad, keep, dropped = _kept(d_grad)
    n_ops = d_grad.shape[0]
    if keep.size < 2:
        return _assemble(n_ops, keep, np.eye(keep.size), dropped, 0.0, 0.0)

    rows = d_grad[keep]
    x = (rows - rows.mean(axis=1, keepdims=True)) / rows.std(axis=1, keepdims=True)
    p, n = x.shape
    s = x @ x.T / n  # the sample correlation, with a unit diagonal
    # mu = trace(s) / p = 1 exactly, so the target mu * I is the identity.
    delta = ((s - np.eye(p)) ** 2).sum() / p
    # (1/n^2) sum_k ||x_k x_k' - s||_F^2 / p, expanded so no (p, p, n) array is built.
    beta = ((x**2).sum(axis=0) ** 2).sum() / n - (s**2).sum()
    beta = min(beta / (p * n), delta)
    lam = 0.0 if delta <= 0 else beta / delta
    return _assemble(n_ops, keep, _pulled(s, lam, 0.0), dropped, lam, 0.0)


def schafer_strimmer(d_grad: np.ndarray, target: float = 0.0) -> Shrunk:
    """Schafer-Strimmer shrinkage of R toward a constant off-diagonal ``target``."""
    if not -1.0 <= target <= 1.0:
        raise ValueError(f"target must be a correlation in [-1, 1], got {target}")
    d_grad, keep, dropped = _kept(d_grad)
    n_ops = d_grad.shape[0]
    n = d_grad.shape[1]
    if keep.size < 2 or n < 3:
        return _assemble(n_ops, keep, np.eye(keep.size), dropped, 0.0, target)

    rows = d_grad[keep]
    x = (rows - rows.mean(axis=1, keepdims=True)) / rows.std(axis=1, ddof=1, keepdims=True)
    w_bar = x @ x.T / n  # mean over images of x_ki * x_kj
    r = w_bar * n / (n - 1)
    # Var(r_ij) = n / (n-1)^3 * sum_k (w_kij - w_bar_ij)^2, the sum expanded.
    x2 = x**2
    var_r = n / (n - 1) ** 3 * (x2 @ x2.T - n * w_bar**2)

    off = ~np.eye(keep.size, dtype=bool)
    spread = ((r[off] - target) ** 2).sum()
    lam = 1.0 if spread <= 0 else float(np.clip(var_r[off].sum() / spread, 0.0, 1.0))
    return _assemble(n_ops, keep, _pulled(r, lam, target), dropped, lam, target)


METHODS = {"ledoit_wolf": ledoit_wolf, "schafer_strimmer": schafer_strimmer}


def shrink(d_grad: np.ndarray, method: str, target: float = 0.0) -> Shrunk:
    """``METHODS[method]`` on G. ``target`` is Schafer-Strimmer's; Ledoit-Wolf
    refuses a nonzero one rather than ignore it."""
    if method == "ledoit_wolf":
        if target != 0.0:
            raise ValueError("ledoit_wolf shrinks toward the identity; use schafer_strimmer for a target")
        return ledoit_wolf(d_grad)
    if method == "schafer_strimmer":
        return schafer_strimmer(d_grad, target)
    raise ValueError(f"unknown shrinkage {method!r}; expected one of {sorted(METHODS)}")
