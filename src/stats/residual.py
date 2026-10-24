"""R with the shared per-image factor regressed out, then rank-correlated: the R the
NMS arm (``selection.nms``) prunes on.

Sensaug's R_scalenorm divides each image's gradients by the mean |g| over every op
(``corr.shared_image_factor``). That normalizer is estimated from the rows it divides,
which closes the composition (an independent pair sits at ``closure_null``, -1/(A-1),
not 0), and it puts each op's own magnitude inside its own normalizer. Here the factor
is regressed out instead, and each op is left out of its own factor:

    Z_a     = G_a / median_i |G_ai|                     each op on a robust scale of its own
    h_a,i   = (sum_b |Z_bi| - |Z_ai|) / (A - 1)          the OTHER ops' mean |Z| on image i
    E_a     = the residual of Z_a regressed (OLS, with intercept) on h_a
    R_ab    = Spearman(E_a, E_b) across images

An independent pair then sits close to 0, though not at it: each op is still inside
every OTHER op's factor, which pulls a pair down by about as much as R_scalenorm's
closure null (-1/(A-1)) on synthetic G. The mean off-diagonal cell is the health check:
far from 0, a shared factor got through. Ranks keep the few images with huge gradients
from deciding a cell, which they do in a Pearson correlation.

No gate is attached. ``null_max_bound`` says how high the largest cell climbs by chance
alone, the yardstick a fixed cutoff is read against.

Pure numpy (and scipy's rankdata). Missing data is NaN, as everywhere in ``stats``.
"""

import math
from typing import NamedTuple

import numpy as np
from scipy.stats import norm, rankdata

from .corr import _DIV_EPS, _STD_EPS, correlate

__all__ = ["Residuals", "null_max_bound", "residual_rank_corr", "residualize"]


class Residuals(NamedTuple):
    """What ``residualize`` returns, over the rows of the G it was given.

    e         (A, N) residuals E. NaN rows for the ``unusable`` ops.
    slopes    (A,) each op's OLS slope on its leave-one-out factor h: how strongly it
              tracked image difficulty. NaN for the ``unusable`` ops.
    unusable  indices of the rows with no residual: median |G| = 0 (no scale to divide
              by), or every row when fewer than two can be scaled
    """

    e: np.ndarray
    slopes: np.ndarray
    unusable: np.ndarray


def residualize(d_grad: np.ndarray) -> Residuals:
    """E over a (A, N) G of finite rows: each op scaled by its median |g|, minus its
    least-squares fit on the other ops' mean |Z| per image.

    A row whose median |g| is 0 can't be scaled, and is left out of every other op's
    factor too. With fewer than two scalable rows there is no "other" op, and every row
    is NaN."""
    g = np.asarray(d_grad, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"G must be (A, N), got {g.shape}")
    n_ops, n_images = g.shape
    e = np.full_like(g, np.nan)
    slopes = np.full(n_ops, np.nan)
    scale = np.median(np.abs(g), axis=1) if n_images else np.zeros(n_ops)
    scaled = np.flatnonzero(scale > _DIV_EPS)
    if scaled.size < 2:
        return Residuals(e, slopes, np.arange(n_ops))

    z = g[scaled] / scale[scaled, None]
    magnitude = np.abs(z)
    h = (magnitude.sum(axis=0) - magnitude) / (scaled.size - 1)
    h_c = h - h.mean(axis=1, keepdims=True)
    z_c = z - z.mean(axis=1, keepdims=True)
    # A factor with no spread across images explains nothing: slope 0.
    flat = h.std(axis=1) < _STD_EPS
    beta = np.where(
        flat, 0.0, (h_c * z_c).sum(axis=1) / np.where(flat, 1.0, (h_c**2).sum(axis=1))
    )
    e[scaled] = z_c - beta[:, None] * h_c
    slopes[scaled] = beta
    return Residuals(e, slopes, np.setdiff1d(np.arange(n_ops), scaled))


def residual_rank_corr(d_grad: np.ndarray):
    """``(R, dropped, slopes)``: the Spearman correlation of ``residualize``'s E across
    images, (A, A) over G's rows. ``dropped`` are the ops with no usable row (no scale,
    or no variance left after the regression), NaN in R as ``correlate`` leaves them."""
    e, slopes, unusable = residualize(d_grad)
    ranks = np.full_like(e, np.nan)
    rows = np.flatnonzero(np.isfinite(e).all(axis=1))
    if rows.size and e.shape[1]:
        # Average ranks for ties, so tied images don't get an arbitrary order.
        ranks[rows] = rankdata(e[rows], axis=1)
    r, no_variance = correlate(ranks)
    # correlate neither keeps nor drops a NaN row; those are the unusable ones.
    dropped = np.union1d(unusable, no_variance).astype(int)
    return r, dropped, slopes


def null_max_bound(n_ops: int, n_images: int, alpha: float = 0.05) -> float:
    """How high the largest of the A(A-1)/2 off-diagonal cells of R climbs by chance
    alone: a (1 - alpha) union bound on it when no pair is related.

    Each cell is one Spearman correlation, whose null distribution has mean 0 and
    variance exactly 1/(N - 1), and is close to normal at the probe's N. The union bound
    holds whatever the dependence between cells. One-sided, because only a high cell
    suppresses an op. At A = 32 it is 0.37 at N = 100 and 0.17 at N = 500 (FACTS D8). A
    cutoff below it prunes on noise."""
    pairs = n_ops * (n_ops - 1) // 2
    if pairs < 1:
        return float("nan")
    if n_images < 2:
        return 1.0
    return min(1.0, float(norm.isf(alpha / pairs)) / math.sqrt(n_images - 1))
