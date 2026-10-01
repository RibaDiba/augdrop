"""Per-cell uncertainty for R: cluster-bootstrap CIs and p-values, then FDR control
over the off-diagonal cells.

Ported from Sensaug's ``sensaug/hooks/grad_sens_analysis.py``. The clusters are
batches: ``probe_ids`` here is the ``batch_ids`` that ``probe.Sweep.matrix()``
returns (Sensaug called one batch a "probe").

Added here: Benjamini-Yekutieli as a second ``fdr_correct`` method, and the Fisher z
interval (``fisher_z_ci``), the analytic reference for a single cell.
"""

import warnings

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from .corr import correlate


def cluster_bootstrap_ci(
    d_grad: np.ndarray,
    probe_ids: np.ndarray,
    n_reps: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    null: float = 0.0,
):
    """Bootstrap CIs for every cell of R by resampling whole PROBES.

    Resampling probes, not individual image columns, is the point. Columns within
    a probe come from one batch scored by one model state, so they are not
    independent draws; a naive column bootstrap treats N_images as the sample
    size, understates the variance, and hands back intervals that are too tight.

    Returns (ci_lo, ci_hi, p). The interval is the percentile interval over the
    replicates. The p-value tests r == `null` (0 for a raw R, closure_null(A) for
    a per-image-normalized one) via a normal approximation on the bootstrap
    standard error.

    The p-value deliberately does NOT come from the replicate percentiles. A
    percentile p-value cannot go below 1/n_reps, and Benjamini-Hochberg over the
    91 pairwise cells needs p <= alpha/91 ~ 5.5e-4 for the very first rejection:
    at n_reps=1000 the floor is 1e-3, so nothing could EVER survive FDR and the
    correction would silently reject every cell. The bootstrap SE has no such
    floor and still carries the clustering.
    """
    rng = np.random.default_rng(seed)
    probes = np.unique(probe_ids)
    columns_by_probe = [np.flatnonzero(probe_ids == p) for p in probes]
    n_ops = d_grad.shape[0]

    observed, _ = correlate(d_grad)

    reps = np.full((n_reps, n_ops, n_ops), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(n_reps):
            picks = rng.integers(0, len(columns_by_probe), size=len(columns_by_probe))
            columns = np.concatenate([columns_by_probe[i] for i in picks])
            reps[k], _ = correlate(d_grad[:, columns])

        ci_lo = np.nanpercentile(reps, 100 * alpha / 2, axis=0)
        ci_hi = np.nanpercentile(reps, 100 * (1 - alpha / 2), axis=0)
        standard_error = np.nanstd(reps, axis=0)

    z = (observed - null) / np.maximum(standard_error, 1e-12)
    p = 2 * norm.sf(np.abs(z))
    return ci_lo, ci_hi, p


#: ``fdr_correct`` methods -> statsmodels' names.
FDR_METHODS = {"bh": "fdr_bh", "by": "fdr_by"}


def fdr_correct(p: np.ndarray, alpha: float = 0.05, method: str = "bh"):
    """FDR control across the A*(A-1)/2 unique off-diagonal cells.

    With 14 ops that is 91 simultaneous tests, several of which will look
    significant by chance -- correcting for that is what stops a pruning decision
    resting on noise. Returns (q, survives), both (A, A) and symmetric.

    ``"bh"`` (Benjamini-Hochberg, Sensaug's) controls the FDR when the tests are
    independent or positively dependent. The cells of R are neither, strictly: every
    cell in a row shares that op's gradients. ``"by"`` (Benjamini-Yekutieli 2001)
    holds under any dependence, at the price of dividing alpha by
    sum_{i<=m} 1/i (5.1 at 91 cells, 6.8 at 496).
    """
    if method not in FDR_METHODS:
        raise ValueError(f"unknown FDR method {method!r}, expected one of {sorted(FDR_METHODS)}")
    n_ops = p.shape[0]
    upper = np.triu_indices(n_ops, k=1)
    flat = p[upper]

    q_flat = np.full(flat.shape, np.nan)
    survives_flat = np.zeros(flat.shape, dtype=bool)
    finite = np.isfinite(flat)
    if finite.any():
        rejected, q_values, _, _ = multipletests(
            flat[finite], alpha=alpha, method=FDR_METHODS[method]
        )
        q_flat[finite] = q_values
        survives_flat[finite] = rejected

    q = np.full((n_ops, n_ops), np.nan)
    survives = np.zeros((n_ops, n_ops), dtype=bool)
    q[upper] = q_flat
    q.T[upper] = q_flat
    survives[upper] = survives_flat
    survives.T[upper] = survives_flat
    return q, survives


def fisher_z_se(n) -> np.ndarray:
    """The standard error of atanh(r) for a correlation over ``n`` independent
    bivariate-normal pairs: 1 / sqrt(n - 3)."""
    n = np.asarray(n, dtype=np.float64)
    if np.any(n <= 3):
        raise ValueError(f"the Fisher z standard error needs n > 3, got {n}")
    return 1.0 / np.sqrt(n - 3.0)


def fisher_z_ci(r: np.ndarray, n, alpha: float = 0.05, null: float = 0.0):
    """Per-cell (ci_lo, ci_hi, p) from Fisher's z transform, shaped like
    ``cluster_bootstrap_ci``'s output.

    It treats the N images as N independent draws, which is what the bootstrap
    refuses to do: columns of one batch share a model state. So this is the
    reference, not the test. At batch size 1 the two should agree; the ratio of the
    bootstrap SE to ``fisher_z_se(n)`` is the design effect of the clustering. Pass
    an effective sample size as ``n`` to use it anyway. ``n`` may be per cell.
    """
    r = np.asarray(r, dtype=np.float64)
    se = fisher_z_se(n)
    # atanh(+-1) is infinite; the diagonal and exact aliases land here.
    z = np.arctanh(np.clip(r, -1.0 + 1e-15, 1.0 - 1e-15))
    half = norm.ppf(1.0 - alpha / 2.0) * se
    ci_lo = np.tanh(z - half)
    ci_hi = np.tanh(z + half)
    p = 2.0 * norm.sf(np.abs(z - np.arctanh(null)) / se)
    return ci_lo, ci_hi, p
