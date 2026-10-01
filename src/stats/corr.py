"""R: the Pearson correlation of G's rows, taken across images.

Ported from Sensaug's ``sensaug/hooks/grad_sens_analysis.py`` (its pure-numpy
functions; the mmengine hook that drove them is not ported).

R is correlated across IMAGES, not across training time. The probe stores one
``d loss / d magnitude`` value PER IMAGE per op, so G is an (A ops x N images) matrix
whose rows can be correlated to answer "do these two augmentations hurt the SAME
images". That is what redundancy means. Correlating across probes instead (one
batch-averaged number per probe) would only ask "do these two augmentations get less
sensitive at the same MOMENTS during training". Nearly every op's sensitivity drifts
down together as the model converges, so that reports almost everything as redundant.
It measures convergence drift, not redundancy.

The rows also carry a shared per-image scale factor. Hard images have a large
gradient for EVERY op, and mmseg's ignore_index weighting adds a per-image
valid-pixel factor on top. Pearson is invariant to a global scale but not to a
per-column one, so that factor would re-inflate R just as convergence drift did.
Divide it out (``G / (shared_image_factor(G) + _DIV_EPS)``, Sensaug's
``R_scalenorm``) and log ``shared_factor_loadings`` with every R, so you can see how
much of R it was accounting for.

R is built from one G, never pooled across sweeps: a warmup model and a converged
model are different measurements, and pooling groups with different underlying
structure manufactures trends neither group has (Simpson's paradox).
"""

import numpy as np

# A row whose sensitivity never varies across the window carries no correlation
# signal; np.corrcoef would divide by its zero std and emit a silent NaN row.
_STD_EPS = 1e-12
# Guards the per-image scale division.
_DIV_EPS = 1e-12
# Above this, the shared image-difficulty factor is eating R.
_LOADING_ALARM = 0.9


def shared_image_factor(d_grad: np.ndarray) -> np.ndarray:
    """The per-image mean absolute sensitivity: the empirical "image difficulty"
    factor that every op loads on (hard image -> big gradient for everything)."""
    return np.abs(d_grad).mean(axis=0)


def shared_factor_loadings(d_grad: np.ndarray, shared: np.ndarray) -> np.ndarray:
    """corr(each op's row, the shared factor).

    THE diagnostic for this matrix: if these are all ~0.9+, R is measuring which
    images are hard, not which augmentations are redundant, and nothing
    downstream of it is trustworthy.
    """
    loadings = np.full(d_grad.shape[0], np.nan)
    if shared.size < 2 or shared.std() < _STD_EPS:
        return loadings
    for i, row in enumerate(d_grad):
        if row.std() >= _STD_EPS:
            loadings[i] = np.corrcoef(row, shared)[0, 1]
    return loadings


def correlate(d_grad: np.ndarray):
    """Pearson correlation across the rows of a (A, N) matrix, with the
    zero-variance rows dropped rather than silently NaN'd.

    Returns (R, dropped_idx). R is always (A, A); a dropped op's row and column
    (diagonal included) are NaN, so a dropped op can never be mistaken for an
    uncorrelated one.
    """
    n_ops = d_grad.shape[0]
    r = np.full((n_ops, n_ops), np.nan)

    stds = d_grad.std(axis=1) if d_grad.shape[1] else np.zeros(n_ops)
    keep = np.flatnonzero(stds >= _STD_EPS)
    dropped = np.flatnonzero(stds < _STD_EPS)

    if keep.size >= 2:
        # Only the kept rows go in, so corrcoef never divides by zero and never
        # emits a RuntimeWarning.
        sub = np.corrcoef(d_grad[keep])
        assert np.isfinite(sub).all(), "corrcoef produced a non-finite kept cell"
        r[np.ix_(keep, keep)] = sub

    return r, dropped


def closure_null(n_ops: int) -> float:
    """The baseline correlation induced purely by the per-image normalization.

    Dividing every row by a normalizer ESTIMATED FROM THOSE SAME ROWS closes the
    composition: the A normalized values in a column are tied together, which
    pushes the pairwise correlations to about -1/(A-1) even when the underlying
    ops are perfectly independent. At A=14 that is -0.077.

    Small, but it is the null: cells must be judged against THIS, not against
    zero, or every mildly negative cell reads as significant. (An external
    normalizer -- e.g. the clean per-image loss, one extra forward per probe --
    would avoid the artifact entirely; that is the clean fix if -0.077 ever
    starts to matter.)
    """
    return -1.0 / (n_ops - 1) if n_ops > 1 else 0.0
