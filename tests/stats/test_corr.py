"""Tests for stats/corr.py, ported from Sensaug's tests/test_grad_sens_analysis.py."""

import numpy as np
import pytest

from stats.corr import closure_null, correlate, shared_factor_loadings, shared_image_factor

# Sensaug's DIFF32 vocabulary, the default axes of R.
N_OPS = 32


def _confounded_grid(n_ops=N_OPS, n_images=800, seed=0):
    """Ops that are genuinely INDEPENDENT of one another, observed through a
    shared per-image scale.

    Two facts drive this: augmentation sensitivities are predominantly positive
    (perturbing an image almost always increases its loss), and a hard image has a
    big gradient for EVERY op. So a shared per-image factor multiplies rows that
    have a positive mean -- which manufactures correlation out of nothing.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=1.0, scale=0.15, size=(n_ops, n_images))
    difficulty = rng.uniform(0.5, 5.0, size=n_images)
    return base * difficulty


# --- correlation + guards ----------------------------------------------------


def test_correlate_recovers_identical_and_independent_rows():
    rng = np.random.default_rng(0)
    d_grad = rng.normal(size=(4, 500))
    d_grad[1] = d_grad[0] + 1e-6 * rng.normal(size=500)  # a redundant pair

    r, dropped = correlate(d_grad)

    assert dropped.size == 0
    assert r[0, 1] == pytest.approx(1.0, abs=1e-3)
    assert r[2, 3] == pytest.approx(0.0, abs=0.15)


def test_correlate_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(0)
    r, _ = correlate(rng.normal(size=(N_OPS, 300)))

    assert r.shape == (N_OPS, N_OPS)
    np.testing.assert_allclose(r, r.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(r), np.ones(N_OPS), atol=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_correlate_drops_a_constant_row_instead_of_emitting_a_silent_nan():
    """An op with no variance over the window has no correlation signal at all.
    np.corrcoef would divide by its zero std, emit a RuntimeWarning, and quietly
    scatter NaN through the matrix. Drop it explicitly instead, and make sure the
    kept submatrix comes back clean (filterwarnings turns the warning into a
    failure)."""
    rng = np.random.default_rng(0)
    d_grad = rng.normal(size=(4, 200))
    d_grad[2] = 0.7  # this op never moves

    r, dropped = correlate(d_grad)

    np.testing.assert_array_equal(dropped, [2])
    assert np.isnan(r[2]).all()  # whole row, diagonal included
    assert np.isnan(r[:, 2]).all()
    keep = [0, 1, 3]
    assert np.isfinite(r[np.ix_(keep, keep)]).all()


# --- the image-difficulty confound -------------------------------------------


def test_shared_image_scale_inflates_raw_r_and_normalization_removes_it():
    """The failure mode this whole normalization exists for.

    Correlating raw per-image gradients does not measure redundancy -- it measures
    image difficulty. These ops are independent by construction, yet raw R calls
    almost every pair redundant, for exactly the same reason the old
    correlate-over-training-time R did: a factor shared by every row.
    """
    d_grad = _confounded_grid()
    off_diagonal = np.triu_indices(N_OPS, k=1)

    r_raw, _ = correlate(d_grad)
    shared = shared_image_factor(d_grad)
    r_norm, _ = correlate(d_grad / (shared + 1e-12))

    # Raw R: "everything is redundant." Nothing is.
    assert r_raw[off_diagonal].mean() > 0.5

    # Normalized R: no structure, correctly. It sits on the closure baseline
    # rather than exactly 0 -- see closure_null.
    assert r_norm[off_diagonal].mean() == pytest.approx(closure_null(N_OPS), abs=0.05)


def test_shared_factor_loadings_expose_the_confound():
    """The headline diagnostic: if every op loads ~1.0 on the shared per-image
    factor, R is reading which images are hard, and nothing built on it is
    trustworthy."""
    d_grad = _confounded_grid()

    loadings = shared_factor_loadings(d_grad, shared_image_factor(d_grad))

    assert loadings.shape == (N_OPS,)
    assert (loadings > 0.9).all()


def test_normalization_preserves_a_real_correlation():
    """The normalization must remove the shared factor WITHOUT flattening genuine
    redundancy -- otherwise it would just be destroying the signal."""
    rng = np.random.default_rng(1)
    n_images = 800
    base = rng.normal(loc=1.0, scale=0.15, size=(N_OPS, n_images))
    base[5] = base[2] + 0.05 * rng.normal(size=n_images)  # ops 2 and 5 really are alike
    d_grad = base * rng.uniform(0.5, 5.0, size=n_images)

    shared = shared_image_factor(d_grad)
    r_norm, _ = correlate(d_grad / (shared + 1e-12))

    assert r_norm[2, 5] > 0.8
    others = [i for i in range(N_OPS) if i not in (2, 5)]
    assert abs(r_norm[np.ix_(others, others)][np.triu_indices(len(others), 1)].mean()) < 0.2


def test_closure_null_is_the_baseline_not_zero():
    assert closure_null(14) == pytest.approx(-1.0 / 13)
    assert closure_null(1) == 0.0
