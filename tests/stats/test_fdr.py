"""Tests for stats/fdr.py, ported from Sensaug's tests/test_grad_sens_analysis.py."""

import numpy as np

from stats.fdr import cluster_bootstrap_ci, fdr_correct

# Sensaug's DIFF32 vocabulary, the default axes of R.
N_OPS = 32


def test_cluster_bootstrap_is_wider_than_a_naive_column_bootstrap():
    """Columns inside one probe share a batch and a model state, so they are not
    independent draws. Resampling COLUMNS pretends there are N_images independent
    observations and returns intervals that are too tight; resampling whole PROBES
    tells the truth. Same function, different cluster ids."""
    rng = np.random.default_rng(0)
    n_probes, batch = 20, 8
    probe_ids = np.repeat(np.arange(n_probes), batch)

    # A probe-level effect shared by every column of that probe -- precisely the
    # clustering a column bootstrap ignores.
    probe_effect = rng.normal(size=(2, n_probes)).repeat(batch, axis=1)
    d_grad = probe_effect + rng.normal(scale=0.3, size=(2, n_probes * batch))

    lo_probe, hi_probe, _ = cluster_bootstrap_ci(d_grad, probe_ids, n_reps=400)
    lo_col, hi_col, _ = cluster_bootstrap_ci(
        d_grad, np.arange(probe_ids.size), n_reps=400
    )

    assert (hi_probe[0, 1] - lo_probe[0, 1]) > (hi_col[0, 1] - lo_col[0, 1])


def test_bootstrap_ci_narrows_as_more_probes_are_collected():
    def ci_width(n_probes):
        rng = np.random.default_rng(0)
        n_images = n_probes * 4
        d_grad = rng.normal(size=(2, n_images))
        d_grad[1] = 0.6 * d_grad[0] + 0.8 * rng.normal(size=n_images)
        lo, hi, _ = cluster_bootstrap_ci(
            d_grad, np.repeat(np.arange(n_probes), 4), n_reps=300
        )
        return hi[0, 1] - lo[0, 1]

    assert ci_width(120) < ci_width(15)


def test_only_a_genuinely_correlated_pair_survives_fdr():
    """Hundreds of pairwise cells means several will look significant by chance. BH
    is what stops a pruning decision resting on one of them."""
    rng = np.random.default_rng(0)
    n_probes, batch = 40, 8
    n_images = n_probes * batch
    probe_ids = np.repeat(np.arange(n_probes), batch)

    d_grad = rng.normal(size=(N_OPS, n_images))
    d_grad[9] = 0.9 * d_grad[3] + np.sqrt(1 - 0.81) * rng.normal(size=n_images)

    _, _, p = cluster_bootstrap_ci(d_grad, probe_ids, n_reps=500)
    q, survives = fdr_correct(p, alpha=0.05)

    assert survives[3, 9], "the injected redundant pair should survive FDR"
    assert q[3, 9] < 0.05
    upper = np.triu_indices(N_OPS, k=1)
    assert survives[upper].sum() <= 3, "too many null pairs slipped through BH"


def test_fdr_is_symmetric_and_ignores_the_diagonal():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=(6, 6))
    p = (p + p.T) / 2

    q, survives = fdr_correct(p)

    np.testing.assert_allclose(q, q.T, equal_nan=True)
    assert np.isnan(np.diag(q)).all()
    assert not np.diag(survives).any()
