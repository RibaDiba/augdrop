"""A sweep's ``skip``: the ops a run prunes get NaN rows labelled ``static_pruned``,
and analyze sets them aside whether or not they were measured (``prune.probe``)."""

import json

import numpy as np
import pytest

import stats
from probe import STATIC_PRUNED
from probe.frozen_sweep import Magnitudes, sweep
from probe.probe_set import ProbeSet


def _probe_set(dataset):
    return ProbeSet.build([item["id"] for item in dataset])


def _sampled():
    return Magnitudes(
        draw=lambda ops, b, rng: {name: rng.uniform(0.1, 0.9, b) for name in ops},
        source="test",
        mode="sampled",
    )


def test_skipped_ops_get_nan_rows_and_the_rest_is_unchanged(adapter, dataset, ops):
    probe_set = _probe_set(dataset)
    full = sweep(adapter, probe_set, dataset, ops, _sampled(), batch_size=2)
    pruned = sweep(adapter, probe_set, dataset, ops, _sampled(), batch_size=2, skip=["noise"])

    assert pruned.methods["noise"] == STATIC_PRUNED
    assert pruned.static_pruned == ["noise"] and "noise" not in pruned.unprobed
    G, _ = pruned.matrix()
    G_full, _ = full.matrix()
    names = list(ops)
    i = names.index("noise")
    assert np.all(np.isnan(G[i]))
    assert np.all(np.isnan(np.concatenate(pruned.magnitudes["noise"])))
    # Every other row, and its magnitudes, bit-identical to the sweep that measured noise.
    keep = [j for j in range(len(names)) if j != i]
    np.testing.assert_array_equal(G[keep], G_full[keep])
    for name in names:
        if name != "noise":
            np.testing.assert_array_equal(
                np.concatenate(pruned.magnitudes[name]), np.concatenate(full.magnitudes[name])
            )


def test_the_gradient_log_records_the_label(adapter, dataset, ops, tmp_path):
    log = tmp_path / "aug_gradient_log.txt"
    sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.3), batch_size=4, skip=["noise"], log_path=str(log))
    record = json.loads(log.read_text().splitlines()[0])
    assert record["methods"]["noise"] == STATIC_PRUNED
    assert all(v is None for v in record["grads"]["noise"])


@pytest.mark.parametrize("skip, match", [(["nope"], "not among the ops"), (None, "every op")])
def test_skip_refuses(adapter, dataset, ops, skip, match):
    skip = list(ops) if skip is None else skip
    with pytest.raises(ValueError, match=match):
        sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.3), skip=skip)


# --- analyze ------------------------------------------------------------------------


def _G(n_ops=5, n_images=150, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n_ops, n_images)) + rng.normal(size=n_images)
    return G, np.arange(n_images)


def test_analyze_labels_static_pruned_apart_from_unprobed():
    G, ids = _G()
    G[1] = np.nan  # pruned, not measured
    G[3] = np.nan  # no magnitude
    names = ["a", "b", "c", "d", "e"]
    analysis = stats.analyze(G, ids, names, static_pruned=["b"], bootstrap_reps=50)
    assert analysis.static_pruned == ["b"]
    assert analysis.unprobed == ["d"]
    assert np.all(np.isnan(analysis.primary[1]))
    assert stats.corr_matrix_record(analysis)["static_pruned"] == ["b"]


def test_a_measured_pruned_row_leaves_r_as_if_it_was_not():
    """prune.probe=true: the row is real, but the R a selector reads must not move."""
    G, ids = _G()
    names = ["a", "b", "c", "d", "e"]
    measured = stats.analyze(G, ids, names, static_pruned=["b"], bootstrap_reps=50)
    G_nan = G.copy()
    G_nan[1] = np.nan
    unmeasured = stats.analyze(G_nan, ids, names, static_pruned=["b"], bootstrap_reps=50)
    np.testing.assert_array_equal(measured.primary, unmeasured.primary)
    np.testing.assert_array_equal(measured.survives, unmeasured.survives)
    assert measured.static_pruned == unmeasured.static_pruned == ["b"]


def test_analyze_sweep_reads_the_label_off_the_sweep(adapter, dataset, ops):
    result = sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.3), batch_size=2, skip=["noise"])
    analysis = stats.analyze_sweep(result, n_min=1, bootstrap=False)
    assert analysis.static_pruned == ["noise"]
    analysis = stats.analyze_sweep(result, n_min=1, bootstrap=False, static_pruned=["darker_G"])
    assert analysis.static_pruned == ["darker_G", "noise"]  # in G's row order


def test_analyze_refuses_an_unknown_name():
    G, ids = _G()
    with pytest.raises(ValueError, match="not rows of G"):
        stats.analyze(G, ids, list("abcde"), static_pruned=["z"])
