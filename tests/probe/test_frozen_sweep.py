"""sweep / sweep_checkpoints: one frozen model state x the probe set -> one G.

Several of these are ported from Sensaug's tests/test_grad_hook.py (the sweep and
the magnitude handoff).
"""

import json

import numpy as np
import pytest
import torch

from probe.frozen_sweep import Magnitudes, strength_to_magnitude, sweep, sweep_checkpoints
from probe.grad_collect import ProbeError, collect_grads
from probe.probe_set import ProbeSet

from probe_helpers import Op, ToyAdapter, lighter_r, make_ops


def _probe_set(dataset):
    return ProbeSet.build([item["id"] for item in dataset])


def _sampled(names=None):
    """A sampled magnitude source: one uniform draw per image per op."""
    return Magnitudes(
        draw=lambda ops, b, rng: {name: rng.uniform(0.1, 0.9, b) for name in ops},
        source="test",
        mode="sampled",
    )


# --- the frozen frame ---------------------------------------------------------------


def test_visits_every_batch_exactly_once_in_probe_order(adapter, dataset, ops):
    """The frozen frame is only worth anything over the WHOLE probe set. Each chunk of
    G must be ITS batch's gradient, not the first batch repeated."""
    probe_set = _probe_set(dataset)
    result = sweep(adapter, probe_set, dataset, ops, Magnitudes.fixed(0.3), batch_size=3)

    assert result.image_ids == probe_set.chunks(3)
    assert result.n_images == len(dataset)
    loader = probe_set.loader(dataset, 3)
    for k, batch in enumerate(loader):
        drawn = {name: np.full(len(batch["id"]), 0.3) for name in ops}
        want = collect_grads(adapter, batch, ops, drawn)
        for name in ops:
            np.testing.assert_array_equal(result.grads[name][k], want[name])


def test_matrix_is_ops_by_images_with_batch_clusters(adapter, dataset, ops):
    result = sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.3), batch_size=3)
    G, batch_ids = result.matrix()

    assert G.shape == (len(ops), len(dataset))
    np.testing.assert_array_equal(batch_ids, [0, 0, 0, 1, 1, 1, 2, 2])
    assert result.ids() == sorted(item["id"] for item in dataset)


def test_leaves_the_model_exactly_as_it_found_it(dataset, ops):
    """Frozen frame: weights and BN running stats bit-identical, no parameter .grad
    to pollute a later optimizer step, and every module's train/eval flag restored,
    including a mix the caller set deliberately."""
    adapter = ToyAdapter()
    adapter.model.train()
    adapter.model[1].eval()  # a frozen BN inside a training model
    before = {k: v.clone() for k, v in adapter.model.state_dict().items()}

    sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.3), batch_size=2)

    for key, value in adapter.model.state_dict().items():
        assert torch.equal(value, before[key]), f"the sweep moved {key}"
    assert all(p.grad is None for p in adapter.model.parameters())
    assert adapter.model.training and adapter.model[0].training
    assert not adapter.model[1].training


def test_restores_the_rng_state(adapter, dataset, ops):
    before = torch.get_rng_state()
    sweep(adapter, _probe_set(dataset), dataset, ops, _sampled(), probe_seed=99)
    assert torch.equal(torch.get_rng_state(), before)


# --- magnitudes -------------------------------------------------------------------


def test_sampled_magnitudes_advance_across_batches(adapter, dataset, ops):
    """Seeded once per sweep, not per batch. Reseeding per batch would hand every
    batch the same draw, and at batch size 1 one magnitude for the whole set."""
    result = sweep(adapter, _probe_set(dataset), dataset, ops, _sampled(), batch_size=1)
    summary = result.magnitude_summary()["lighter_R"]
    assert summary["min"] < summary["max"]


def test_fixed_magnitudes_stay_constant(adapter, dataset, ops):
    result = sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed({n: 0.25 for n in ops}))
    for name, summary in result.magnitude_summary().items():
        assert summary["min"] == summary["max"] == 0.25, name


def test_a_fixed_mapping_must_cover_every_op(adapter, dataset, ops):
    with pytest.raises(ValueError, match="no fixed magnitude"):
        sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed({"lighter_R": 0.2}))


def test_strength_is_converted_into_each_ops_own_units():
    def op(lo, hi, identity):
        return Op("x", lighter_r, mag_range=(lo, hi), identity_at=identity)

    assert strength_to_magnitude(op(0.0, 1.0, 0.0), 0.3) == pytest.approx(0.3)  # Sensaug
    assert strength_to_magnitude(op(-30.0, 30.0, 0.0), 0.5) == pytest.approx(15.0)  # Rotate
    assert strength_to_magnitude(op(0.1, 1.9, 1.0), 0.5) == pytest.approx(1.45)  # enhance
    assert strength_to_magnitude(op(0.0, 256.0, 256.0), 0.25) == pytest.approx(192.0)  # Solarize
    assert strength_to_magnitude(op(4.0, 8.0, 8.0), 1.0) == pytest.approx(4.0)  # Posterize
    with pytest.raises(ValueError, match="strength"):
        strength_to_magnitude(op(0.0, 1.0, 0.0), 1.5)


def test_an_op_without_a_magnitude_is_a_nan_row_labelled_unprobed(adapter, dataset, ops):
    ops = {**ops, "equalize": Op("equalize", lambda x, m: x, mag_range=None, identity_at=None, grad_mode="none")}
    result = sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.relative(0.5))

    G, _ = result.matrix()
    assert G.shape == (len(ops), len(dataset))  # the shape doesn't depend on the vocab
    assert np.isnan(G[result.names.index("equalize")]).all()
    assert result.unprobed == ["equalize"]
    assert "equalize" not in result.magnitude_summary()


# --- failures -----------------------------------------------------------------------


def test_probe_error_is_never_skipped(adapter, dataset):
    """The blanket catch exists so a TRANSIENT failure costs one batch. A structurally
    broken gradient must not be downgraded into a skipped batch."""
    ops = {"explode": Op("explode", lambda x, m: x * (torch.as_tensor(m).reshape(-1, 1, 1, 1) / 0.0))}
    with pytest.raises(ProbeError):
        sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.5))


def test_a_transient_failure_costs_only_its_batch(adapter, dataset):
    probe_set = _probe_set(dataset)
    second_batch_first_id = probe_set.chunks(2)[1][0]
    marker = next(item["image"] for item in dataset if item["id"] == second_batch_first_id)

    def flaky(images, magnitude):
        if torch.equal(images[0].detach(), marker):
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return lighter_r(images, magnitude)

    ops = {"flaky": Op("flaky", flaky)}
    result = sweep(adapter, probe_set, dataset, ops, Magnitudes.fixed(0.5), batch_size=2)

    assert result.n_batches == 3
    assert [f["batch"] for f in result.failed_batches] == [1]
    assert result.failed_batches[0]["image_ids"] == list(probe_set.chunks(2)[1])
    assert set(result.ids()).isdisjoint(result.failed_batches[0]["image_ids"])
    assert result.matrix()[0].shape == (1, 6)


def test_failing_every_leading_batch_is_systematic_not_transient(adapter, dataset):
    def broken(images, magnitude):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")

    ops = {"broken": Op("broken", broken)}
    with pytest.raises(RuntimeError, match="systematic"):
        sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.5))


def test_a_bug_in_an_op_is_not_treated_as_transient(adapter, dataset):
    def buggy(images, magnitude):
        raise ValueError("label and image differ in size")

    ops = {"buggy": Op("buggy", buggy)}
    with pytest.raises(ValueError, match="differ in size"):
        sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.5))


def test_a_loader_out_of_probe_order_is_caught(dataset, ops):
    class Lying(ToyAdapter):
        def batch_ids(self, batch):
            return list(reversed(batch["id"]))

    with pytest.raises(ProbeError, match="out of probe order"):
        sweep(Lying(), _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.3), batch_size=2)


def test_the_first_batch_is_checked_for_coupling(adapter, dataset):
    ops = {"pooled": Op("pooled", lambda x, m: x * (1 - 0.5 * torch.as_tensor(m).mean()))}
    with pytest.raises(ProbeError, match="couples the images"):
        sweep(adapter, _probe_set(dataset), dataset, ops, Magnitudes.fixed(0.3), batch_size=2)


# --- the log --------------------------------------------------------------------------


def test_logs_every_batch_per_image(adapter, dataset, ops, tmp_path):
    """The log is what makes G recomputable offline. numpy arrays aren't JSON, and
    NaN isn't valid JSON either, so both have to be converted on the way out."""
    ops = {**ops, "equalize": Op("equalize", lambda x, m: x, mag_range=None, identity_at=None, grad_mode="none")}
    log_path = tmp_path / "aug_gradient_log.txt"
    probe_set = _probe_set(dataset)
    sweep(
        adapter, probe_set, dataset, ops, Magnitudes.fixed(0.3),
        batch_size=3, checkpoint="iter_4000.pth", log_path=str(log_path), meta={"run": "a"},
    )

    lines = log_path.read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 3
    assert "NaN" not in log_path.read_text()
    for k, record in enumerate(records):
        assert record["checkpoint"] == "iter_4000.pth"
        assert record["batch"] == k
        assert record["image_ids"] == list(probe_set.chunks(3)[k])
        assert record["probe_set"] == probe_set.digest
        assert record["magnitude_source"] == "fixed" and record["magnitude_mode"] == "fixed"
        assert record["meta"] == {"run": "a"}
        assert record["methods"]["lighter_R"] == "autograd"
        assert record["methods"]["equalize"] == "unprobed"
        assert len(record["grads"]["lighter_R"]) == record["batch_size"]
        assert record["grads"]["equalize"] == [None] * record["batch_size"]
        assert record["magnitudes"]["lighter_R"] == [0.3] * record["batch_size"]


# --- checkpoints ----------------------------------------------------------------------


def test_one_g_per_checkpoint_at_identical_magnitudes(dataset, ops):
    """Checkpoints differ by their weights alone: same images, same order, same
    (sampled) magnitudes. And nothing pools them."""
    probe_set = _probe_set(dataset)
    loaded = []

    def load_adapter(checkpoint):
        loaded.append(checkpoint)
        return ToyAdapter(seed=checkpoint)

    sweeps = sweep_checkpoints([0, 1], load_adapter, probe_set, dataset, ops, _sampled(), batch_size=2)

    assert loaded == [0, 1]
    assert [s.checkpoint for s in sweeps] == ["0", "1"]
    a, b = sweeps
    assert a.ids() == b.ids()
    for name in ops:
        np.testing.assert_array_equal(np.concatenate(a.magnitudes[name]), np.concatenate(b.magnitudes[name]))
    assert not np.allclose(a.matrix()[0], b.matrix()[0])
