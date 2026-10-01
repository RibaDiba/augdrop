import numpy as np
import pytest

from probe import magnitude_sweep
from probe.magnitude_sweep import sweep_magnitudes
from probe.probe_set import ProbeSet

from probe_helpers import Op, enhance


def _probe_set(dataset):
    return ProbeSet.build([item["id"] for item in dataset])


def _mixed_units(ops):
    """Sensaug's [0, 1] ops plus one in its own units (an enhance factor)."""
    return {**ops, "brightness": Op("brightness", enhance, mag_range=(0.1, 1.9), identity_at=1.0)}


def test_one_labelled_g_per_strength(adapter, dataset, ops):
    ops = _mixed_units(ops)
    sweeps = sweep_magnitudes(adapter, _probe_set(dataset), dataset, ops, [0.2, 0.6], batch_size=4)

    assert [s.meta["reference"] for s in sweeps] == [0.2, 0.6]
    assert all(s.meta["reference_kind"] == "strength" for s in sweeps)
    assert all(s.magnitude_source == "reference" for s in sweeps)
    # Converted into each op's own units, constant across images.
    for sweep, strength in zip(sweeps, [0.2, 0.6]):
        summary = sweep.magnitude_summary()
        assert summary["lighter_R"]["min"] == summary["lighter_R"]["max"] == pytest.approx(strength)
        assert summary["brightness"]["mean"] == pytest.approx(1.0 + 0.9 * strength)
    # Never pooled: two separate Gs, and the magnitude changes them.
    assert not np.allclose(sweeps[0].matrix()[0], sweeps[1].matrix()[0])


def test_an_absolute_reference_uses_the_ops_own_units(adapter, dataset, ops):
    reference = {name: 0.4 for name in ops}
    (result,) = sweep_magnitudes(adapter, _probe_set(dataset), dataset, ops, [reference])
    assert result.meta == {"reference": reference, "reference_kind": "absolute"}
    assert result.magnitude_summary()["noise"]["mean"] == pytest.approx(0.4)


def test_meta_is_merged_into_every_label(adapter, dataset, ops):
    sweeps = sweep_magnitudes(adapter, _probe_set(dataset), dataset, ops, [0.3], meta={"run": "e3"})
    assert sweeps[0].meta == {"run": "e3", "reference": 0.3, "reference_kind": "strength"}


def test_every_reference_is_checked_before_anything_runs(adapter, dataset, ops, monkeypatch):
    """A bad third reference must fail in seconds, not after two full sweeps."""
    ran = []
    monkeypatch.setattr(magnitude_sweep, "sweep", lambda *a, **k: ran.append(1))

    with pytest.raises(ValueError, match="strength"):
        sweep_magnitudes(adapter, _probe_set(dataset), dataset, ops, [0.2, 0.5, 1.5])
    with pytest.raises(ValueError, match="no magnitude for op"):
        sweep_magnitudes(adapter, _probe_set(dataset), dataset, ops, [{"lighter_R": 0.2}])
    with pytest.raises(ValueError, match="outside"):
        sweep_magnitudes(adapter, _probe_set(dataset), dataset, ops, [{n: 2.0 for n in ops}])
    with pytest.raises(ValueError, match="no reference"):
        sweep_magnitudes(adapter, _probe_set(dataset), dataset, ops, [])
    assert ran == []
