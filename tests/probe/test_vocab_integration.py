"""The probe over the real vocab/ ops, end to end on the toy model."""

import numpy as np
import pytest

pytest.importorskip("kornia")
pytest.importorskip("torchvision")

from vocab import build_vocab  # noqa: E402

from probe import Magnitudes, ProbeSet, sweep  # noqa: E402
from probe_helpers import ToyAdapter, make_dataset  # noqa: E402


def _sweep(ops, **kwargs):
    dataset = make_dataset(n=4, size=32)
    probe_set = ProbeSet.build([item["id"] for item in dataset])
    return sweep(ToyAdapter(), probe_set, dataset, ops, Magnitudes.relative(0.5), batch_size=2, **kwargs)


def test_default_vocab_is_probed_by_autograd_in_sensaug_order():
    ops = build_vocab()
    result = _sweep(ops)
    G, _ = result.matrix()

    assert result.names == list(ops)
    assert G.shape == (len(ops), 4)
    assert np.isfinite(G).all()
    assert set(result.methods.values()) == {"autograd"}


def test_torchvision_family_falls_back_to_finite_differences():
    ops = build_vocab({"families": ["randaug_default"]})
    result = _sweep(ops, fd_step=0.1)
    G, _ = result.matrix()

    for name, row in zip(result.names, G):
        if ops[name].mag_range is None:
            assert result.methods[name] == "unprobed" and np.isnan(row).all(), name
        else:
            assert result.methods[name] == "finite_diff" and np.isfinite(row).all(), name
