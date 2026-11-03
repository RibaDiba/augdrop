"""The probe on the real vocab, every op of every family, end to end.

The rest of tests/probe uses toy ops that mirror the OpSpec contract, so the
probe's own tests need no kornia or torchvision. This file checks that the real
vocab keeps that contract: each op is measured the way its spec says (autograd,
finite differences, or a NaN "unprobed" row), and the probe's runtime guards
(label moved, deterministic input, per-image independence) pass on it.
"""

import numpy as np
import pytest

vocab = pytest.importorskip("vocab")

from probe import collect_grads, strength_to_magnitude  # noqa: E402
from probe.grad_collect import AUTOGRAD, FINITE_DIFF, TANGENT, UNPROBED, op_method  # noqa: E402
from probe_helpers import ToyAdapter, make_batch  # noqa: E402
from vocab.audit import _strongest  # noqa: E402

EXPECTED_METHOD = {
    "legacy_hsv": AUTOGRAD,
    "geometric": AUTOGRAD,
    "pil_autoaug": AUTOGRAD,
    "randaug_default": FINITE_DIFF,
    "trivialaug_wide": FINITE_DIFF,
    "trivialaug_diff": AUTOGRAD,
    "randaug_diff": AUTOGRAD,
    "augmix_diff": AUTOGRAD,
    "smartaug_diff": AUTOGRAD,
    "autoaugment": UNPROBED,
    "madaug": FINITE_DIFF,
    "faster_aa": AUTOGRAD,
}
# The split families: each keeps four torchvision ops as they are.
DIFF_FAMILIES = {"trivialaug_diff", "randaug_diff", "augmix_diff", "smartaug_diff"}
MAGNITUDE_FREE = {"Identity", "AutoContrast", "Equalize", "Invert", "Flip", "HorizontalFlip"}
# torchvision's enhance ops carry an exact tangent.
ENHANCE = {"Brightness", "Color", "Contrast", "Sharpness"}
# The split families keep these as torchvision's own ops: no tangent, finite differences.
EXACT_STEPPED = {"Posterize", "Solarize"}
# faster_aa's ops that step with the magnitude (dda gives them a straight-through
# gradient in the search; the probe has none).
FASTER_AA_STEPPED = {"Posterize", "Solarize", "Cutout"}


def test_audit_and_probe_agree_on_the_strongest_magnitude():
    for name, spec in vocab.OPS.items():
        if spec.has_magnitude:
            assert _strongest(spec) == strength_to_magnitude(spec, 1.0), name


def test_every_op_is_measured_the_way_its_spec_says():
    for name, spec in vocab.OPS.items():
        base = name.split(".")[-1]
        expected = EXPECTED_METHOD[spec.family]
        if base in MAGNITUDE_FREE or spec.couples_images:
            expected = UNPROBED
        elif spec.family == "faster_aa" and base in FASTER_AA_STEPPED:
            expected = FINITE_DIFF
        elif spec.family in DIFF_FAMILIES and base in EXACT_STEPPED:
            expected = FINITE_DIFF
        elif expected == FINITE_DIFF and base in ENHANCE:
            expected = TANGENT
        assert op_method(spec) == expected, name


def test_probe_runs_on_every_vocab_op():
    ops = vocab.OPS
    batch = make_batch(n=4, size=40)
    magnitudes = {
        name: strength_to_magnitude(spec, 0.5) for name, spec in ops.items() if spec.has_magnitude
    }
    rows = collect_grads(ToyAdapter(), batch, ops, magnitudes, check=True)

    assert list(rows) == list(ops)
    for name, row in rows.items():
        assert row.shape == (4,), name
        if op_method(ops[name]) == UNPROBED:
            assert np.isnan(row).all(), name
        else:
            assert np.isfinite(row).all(), name
    # Autograd is exact, so a smooth op at mid-strength has a nonzero gradient on
    # every image. Finite-difference rows of quantized ops can legitimately be 0.
    for name, spec in ops.items():
        if op_method(spec) == AUTOGRAD:
            assert (rows[name] != 0).all(), name


def test_every_host_vocabulary_probes():
    """A host's vocabulary is a valid probe op list, in the host's order."""
    hosts = pytest.importorskip("hosts")
    batch = make_batch(n=2, size=24)
    required = {
        "sensaug": {"max_iters": 20000},
        "randaugment_diff": {"dataset": "cityscapes"},
        "smartsampling_diff": {"max_iters": 20000},
        "madaug": {"max_iters": 20000},
    }
    for host_name, cls in hosts.HOSTS.items():
        host = cls(**required.get(host_name, {}))
        ops = vocab.build_vocab({"ops": host.vocabulary()})
        magnitudes = {n: strength_to_magnitude(s, 0.3) for n, s in ops.items() if s.has_magnitude}
        rows = collect_grads(ToyAdapter(), batch, ops, magnitudes)
        assert tuple(rows) == host.vocabulary(), host_name
