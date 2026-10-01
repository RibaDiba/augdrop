"""audit() passes on every real op, and fails on ops broken in each of the ways
Sensaug's probe actually broke."""

import json

import pytest
import torch

from vocab import OPS, OpSpec, audit
from vocab.audit import CHECKS, FAIL, PASS, SKIP
from vocab.diff_ops import _as_delta, _warp, color_channel, rotate_matrix, rotate_op, warp_image_and_label


def test_every_op_passes_every_check():
    report = audit()
    assert report.ok, report.summary()
    assert len(report.results) == len(OPS) * len(CHECKS)


def test_checks_that_do_not_apply_are_skipped_not_passed():
    report = audit([OPS["randaug_default.Equalize"], OPS["randaug_default.Rotate"]])
    assert report.status("randaug_default.Equalize", "identity") == SKIP
    assert report.status("randaug_default.Rotate", "gradient") == SKIP
    assert report.status("randaug_default.Rotate", "label") == PASS


def test_report_serializes_and_summarizes():
    report = audit([OPS["blur"]])
    json.dumps(report.to_dict())
    assert "blur" in report.summary()


def test_audit_does_not_consume_the_callers_rng():
    torch.manual_seed(1)
    expected = torch.rand(3)
    torch.manual_seed(1)
    audit([OPS["noise"]])
    assert torch.equal(torch.rand(3), expected)


# --- broken ops the audit must catch ----------------------------------------


def _photometric(name, op, **overrides):
    fields = dict(
        name=name, family="broken", kind="photometric", source="test", mag_range=(0.0, 1.0),
        identity_at=0.0, grad_mode="autograd", warps_label=False,
        fn=lambda images, m, label=None: (op(images, _as_delta(m, images)), label),
    )
    fields.update(overrides)
    return OpSpec(**fields)


def _failed(spec):
    report = audit([spec])
    return {r.check for r in report.results if r.status == FAIL}


def test_catches_an_op_that_collapses_the_per_image_magnitude():
    """The bug the shape check cannot see: delta.mean() yields a gradient of the
    right shape and the wrong values."""
    spec = _photometric("mean_delta", lambda x, d: color_channel(x, 0, d.mean()))
    assert {"per_image", "gradient"} <= _failed(spec)


def test_catches_an_op_that_detaches_the_magnitude():
    """torch.tensor([[...]]) detaches silently; the forward pass looks perfect."""
    spec = _photometric("detached", lambda x, d: color_channel(x, 0, d.detach()))
    assert _failed(spec) == {"gradient"}


def test_catches_a_geometric_op_that_leaves_the_label_behind():
    """What contaminated Sensaug's geometric rows of R until it was fixed."""
    spec = OpSpec(
        name="rotate_no_label", family="broken", kind="geometric", source="test",
        mag_range=(0.0, 1.0), identity_at=0.0, grad_mode="autograd", warps_label=True,
        fn=lambda images, m, label=None: (rotate_op(images, m), label),
    )
    assert "label" in _failed(spec)


def test_catches_a_label_warped_by_a_different_amount():
    def fn(images, m, label=None):
        out = _warp(images, rotate_matrix(images, m))
        if label is None:
            return out, None
        _, moved = warp_image_and_label(images, label, rotate_matrix(images, 0.5 * _as_delta(m, images)))
        return out, moved

    spec = OpSpec(
        name="rotate_half_label", family="broken", kind="geometric", source="test",
        mag_range=(0.0, 1.0), identity_at=0.0, grad_mode="autograd", warps_label=True, fn=fn,
    )
    assert "label" in _failed(spec)


def test_catches_a_biased_zero_point():
    spec = _photometric("biased", lambda x, d: color_channel(x, 0, d + 0.05))
    assert "identity" in _failed(spec)


def test_catches_an_op_that_couples_images_without_a_magnitude():
    def batch_normalize(images, m, label=None):
        return (images - images.mean()).clamp(0.0, 1.0), label

    spec = OpSpec(
        name="batch_stats", family="broken", kind="photometric", source="test",
        mag_range=None, identity_at=None, grad_mode="none", warps_label=False, fn=batch_normalize,
    )
    assert _failed(spec) == {"per_image"}


def test_catches_an_op_that_ignores_its_magnitude():
    spec = _photometric("inert", lambda x, d: x + 0.0 * d)
    assert {"moves_pixels", "per_image", "gradient"} <= _failed(spec)


def test_a_crashing_op_fails_its_checks_without_stopping_the_audit():
    def boom(images, m, label=None):
        raise RuntimeError("boom")

    spec = _photometric("boom", None, fn=boom)
    report = audit([spec, OPS["blur"]])
    assert all(r.status == FAIL for r in report.results if r.op == "boom")
    assert "RuntimeError: boom" in report.failures()[0].detail
    assert all(r.status != FAIL for r in report.results if r.op == "blur")


def test_batch_size_below_two_is_refused():
    with pytest.raises(ValueError):
        audit([OPS["blur"]], batch_size=1)
