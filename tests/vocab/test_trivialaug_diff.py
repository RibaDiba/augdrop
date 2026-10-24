"""trivialaug_diff: TrivialAugmentWide's ops, differentiable, split into _pos/_neg.

Each split op is checked against torchvision itself: the enhance ops equal its float
blend and come within one 8-bit level of TrivialAugmentWide's uint8 output; the
geometric ops use torchvision's matrix, so at NEAREST they resample the same pixels
it does, and the bilinear op stays close. Autograd's derivative in the magnitude is
checked against central differences, and the bins against TrivialAugmentWide's.
"""

import pytest
import torch
import torchvision.transforms.v2.functional as TF
from kornia.geometry.transform import warp_affine

from hosts.trivialaugment import TrivialAugmentDiffHost, split_ops
from vocab import FAMILIES, build_vocab, get_op, magnitude_bins
from vocab.families import trivialaug_diff as tad

FAMILY = "trivialaug_diff"
ENHANCE = ("Brightness", "Color", "Contrast", "Sharpness")
GEOMETRIC = ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY")
SIGN = {"pos": 1.0, "neg": -1.0}


def _smooth(b=3, h=48, w=64, seed=0):
    """Smooth images, so a sub-pixel resampling difference is a small pixel difference."""
    g = torch.Generator().manual_seed(seed)
    coarse = torch.rand(b, 3, h // 8, w // 8, generator=g)
    return torch.nn.functional.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)


def _pixels(b=3, h=48, w=64, seed=0):
    """8-bit-representable images, so torchvision's uint8 path is exact on them."""
    return (_smooth(b, h, w, seed) * 255).round() / 255


# --- the op list ---------------------------------------------------------------


def test_names_and_order_match_the_host():
    names = [spec.name for spec in FAMILIES[FAMILY]]
    assert tuple(names) == split_ops(FAMILY) == TrivialAugmentDiffHost().vocabulary()
    assert len(names) == 22
    assert tuple(build_vocab({"families": [FAMILY]})) == tuple(names)


def test_split_ops_are_autograd_and_unsigned():
    for spec in FAMILIES[FAMILY]:
        base = spec.name.split(".", 1)[1]
        if base in tad.EXACT_IDS:
            assert spec.grad_mode == "none"
            continue
        transform_id, direction = base.rsplit("_", 1)
        assert direction in SIGN
        assert spec.grad_mode == "autograd"
        assert spec.mag_range == (0.0, tad.SPLIT_RANGES[transform_id]) and spec.identity_at == 0.0
        assert spec.warps_label == (transform_id in GEOMETRIC)


@pytest.mark.parametrize("transform_id", tad.EXACT_IDS)
def test_exact_ops_are_trivialaug_wides(transform_id):
    ours, theirs = get_op(f"{FAMILY}.{transform_id}"), get_op(f"trivialaug_wide.{transform_id}")
    assert (ours.mag_range, ours.identity_at, ours.kind) == (theirs.mag_range, theirs.identity_at, theirs.kind)
    images = _pixels()
    m = None if theirs.mag_range is None else (4.0 if transform_id == "Posterize" else 128.0)
    torch.testing.assert_close(ours.apply(images, m)[0], theirs.apply(images, m)[0], rtol=0, atol=0)


# --- bins -------------------------------------------------------------------------


@pytest.mark.parametrize("transform_id", sorted(tad.SPLIT_RANGES))
def test_bins_are_trivialaugments_unsigned(transform_id):
    wide, wide_signed = magnitude_bins(f"trivialaug_wide.{transform_id}", 31)
    identity = get_op(f"trivialaug_wide.{transform_id}").identity_at
    for direction in SIGN:
        values, signed = magnitude_bins(f"{FAMILY}.{transform_id}_{direction}", 31)
        assert wide_signed and not signed
        assert values[0] == 0.0
        assert values == pytest.approx([abs(v - identity) for v in wide], abs=1e-6)


@pytest.mark.parametrize("transform_id", ["Posterize", "Solarize"])
def test_exact_ops_keep_their_bins(transform_id):
    assert magnitude_bins(f"{FAMILY}.{transform_id}", 31) == magnitude_bins(f"trivialaug_wide.{transform_id}", 31)


# --- enhance ops: torchvision's blend ------------------------------------------------

_ADJUST = {
    "Brightness": TF.adjust_brightness,
    "Color": TF.adjust_saturation,
    "Contrast": TF.adjust_contrast,
    "Sharpness": TF.adjust_sharpness,
}


@pytest.mark.parametrize("transform_id", ENHANCE)
@pytest.mark.parametrize("direction", SIGN)
@pytest.mark.parametrize("m", [0.0, 0.033, 0.5, 0.99])
def test_enhance_equals_torchvisions_float_op(transform_id, direction, m):
    images = _smooth()
    out, _ = get_op(f"{FAMILY}.{transform_id}_{direction}").apply(images, m)
    expected = torch.stack([_ADJUST[transform_id](x, 1.0 + SIGN[direction] * m) for x in images])
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("transform_id", ENHANCE)
@pytest.mark.parametrize("direction", SIGN)
def test_enhance_is_trivialaugments_up_to_rounding(transform_id, direction):
    images = _pixels()
    m = 0.5
    ours, _ = get_op(f"{FAMILY}.{transform_id}_{direction}").apply(images, m)
    theirs, _ = get_op(f"trivialaug_wide.{transform_id}").apply(images, 1.0 + SIGN[direction] * m)
    # torchvision rounds (or truncates) the blend to uint8; Contrast's mean is truncated too.
    assert (ours - theirs).abs().max() <= 1.5 / 255


# --- geometric ops: torchvision's matrix ----------------------------------------------


@pytest.mark.parametrize("transform_id", GEOMETRIC)
@pytest.mark.parametrize("direction", SIGN)
def test_geometric_resamples_torchvisions_pixels(transform_id, direction):
    """At NEAREST, our matrix picks the pixels torchvision's TrivialAugmentWide does."""
    images = _pixels(seed=1)
    m = {"Rotate": 30.0, "ShearX": 0.3, "ShearY": 0.3, "TranslateX": 7.0, "TranslateY": 7.0}[transform_id]
    signed = SIGN[direction] * m
    matrix = tad._geometric_matrix(transform_id)(images, torch.full((images.shape[0],), signed))
    ours = warp_affine(images, matrix, dsize=images.shape[-2:], mode="nearest")
    theirs, _ = get_op(f"trivialaug_wide.{transform_id}").apply(images, signed)
    agree = ((ours - theirs).abs() < 1e-6).float().mean()
    assert agree > 0.98, f"{transform_id}_{direction}: {agree:.3f} of pixels agree"


@pytest.mark.parametrize("transform_id", GEOMETRIC)
@pytest.mark.parametrize("direction", SIGN)
def test_geometric_bilinear_is_close_to_torchvision(transform_id, direction):
    images = _smooth(seed=2)
    m = {"Rotate": 30.0, "ShearX": 0.3, "ShearY": 0.3, "TranslateX": 7.0, "TranslateY": 7.0}[transform_id]
    ours, _ = get_op(f"{FAMILY}.{transform_id}_{direction}").apply(images, m)
    theirs, _ = get_op(f"trivialaug_wide.{transform_id}").apply(images, SIGN[direction] * m)
    moved = (theirs - images).abs().mean()
    assert (ours - theirs).abs().mean() < 0.2 * moved


@pytest.mark.parametrize("transform_id", GEOMETRIC)
def test_geometric_moves_the_label(transform_id):
    images = _smooth(b=2, seed=3)
    label = torch.arange(48 * 64).reshape(1, 48, 64).repeat(2, 1, 1) % 19
    m = {"Rotate": 30.0, "ShearX": 0.3, "ShearY": 0.3, "TranslateX": 7.0, "TranslateY": 7.0}[transform_id]
    _, moved = get_op(f"{FAMILY}.{transform_id}_pos").apply(images, m, label)
    _, theirs = get_op(f"trivialaug_wide.{transform_id}").apply(images, m, label)
    assert moved.shape == label.shape and not torch.equal(moved, label)
    assert (moved == theirs).float().mean() > 0.98


# --- autograd ------------------------------------------------------------------------


@pytest.mark.parametrize("transform_id", ENHANCE + GEOMETRIC)
@pytest.mark.parametrize("direction", SIGN)
def test_autograd_matches_central_differences(transform_id, direction):
    op = get_op(f"{FAMILY}.{transform_id}_{direction}")
    images = _smooth(b=2, seed=4).double()
    weights = torch.rand(images.shape, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    hi = op.mag_range[1]
    m = torch.tensor([0.3 * hi, 0.6 * hi], dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad((op.apply(images, m)[0] * weights).sum(), m)

    h = 1e-4 * hi
    f = lambda mm: (op.apply(images, mm)[0] * weights).flatten(1).sum(1)  # noqa: E731
    fd = (f(m.detach() + h) - f(m.detach() - h)) / (2 * h)
    # Bilinear resampling and the clamp are piecewise linear in m: a finite step crosses
    # kinks the derivative doesn't see, a ~1% effect on these smooth images.
    torch.testing.assert_close(grad, fd, rtol=3e-2, atol=1e-6)
