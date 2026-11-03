"""randaug_diff and augmix_diff: RandAugment's and AugMix's ops, differentiable, split
into _pos/_neg (built by trivialaug_diff's `build_split_family`).

Each split op is checked against its method's own torchvision dispatcher: the enhance
ops equal torchvision's float op and come within one 8-bit level of its uint8 output;
the geometric ops use torchvision's matrix, so at NEAREST they resample the pixels it
does, with the translate scaled to the image as the method scales it. Autograd's
derivative is checked against central differences, the bins against the method's
`_AUGMENTATION_SPACE`, and randaug_diff's grid past 31 bins (levels above 10) against
the paper's linear scale.
"""

import pytest
import torch
import torchvision.transforms.v2.functional as TF
from kornia.geometry.transform import warp_affine
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import AugMix, RandAugment

from hosts.augmix import AugMixDiffHost
from hosts.randaugment import RandAugmentDiffHost, split_ops
from vocab import FAMILIES, build_vocab, get_op, magnitude_bins
from vocab.families import augmix_diff, randaug_diff
from vocab.families import trivialaug_diff as tad
from vocab.families.randaug_default import _fill_of

ENHANCE = ("Brightness", "Color", "Contrast", "Sharpness")
GEOMETRIC = ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY")
SIGN = {"pos": 1.0, "neg": -1.0}
H, W = 48, 64

#: family -> (its module, the torchvision method, its bins, a host over it)
FAMILY = {
    "randaug_diff": (randaug_diff, RandAugment, 31, lambda: RandAugmentDiffHost(dataset="cityscapes")),
    "augmix_diff": (augmix_diff, AugMix, 10, AugMixDiffHost),
}


def _smooth(b=3, h=H, w=W, seed=0):
    """Smooth images, so a sub-pixel resampling difference is a small pixel difference."""
    g = torch.Generator().manual_seed(seed)
    coarse = torch.rand(b, 3, h // 8, w // 8, generator=g)
    return torch.nn.functional.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)


def _pixels(b=3, h=H, w=W, seed=0):
    """8-bit-representable images, so torchvision's uint8 path is exact on them."""
    return (_smooth(b, h, w, seed) * 255).round() / 255


def _torchvision(method_cls, transform_id, images, native, interpolation=InterpolationMode.NEAREST):
    """The method's own dispatcher on each image (uint8), at torchvision's magnitude."""
    method = method_cls()
    pixels = (images * 255).round().to(torch.uint8)
    out = [
        method._apply_image_or_video_transform(x, transform_id, native, interpolation, _fill_of(method))
        for x in pixels
    ]
    return torch.stack(out).float() / 255


def _native_translate(family, transform_id, pixels):
    """The op-unit magnitude that translates by ``pixels`` on an H x W image."""
    extent = W if transform_id == "TranslateX" else H
    ref = randaug_diff.AA_IMAGE_SIZE if family == "randaug_diff" else augmix_diff.TRANSLATE_REF
    return pixels * ref / extent


def _strong(family, transform_id):
    """A magnitude in op units, and the native one torchvision's dispatcher takes."""
    if transform_id.startswith("Translate"):
        return _native_translate(family, transform_id, 7.0), 7.0
    m = {"Rotate": 25.0, "ShearX": 0.25, "ShearY": 0.25}.get(transform_id, 0.5)
    return m, m


# --- the op lists ------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILY))
def test_names_and_order_match_the_host(family):
    names = tuple(spec.name for spec in FAMILIES[family])
    assert names == split_ops(family) == FAMILY[family][3]().vocabulary()
    assert len(names) == 22
    assert tuple(build_vocab({"families": [family]})) == names


@pytest.mark.parametrize("family", sorted(FAMILY))
def test_split_ops_are_autograd_and_unsigned(family):
    module = FAMILY[family][0]
    for spec in FAMILIES[family]:
        base = spec.name.split(".", 1)[1]
        if base in module.EXACT_IDS:
            assert spec.grad_mode == "none"
            continue
        transform_id, direction = base.rsplit("_", 1)
        assert direction in SIGN and spec.grad_mode == "autograd"
        assert spec.mag_range == (0.0, module.SPLIT_RANGES[transform_id]) and spec.identity_at == 0.0
        assert spec.warps_label == (transform_id in GEOMETRIC)


@pytest.mark.parametrize("transform_id", ["AutoContrast", "Equalize", "Posterize", "Solarize"])
def test_randaug_diffs_exact_ops_are_randaug_defaults(transform_id):
    ours, theirs = get_op(f"randaug_diff.{transform_id}"), get_op(f"randaug_default.{transform_id}")
    assert (ours.mag_range, ours.identity_at, ours.kind) == (theirs.mag_range, theirs.identity_at, theirs.kind)
    images = _pixels()
    m = None if theirs.mag_range is None else (5.0 if transform_id == "Posterize" else 128.0)
    torch.testing.assert_close(ours.apply(images, m)[0], theirs.apply(images, m)[0], rtol=0, atol=0)


@pytest.mark.parametrize("transform_id, m, native", [("Posterize", 3.0, 3.0), ("Solarize", 200.0, 200.0 / 255)])
def test_augmix_diffs_exact_ops_are_augmixs(transform_id, m, native):
    images = _pixels()
    ours, _ = get_op(f"augmix_diff.{transform_id}").apply(images, m)
    torch.testing.assert_close(ours, _torchvision(AugMix, transform_id, images, native), rtol=0, atol=0)


# --- bins ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILY))
@pytest.mark.parametrize("transform_id", GEOMETRIC + ENHANCE)
def test_bins_are_the_methods_unsigned(family, transform_id):
    _, method_cls, num_bins, _ = FAMILY[family]
    magnitudes_fn, signed = method_cls._AUGMENTATION_SPACE[transform_id]
    expected = [abs(float(v)) for v in magnitudes_fn(num_bins, 331, 331)]
    if transform_id.startswith("Translate") and family == "augmix_diff":
        expected = [v / 331 for v in expected]  # a fraction of the image
    assert signed
    for direction in SIGN:
        values, ours_signed = magnitude_bins(f"{family}.{transform_id}_{direction}", num_bins)
        assert not ours_signed and values[0] == 0.0
        assert values == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("transform_id", ["Posterize", "Solarize"])
def test_randaug_diffs_exact_ops_keep_their_bins(transform_id):
    assert magnitude_bins(f"randaug_diff.{transform_id}", 31) == magnitude_bins(f"randaug_default.{transform_id}", 31)


def test_augmix_diffs_exact_bins_are_augmixs():
    posterize, _ = magnitude_bins("augmix_diff.Posterize", 10)
    solarize, _ = magnitude_bins("augmix_diff.Solarize", 10)
    space = AugMix._AUGMENTATION_SPACE
    assert posterize == tuple(float(v) for v in space["Posterize"][0](10, 331, 331))
    assert solarize == pytest.approx([255.0 * float(v) for v in space["Solarize"][0](10, 331, 331)])


# --- randaug_diff's grid past level 10 ---------------------------------------------------


@pytest.mark.parametrize("name", sorted(randaug_diff.BINS))
def test_randaug_diffs_grid_continues_torchvisions(name):
    """The extension's formula gives torchvision's own bins on the 31-bin grid, so the
    bins past it continue the same scale."""
    values, _ = magnitude_bins(name, 31)
    beyond = randaug_diff._beyond(name)
    assert [beyond(b) for b in range(31)] == pytest.approx(values, abs=1e-5)
    extended, _ = magnitude_bins(name, 43)
    assert extended[:31] == values and len(extended) == 43


def test_randaug_diff_at_level_14():
    """CIFAR-100 Wide-ResNet-28-10's M = 14: 1.4 times Table 6's ends, and each op stops
    where it stops meaning anything."""
    host = RandAugmentDiffHost(dataset="cifar100")

    def at(name):
        values, _ = magnitude_bins(name, host.num_magnitude_bins)
        return values[host.magnitude]

    assert at("randaug_diff.Rotate_pos") == pytest.approx(42.0)
    assert at("randaug_diff.ShearX_neg") == pytest.approx(0.42)
    assert at("randaug_diff.TranslateY_pos") == pytest.approx(210.0)
    assert at("randaug_diff.Brightness_pos") == pytest.approx(1.26)
    assert at("randaug_diff.Brightness_neg") == 1.0  # factor 0, not -0.26
    assert at("randaug_diff.Posterize") == 2.0  # 8 - round(42 / 7.5)
    assert at("randaug_diff.Solarize") == 0.0  # every pixel inverted


# --- enhance ops: the method's blend ----------------------------------------------------

_ADJUST = {
    "Brightness": TF.adjust_brightness,
    "Color": TF.adjust_saturation,
    "Contrast": TF.adjust_contrast,
    "Sharpness": TF.adjust_sharpness,
}


@pytest.mark.parametrize("family", sorted(FAMILY))
@pytest.mark.parametrize("transform_id", ENHANCE)
@pytest.mark.parametrize("direction", SIGN)
def test_enhance_equals_torchvisions_float_op(family, transform_id, direction):
    images = _smooth()
    for m in (0.0, 0.3, 0.9):
        out, _ = get_op(f"{family}.{transform_id}_{direction}").apply(images, m)
        expected = torch.stack([_ADJUST[transform_id](x, 1.0 + SIGN[direction] * m) for x in images])
        torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("family", sorted(FAMILY))
@pytest.mark.parametrize("transform_id", ENHANCE)
@pytest.mark.parametrize("direction", SIGN)
def test_enhance_is_the_methods_up_to_rounding(family, transform_id, direction):
    images = _pixels()
    ours, _ = get_op(f"{family}.{transform_id}_{direction}").apply(images, 0.5)
    theirs = _torchvision(FAMILY[family][1], transform_id, images, SIGN[direction] * 0.5)
    # torchvision rounds (or truncates) the blend to uint8; Contrast's mean is truncated too.
    assert (ours - theirs).abs().max() <= 1.5 / 255


# --- geometric ops: the method's matrix -------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILY))
@pytest.mark.parametrize("transform_id", GEOMETRIC)
@pytest.mark.parametrize("direction", SIGN)
def test_geometric_resamples_torchvisions_pixels(family, transform_id, direction):
    """At NEAREST, the op's matrix picks the pixels the method's dispatcher does."""
    images = _pixels(seed=1)
    _, native = _strong(family, transform_id)
    signed = SIGN[direction] * native
    matrix = tad._geometric_matrix(transform_id)(images, torch.full((images.shape[0],), signed))
    ours = warp_affine(images, matrix, dsize=images.shape[-2:], mode="nearest")
    theirs = _torchvision(FAMILY[family][1], transform_id, images, signed)
    agree = ((ours - theirs).abs() < 1e-6).float().mean()
    assert agree > 0.98, f"{family}.{transform_id}_{direction}: {agree:.3f} of pixels agree"


@pytest.mark.parametrize("family", sorted(FAMILY))
@pytest.mark.parametrize("transform_id", GEOMETRIC)
@pytest.mark.parametrize("direction", SIGN)
def test_geometric_op_is_its_matrix_at_the_methods_scale(family, transform_id, direction):
    """The op applies that matrix, bilinear, with a translate scaled to the image (7 px
    here) as the method scales it."""
    images = _smooth(seed=2)
    m, native = _strong(family, transform_id)
    ours, _ = get_op(f"{family}.{transform_id}_{direction}").apply(images, m)
    matrix = tad._geometric_matrix(transform_id)(images, torch.full((images.shape[0],), SIGN[direction] * native))
    torch.testing.assert_close(ours, warp_affine(images, matrix, dsize=images.shape[-2:]), rtol=0, atol=1e-5)


@pytest.mark.parametrize("family", sorted(FAMILY))
@pytest.mark.parametrize("transform_id", GEOMETRIC)
def test_geometric_moves_the_label(family, transform_id):
    images = _smooth(b=2, seed=3)
    label = torch.arange(H * W).reshape(1, H, W).repeat(2, 1, 1) % 19
    m, _ = _strong(family, transform_id)
    _, moved = get_op(f"{family}.{transform_id}_pos").apply(images, m, label)
    assert moved.shape == label.shape and not torch.equal(moved, label)


# --- autograd ----------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILY))
@pytest.mark.parametrize("transform_id", ENHANCE + GEOMETRIC)
@pytest.mark.parametrize("direction", SIGN)
def test_autograd_matches_central_differences(family, transform_id, direction):
    op = get_op(f"{family}.{transform_id}_{direction}")
    images = _smooth(b=2, seed=4).double()
    weights = torch.rand(images.shape, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    hi = op.mag_range[1]
    m = torch.tensor([0.3 * hi, 0.6 * hi], dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad((op.apply(images, m)[0] * weights).sum(), m)

    h = 1e-4 * hi
    f = lambda mm: (op.apply(images, mm)[0] * weights).flatten(1).sum(1)  # noqa: E731
    fd = (f(m.detach() + h) - f(m.detach() - h)) / (2 * h)
    # Bilinear resampling and the clamp are piecewise linear in m (see test_trivialaug_diff).
    torch.testing.assert_close(grad, fd, rtol=3e-2, atol=1e-6)
