"""The five families: their op lists, their units, and -- for the two
torchvision-backed families -- that each op is exactly torchvision's."""

import math

import pytest
import torch
import torchvision.transforms.v2.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import AutoAugment, RandAugment, TrivialAugmentWide

from vocab import FAMILIES, get_op
from vocab.diff_ops import LABEL_IGNORE_INDEX
from vocab.families import randaug_default, trivialaug_wide

LEGACY_HSV_NAMES = [
    "lighter_R", "darker_R", "lighter_G", "darker_G", "lighter_B", "darker_B",
    "lighter_H", "darker_H", "lighter_S", "darker_S", "lighter_V", "darker_V",
    "blur", "noise",
]
GEOMETRIC_NAMES = [
    f"{base}_{d}" for base in ("rotate", "shear_x", "shear_y", "translate_x", "translate_y")
    for d in ("pos", "neg")
]
PIL_AUTOAUG_NAMES = [
    f"{base}_{d}" for base in ("brightness", "contrast", "sharpness", "color") for d in ("pos", "neg")
]
# RandAugment paper, Fig. 2.
PAPER_OP_ORDER = [
    "Identity", "AutoContrast", "Equalize", "Rotate", "Solarize", "Color", "Posterize",
    "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateX", "TranslateY",
]
TORCHVISION_FAMILIES = {
    "randaug_default": (randaug_default, RandAugment),
    "trivialaug_wide": (trivialaug_wide, TrivialAugmentWide),
}


def _names(family):
    return [spec.name for spec in FAMILIES[family]]


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.rand(2, 3, 32, 32)


@pytest.fixture
def pixels():
    """8-bit images, the originals' domain; float [0, 1] view of them in `.float01`."""
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (2, 3, 40, 56), generator=g, dtype=torch.uint8)


def test_family_sizes_and_names():
    assert _names("legacy_hsv") == LEGACY_HSV_NAMES
    assert _names("geometric") == GEOMETRIC_NAMES
    assert _names("pil_autoaug") == PIL_AUTOAUG_NAMES
    assert _names("randaug_default") == [f"randaug_default.{t}" for t in PAPER_OP_ORDER]
    assert _names("trivialaug_wide") == [f"trivialaug_wide.{t}" for t in PAPER_OP_ORDER]
    assert _names("autoaugment") == ["autoaugment.Invert"]


@pytest.mark.parametrize("family", ["legacy_hsv", "geometric", "pil_autoaug"])
def test_sensaug_families_keep_sensaugs_units(family):
    for spec in FAMILIES[family]:
        assert spec.mag_range == (0.0, 1.0) and spec.identity_at == 0.0
        assert spec.grad_mode == "autograd"


def test_only_geometric_ops_warp_the_label():
    for specs in FAMILIES.values():
        for spec in specs:
            assert spec.warps_label == (spec.kind == "geometric")
    assert {s.name for s in FAMILIES["geometric"]} == set(GEOMETRIC_NAMES)
    for family in TORCHVISION_FAMILIES:
        geometric = {s.name.split(".", 1)[1] for s in FAMILIES[family] if s.warps_label}
        assert geometric == {"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"}


@pytest.mark.parametrize("base", sorted({n.rsplit("_", 1)[0] for n in GEOMETRIC_NAMES + PIL_AUTOAUG_NAMES}))
def test_pos_and_neg_move_in_opposite_directions(batch, base):
    """If _pos and _neg produced the same image, half the pairs would be duplicate
    axes of R."""
    pos, _ = get_op(f"{base}_pos").apply(batch, 0.5)
    neg, _ = get_op(f"{base}_neg").apply(batch, 0.5)
    assert not torch.allclose(pos, neg, atol=1e-4)


# --- torchvision families: op list and units ---------------------------------


@pytest.mark.parametrize("family", TORCHVISION_FAMILIES)
def test_op_list_is_torchvisions(family):
    _, method = TORCHVISION_FAMILIES[family]
    assert set(PAPER_OP_ORDER) == set(method._AUGMENTATION_SPACE)


def _torchvision_range_in_paper_units(family, transform_id):
    """torchvision's full bin table for one op, converted to the paper's units,
    signed where torchvision signs it. Evaluated at 331x331, the size the
    AutoAugment ranges are stated for."""
    _, method = TORCHVISION_FAMILIES[family]
    magnitudes_fn, signed = method._AUGMENTATION_SPACE[transform_id]
    bins = magnitudes_fn(31, 331, 331)
    if bins is None:
        return None
    values = bins.double()
    if signed:
        values = torch.cat([values, -values])
    if transform_id in randaug_default.ENHANCE_IDS:
        values = 1.0 + values
    elif transform_id == "Solarize":
        values = values * 255.0
    return float(values.min()), float(values.max())


# (family, transform id) -> the torchvision table's (lo, hi), where it differs
# from the paper's range. Both are the originals' own ends: torchvision's top
# solarize bin is threshold 255 (it rejects 256, the paper's identity end), and
# its TrivialAugment Wide enhance bins stop at 1.99 where the paper says 2.
KNOWN_TORCHVISION_DIFFERENCES = {
    ("randaug_default", "Solarize"): (0.0, 255.0),
    ("trivialaug_wide", "Solarize"): (0.0, 255.0),
    **{("trivialaug_wide", t): (0.01, 1.99) for t in randaug_default.ENHANCE_IDS},
}


@pytest.mark.parametrize("family", TORCHVISION_FAMILIES)
@pytest.mark.parametrize("transform_id", PAPER_OP_ORDER)
def test_paper_range_agrees_with_torchvisions_table(family, transform_id):
    spec = get_op(f"{family}.{transform_id}")
    table = _torchvision_range_in_paper_units(family, transform_id)
    if table is None:
        assert spec.mag_range is None and spec.identity_at is None
        return
    expected = KNOWN_TORCHVISION_DIFFERENCES.get((family, transform_id), spec.mag_range)
    assert table == pytest.approx(expected, abs=1e-6)


def test_ranges_are_the_papers():
    """AutoAugment Table 6 (inherited by RandAugment) and TrivialAugment Table 8."""
    ra = {t: get_op(f"randaug_default.{t}").mag_range for t in PAPER_OP_ORDER}
    ta = {t: get_op(f"trivialaug_wide.{t}").mag_range for t in PAPER_OP_ORDER}
    assert ra["Rotate"] == (-30.0, 30.0) and ta["Rotate"] == (-135.0, 135.0)
    assert ra["ShearX"] == (-0.3, 0.3) and ta["ShearX"] == (-0.99, 0.99)
    assert ra["TranslateX"] == (-150.0, 150.0) and ta["TranslateX"] == (-32.0, 32.0)
    assert ra["Solarize"] == ta["Solarize"] == (0.0, 256.0)
    assert ra["Posterize"] == (4.0, 8.0) and ta["Posterize"] == (2.0, 8.0)
    for t in ("Color", "Contrast", "Brightness", "Sharpness"):
        assert ra[t] == (0.1, 1.9) and ta[t] == (0.01, 2.0)
    for t in ("Identity", "AutoContrast", "Equalize"):
        assert ra[t] is None and ta[t] is None


# --- torchvision families: each op is torchvision's --------------------------


def _reference(family, transform_id, image, m):
    """What the paper describes, via torchvision's PUBLIC functionals, on one
    uint8 (3, H, W) image -- independent of the dispatcher the op calls."""
    h, w = image.shape[-2:]
    nearest = InterpolationMode.NEAREST
    if transform_id == "Identity":
        return image
    if transform_id == "AutoContrast":
        return TF.autocontrast(image)
    if transform_id == "Equalize":
        return TF.equalize(image)
    if transform_id == "Rotate":
        return TF.rotate(image, angle=m, interpolation=nearest)
    if transform_id in ("ShearX", "ShearY"):
        shear = [math.degrees(math.atan(m)), 0.0]
        if transform_id == "ShearY":
            shear = shear[::-1]
        return TF.affine(image, angle=0.0, translate=[0, 0], scale=1.0, shear=shear,
                         interpolation=nearest, center=[0, 0])
    if transform_id in ("TranslateX", "TranslateY"):
        extent = w if transform_id == "TranslateX" else h
        pixels = m * extent / 331.0 if family == "randaug_default" else m
        translate = [int(pixels), 0] if transform_id == "TranslateX" else [0, int(pixels)]
        return TF.affine(image, angle=0.0, translate=translate, scale=1.0, shear=[0.0, 0.0],
                         interpolation=nearest)
    if transform_id == "Brightness":
        return TF.adjust_brightness(image, m)
    if transform_id == "Color":
        return TF.adjust_saturation(image, m)
    if transform_id == "Contrast":
        return TF.adjust_contrast(image, m)
    if transform_id == "Sharpness":
        return TF.adjust_sharpness(image, m)
    if transform_id == "Posterize":
        return TF.posterize(image, int(m))
    if transform_id == "Solarize":
        return image if m > 255 else TF.solarize(image, m)
    raise AssertionError(transform_id)


def _probe_magnitudes(spec):
    if spec.mag_range is None:
        return [None]
    lo, hi = spec.mag_range
    return sorted({lo, hi, spec.identity_at, lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)})


@pytest.mark.parametrize("family", TORCHVISION_FAMILIES)
@pytest.mark.parametrize("transform_id", PAPER_OP_ORDER)
def test_op_is_torchvisions_op_in_paper_units(family, transform_id, pixels):
    spec = get_op(f"{family}.{transform_id}")
    images = pixels.float() / 255.0
    for m in _probe_magnitudes(spec):
        out, _ = spec.apply(images, m)
        assert out.dtype == images.dtype and out.shape == images.shape
        # The op takes its magnitude at the images' precision (float32 here).
        m_image = None if m is None else float(torch.tensor(m, dtype=images.dtype))
        for i in range(images.shape[0]):
            expected = _reference(family, transform_id, pixels[i], m_image)
            got = (out[i] * 255.0).round().to(torch.uint8)
            assert torch.equal(got, expected), f"{spec.name} at {m}, image {i}"


def test_autoaugment_invert_is_torchvisions(pixels):
    spec = get_op("autoaugment.Invert")
    assert spec.mag_range is None and spec.kind == "photometric"
    out, _ = spec.apply(pixels.float() / 255.0)
    for i in range(pixels.shape[0]):
        assert torch.equal((out[i] * 255.0).round().to(torch.uint8), TF.invert(pixels[i]))


def test_autoaugment_ops_are_randaug_defaults_ops():
    """AutoAugment's other 13 ops are named by randaug_default's (hosts/autoaugment),
    because RandAugment took them from AutoAugment's Table 6. Pin that torchvision's
    two tables agree on every one: same ops, same sign handling, same range. Only
    the number of bins differs (10 vs 31)."""
    shared = set(AutoAugment._AUGMENTATION_SPACE) - {"Invert"}
    assert shared == set(PAPER_OP_ORDER) - {"Identity"}
    for transform_id in shared:
        aa_fn, aa_signed = AutoAugment._AUGMENTATION_SPACE[transform_id]
        ra_fn, ra_signed = RandAugment._AUGMENTATION_SPACE[transform_id]
        assert aa_signed == ra_signed, transform_id
        aa, ra = aa_fn(10, 331, 331), ra_fn(31, 331, 331)
        if aa is None:
            assert ra is None, transform_id
            continue
        assert float(aa[0]) == pytest.approx(float(ra[0])), transform_id
        assert float(aa[-1]) == pytest.approx(float(ra[-1])), transform_id


@pytest.mark.parametrize("family", TORCHVISION_FAMILIES)
@pytest.mark.parametrize("transform_id", ["Color", "Sharpness", "Solarize", "TranslateX"])
def test_python_float_and_tensor_magnitudes_agree(family, transform_id, pixels):
    """1.45 as a python float and as a float32 tensor once landed either side of
    a rounding boundary in the 8-bit output. The probe passes tensors; the audit
    compares the two."""
    spec = get_op(f"{family}.{transform_id}")
    images = pixels.float() / 255.0
    lo, hi = spec.mag_range
    ident = spec.identity_at
    for m in ((ident + hi) / 2, (ident + lo) / 2):  # the audit's midway magnitudes
        as_float, _ = spec.apply(images, m)
        as_tensor, _ = spec.apply(images, torch.full((images.shape[0],), m))
        assert torch.equal(as_float, as_tensor), m


def test_ops_take_a_per_image_magnitude(pixels):
    images = pixels.float() / 255.0
    out, _ = get_op("randaug_default.Rotate").apply(images, torch.tensor([0.0, 30.0]))
    assert torch.equal(out[0], images[0])
    single, _ = get_op("randaug_default.Rotate").apply(images[1:], 30.0)
    assert torch.equal(out[1:], single)


def test_ops_reject_a_magnitude_of_the_wrong_length(pixels):
    with pytest.raises(ValueError, match="length-2 vector"):
        get_op("trivialaug_wide.Rotate").apply(pixels.float() / 255.0, torch.tensor([1.0, 2.0, 3.0]))


def test_ops_carry_no_gradient(pixels):
    m = torch.tensor(10.0, requires_grad=True)
    out, _ = get_op("randaug_default.Rotate").apply(pixels.float() / 255.0, m)
    assert not out.requires_grad


# --- torchvision families: labels --------------------------------------------


@pytest.mark.parametrize("family", TORCHVISION_FAMILIES)
def test_out_of_frame_label_is_the_ignore_index(family, pixels):
    """Sensaug's test_out_of_frame_is_ignore, through the vocab call."""
    images = pixels.float() / 255.0
    label = torch.zeros(images.shape[0], *images.shape[-2:], dtype=torch.long)
    spec = get_op(f"{family}.TranslateX")
    m = 32.0 if family == "trivialaug_wide" else 150.0
    shift = int(m * images.shape[-1] / 331.0) if family == "randaug_default" else int(m)
    _, label_out = spec.apply(images, m, label)
    assert (label_out[:, :, :shift] == LABEL_IGNORE_INDEX).all()
    assert (label_out[:, :, shift:] == 0).all()


def test_label_keeps_its_rank_and_dtype(pixels):
    images = pixels.float() / 255.0
    b, _, h, w = images.shape
    spec = get_op("randaug_default.ShearX")
    _, out3 = spec.apply(images, 0.3, torch.zeros(b, h, w, dtype=torch.long))
    _, out4 = spec.apply(images, 0.3, torch.zeros(b, 1, h, w, dtype=torch.long))
    assert out3.shape == (b, h, w) and out3.dtype == torch.long
    assert out4.shape == (b, 1, h, w) and out4.dtype == torch.long


def test_label_size_mismatch_is_refused(pixels):
    """Unlike the differentiable geometric ops, these cannot re-express the warp
    in another frame, so a label at another resolution is refused (as Sensaug's
    label-safe transforms refuse it)."""
    images = pixels.float() / 255.0
    label = torch.zeros(images.shape[0], images.shape[-2] - 1, images.shape[-1], dtype=torch.long)
    with pytest.raises(ValueError, match="differ in size"):
        get_op("randaug_default.Rotate").apply(images, 10.0, label)


def test_photometric_ops_pass_the_label_through(pixels):
    images = pixels.float() / 255.0
    label = torch.randint(0, 5, (images.shape[0], *images.shape[-2:]))
    _, label_out = get_op("trivialaug_wide.Posterize").apply(images, 2.0, label)
    assert label_out is label
