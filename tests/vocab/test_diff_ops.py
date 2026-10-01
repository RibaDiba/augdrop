"""The differentiable op implementations, ported with Sensaug's own tests
(tests/test_differentiable_augmentations.py and
tests/test_differentiable_augmentations_aa.py). Spec-level behaviour
(identity, per-image independence, gradients over every op) is covered by
test_audit.py; this file pins the implementation details those checks would not
localize."""

import math

import kornia.color
import pytest
import torch
from kornia.geometry.transform import warp_affine

from vocab import diff_ops
from vocab.diff_ops import (
    BLUR_KERNEL_SIZE,
    BRIGHTNESS_POS_SCALE,
    COLOR_POS_SCALE,
    CONTRAST_POS_SCALE,
    LABEL_IGNORE_INDEX,
    PHOTOMETRIC_NEG_SCALE,
    ROTATE_MAX_DEG,
    SHARPNESS_POS_SCALE,
    SHEAR_MAX_FACTOR,
    TRANSLATE_MAX_FRACTION,
    R,
    V,
    _broadcast_delta,
    _matrix_in_frame,
    _signed_scale,
    affine_matrix,
    blur,
    brightness_op,
    color_channel,
    color_op,
    contrast_op,
    gaussian_noise,
    hsv_channel,
    rotate_matrix,
    rotate_op,
    sharpness_op,
    shear_x_matrix,
    shear_y_matrix,
    translate_x_matrix,
    warp_image_and_label,
)

AA_OPS = [
    diff_ops.rotate_op, diff_ops.shear_x_op, diff_ops.shear_y_op,
    diff_ops.translate_x_op, diff_ops.translate_y_op,
    brightness_op, contrast_op, sharpness_op, color_op,
]


@pytest.fixture
def batch():
    """Textured, not flat: several ops (sharpness above all) are the identity on
    a constant image, so a flat fixture would pass every gradient test."""
    torch.manual_seed(0)
    return torch.rand(2, 3, 32, 32)


@pytest.fixture
def big_batch():
    torch.manual_seed(0)
    return torch.rand(4, 3, 32, 32)


# --- RGB / HSV rails ---------------------------------------------------------


def test_color_channel_matches_hand_derived_cpu_formula(batch):
    """Sensaug's CPU perturb_rgb computes out = channel*(1-alpha) + direction*alpha
    on the [0, 1] scale; color_channel must reproduce it for both directions."""
    m = 0.4
    lighter = color_channel(batch, R, delta=m)[:, R]
    darker = color_channel(batch, R, delta=-m)[:, R]
    assert torch.allclose(lighter, batch[:, R] * (1 - m) + m)
    assert torch.allclose(darker, batch[:, R] * (1 - m))


def test_hsv_channel_round_trip_is_identity_at_zero_delta(batch):
    assert torch.allclose(hsv_channel(batch, V, delta=0.0), batch, atol=1e-5)


def test_hsv_darken_v_rails_to_floor_not_black(batch):
    v = kornia.color.rgb_to_hsv(hsv_channel(batch, V, delta=-1.0))[:, 2]
    assert torch.allclose(v, torch.full_like(v, 10.0 / 255.0), atol=1e-4)


def test_color_channel_rejects_invalid_channel(batch):
    with pytest.raises(AssertionError):
        color_channel(batch, channel=99, delta=0.3)


def test_broadcast_delta_accepts_scalar_and_per_image():
    assert _broadcast_delta(torch.tensor(0.5), batch_size=4).shape == (1, 1, 1, 1)
    assert _broadcast_delta(torch.zeros(4), batch_size=4).shape == (4, 1, 1, 1)


def test_broadcast_delta_rejects_wrong_length():
    with pytest.raises(AssertionError):
        _broadcast_delta(torch.zeros(3), batch_size=4)


def test_rail_selection_is_per_image_not_per_batch(big_batch):
    """The sign of delta picks the rail. With a per-image delta the sign differs
    across the batch, so a batch-wide `if delta >= 0` would send half the images
    toward the wrong rail."""
    deltas = torch.tensor([0.5, -0.5, 0.5, -0.5])
    out = color_channel(big_batch, R, deltas)[:, R]
    for i, d in enumerate(deltas):
        rail = 1.0 if d >= 0 else 0.0
        expected = big_batch[i, R] * (1 - d.abs()) + rail * d.abs()
        assert torch.allclose(out[i], expected, atol=1e-6), f"image {i} wrong rail"


def test_hsv_v_darken_floor_is_selected_per_image(big_batch):
    deltas = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    v = kornia.color.rgb_to_hsv(hsv_channel(big_batch, V, deltas))[:, 2]
    for i in (0, 2):
        assert torch.allclose(v[i], torch.full_like(v[i], 10.0 / 255.0), atol=1e-4)
    for i in (1, 3):
        assert torch.allclose(v[i], torch.ones_like(v[i]), atol=1e-4)


# --- blur --------------------------------------------------------------------


def _dense_blur_reference(images, sigma, kernel_size=(67, 67)):
    """Sensaug's previous blur: one dense 2D kernel, 67 taps. Kept here so the
    separable, contract-sized blur keeps being compared against the old numbers."""
    kh, kw = kernel_size
    s = torch.as_tensor(sigma, dtype=images.dtype).reshape(-1).clamp(min=1e-3)
    ax_y = torch.arange(kh, dtype=images.dtype) - (kh - 1) / 2.0
    ax_x = torch.arange(kw, dtype=images.dtype) - (kw - 1) / 2.0
    var = (2 * s**2 + 1e-12).view(-1, 1)
    gauss_y = torch.exp(-(ax_y**2).view(1, -1) / var)
    gauss_x = torch.exp(-(ax_x**2).view(1, -1) / var)
    kernel2d = gauss_y[:, :, None] * gauss_x[:, None, :]
    kernel2d = kernel2d / kernel2d.sum(dim=(1, 2), keepdim=True)
    kernel = kernel2d.view(-1, 1, kh, kw)

    b, c = images.shape[0], images.shape[1]
    if kernel.shape[0] == 1:
        return torch.nn.functional.conv2d(
            images, kernel.repeat(c, 1, 1, 1), padding=(kh // 2, kw // 2), groups=c
        )
    kernel = kernel.repeat_interleave(c, dim=0)
    out = torch.nn.functional.conv2d(
        images.reshape(1, b * c, *images.shape[2:]), kernel, padding=(kh // 2, kw // 2), groups=b * c
    )
    return out.reshape(b, c, *out.shape[2:])


@pytest.mark.filterwarnings("ignore:blur. sigma up to:RuntimeWarning")
def test_blur_applies_each_images_own_sigma(big_batch):
    """Per-image kernels must be repeat_interleave'd, not tiled; far-apart sigmas
    so a misassignment cannot hide inside the tolerance. The kernel is
    deliberately too small for the larger sigmas, which makes a mix-up easier to
    see."""
    sigmas = torch.tensor([0.5, 3.0, 1.0, 8.0])
    out = blur(big_batch, sigmas, kernel_size=(15, 15))
    for i, sigma in enumerate(sigmas):
        expected = blur(big_batch[i : i + 1], sigma, kernel_size=(15, 15))
        assert torch.allclose(out[i : i + 1], expected, atol=1e-5), f"image {i}"


@pytest.mark.parametrize("sigma", [0.25, 0.5, 1.0])
def test_blur_matches_dense_67_reference_over_magnitude_contract(batch, sigma):
    assert torch.allclose(blur(batch, sigma), _dense_blur_reference(batch, sigma), atol=1e-5)


def test_blur_separable_matches_dense_at_matched_kernel_size(batch):
    assert torch.allclose(
        blur(batch, 0.6, kernel_size=(67, 67)),
        _dense_blur_reference(batch, 0.6, kernel_size=(67, 67)),
        atol=1e-5,
    )


def test_blur_per_image_sigma_matches_dense_reference(big_batch):
    sigmas = torch.tensor([0.1, 0.5, 0.8, 1.0])
    assert torch.allclose(blur(big_batch, sigmas), _dense_blur_reference(big_batch, sigmas), atol=1e-5)


def test_blur_gradient_wrt_sigma_matches_dense_reference(batch):
    s_new = torch.tensor(0.7, requires_grad=True)
    s_old = torch.tensor(0.7, requires_grad=True)
    g_new = torch.autograd.grad(blur(batch, s_new).sum(), s_new)[0]
    g_old = torch.autograd.grad(_dense_blur_reference(batch, s_old).sum(), s_old)[0]
    assert g_new != 0.0
    assert abs(g_new - g_old) / abs(g_old) < 1e-4


def test_blur_warns_when_sigma_outruns_kernel(batch):
    with pytest.warns(RuntimeWarning, match="exceeds what kernel_size"):
        blur(batch, 5.0)


def test_blur_does_not_warn_within_magnitude_contract(batch, recwarn):
    blur(batch, 1.0)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_default_kernel_resolves_the_magnitude_contract():
    assert (min(BLUR_KERNEL_SIZE) - 1) / 6.0 >= 1.0


# --- noise -------------------------------------------------------------------


def test_noise_applies_each_images_own_delta(big_batch, monkeypatch):
    """With the sample held fixed, image i must be scaled by delta[i]; delta 0 is
    a no-op."""
    fixed = torch.randn_like(big_batch)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: fixed)
    deltas = torch.tensor([0.0, 0.1, -0.2, 0.4])
    out = gaussian_noise(big_batch, deltas)
    expected = torch.clamp(big_batch + fixed * deltas.view(-1, 1, 1, 1), 0.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out[0], big_batch[0], atol=1e-6)


# --- AutoAugment-family ops --------------------------------------------------


@pytest.mark.parametrize("op", AA_OPS, ids=lambda f: f.__name__)
def test_scalar_magnitude_is_expanded_not_repeated(batch, op):
    scalar = op(batch, 0.35)
    per_image = op(batch, torch.tensor([0.35, 0.35]))
    assert torch.allclose(scalar, per_image, atol=1e-6)


def test_scalar_magnitude_accumulates_gradient_over_the_batch(batch):
    """Expanding rather than repeating means one scalar receives the sum of the
    batch's gradients -- invisible in the forward pass, so pinned here."""
    scalar = torch.tensor(0.4, requires_grad=True)
    per_image = torch.tensor([0.4, 0.4], requires_grad=True)
    g_scalar = torch.autograd.grad(rotate_op(batch, scalar).sum(), scalar)[0]
    g_each = torch.autograd.grad(rotate_op(batch, per_image).sum(), per_image)[0]
    assert torch.allclose(g_scalar, g_each.sum(), rtol=1e-4)


def test_rejects_a_magnitude_that_is_neither_scalar_nor_length_b(batch):
    with pytest.raises(AssertionError, match="length-2 vector"):
        rotate_op(batch, torch.tensor([0.1, 0.2, 0.3]))


def test_calibration_constants_match_the_sensaug_cpu_classes():
    assert ROTATE_MAX_DEG == 45.0  # Rotate.max_angle
    assert SHEAR_MAX_FACTOR == 0.3  # ShearX: magnitude * 0.3
    assert TRANSLATE_MAX_FRACTION == 0.25  # TranslateX: dim / 4.0


def test_affine_matrix_with_no_arguments_is_the_identity(batch):
    matrix = affine_matrix(batch)
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).expand(batch.shape[0], 2, 3)
    assert matrix.shape == (batch.shape[0], 2, 3)
    assert torch.allclose(matrix, identity, atol=1e-6)


def test_shear_matrix_reproduces_the_cpu_matrix_element(batch):
    """The CPU shear matrix puts the raw factor in M[0, 1]; kornia writes -tan of
    its angle there. Pins the -atan() conversion (a raw factor would overshoot
    by tan(x)/x AND flip the sign)."""
    m = 0.6
    assert shear_x_matrix(batch, m)[0, 0, 1].item() == pytest.approx(m * SHEAR_MAX_FACTOR, abs=1e-6)
    assert shear_y_matrix(batch, m)[0, 1, 0].item() == pytest.approx(m * SHEAR_MAX_FACTOR, abs=1e-6)


def test_translate_matrix_is_a_quarter_extent_at_full_magnitude(batch):
    assert translate_x_matrix(batch, 1.0)[0, 0, 2].item() == pytest.approx(batch.shape[-1] / 4.0, abs=1e-4)


def test_rotate_matrix_is_45_degrees_at_full_magnitude(batch):
    assert rotate_matrix(batch, 1.0)[0, 0, 0].item() == pytest.approx(math.cos(math.radians(45.0)), abs=1e-5)


def test_photometric_negative_direction_uses_the_smaller_cpu_scale():
    """The negative CPU classes carry their own 0.6 scale rather than flipping the
    sign, and only color has a positive scale other than 1.0."""
    d = torch.tensor([0.5, -0.5])
    for pos_scale in (BRIGHTNESS_POS_SCALE, CONTRAST_POS_SCALE, SHARPNESS_POS_SCALE):
        assert pos_scale == 1.0
        assert torch.allclose(_signed_scale(d, pos_scale, PHOTOMETRIC_NEG_SCALE), torch.tensor([0.5, -0.3]))
    assert COLOR_POS_SCALE == 2.0
    assert torch.allclose(_signed_scale(d, COLOR_POS_SCALE, PHOTOMETRIC_NEG_SCALE), torch.tensor([1.0, -0.3]))


def test_signed_scale_selects_per_image_not_per_batch():
    d = torch.tensor([0.5, -0.5, 0.5, -0.5])
    scaled = _signed_scale(d, COLOR_POS_SCALE, PHOTOMETRIC_NEG_SCALE)
    assert torch.allclose(scaled, torch.tensor([1.0, -0.3, 1.0, -0.3]))


def _mean_pixel_shift(op, batch, delta):
    return float((op(batch, delta) - batch).abs().mean())


@pytest.mark.parametrize("op", [contrast_op, color_op], ids=["contrast", "color"])
def test_multiplicative_ops_invert_under_the_unit_clamp(batch, op):
    """Known behaviour, not desirable: kornia's multiplicative contrast/saturation
    push pixels past 1.0 where the clamp truncates them, so the 0.6-scaled
    negative direction moves pixels FARTHER. One of the form mismatches
    TODO(calibration) is about."""
    assert _mean_pixel_shift(op, batch, -0.9) > _mean_pixel_shift(op, batch, 0.9)


@pytest.mark.parametrize("op", [brightness_op, sharpness_op], ids=["brightness", "sharpness"])
def test_additive_and_blend_ops_do_not_invert(batch, op):
    assert _mean_pixel_shift(op, batch, -0.9) < _mean_pixel_shift(op, batch, 0.9)


# --- label-safe warp ---------------------------------------------------------


def test_matrix_builder_is_the_matrix_the_op_warps_with(batch):
    """Image-only op and label path share one matrix builder; pin that the op
    really is warp_affine with it."""
    via_matrix = warp_affine(batch, rotate_matrix(batch, 0.5), dsize=batch.shape[-2:])
    assert torch.allclose(via_matrix, rotate_op(batch, 0.5), atol=1e-6)


def test_warp_image_and_label_keeps_image_gradient_and_drops_label_gradient(batch):
    magnitude = torch.tensor(0.5, requires_grad=True)
    label = torch.randint(0, 19, (batch.shape[0], *batch.shape[-2:]))
    img_out, label_out = warp_image_and_label(batch, label, rotate_matrix(batch, magnitude))
    assert img_out.requires_grad
    assert not label_out.is_floating_point()
    assert float(torch.autograd.grad(img_out.sum(), magnitude)[0].abs()) > 0.0


def test_warp_image_and_label_invents_no_class_ids(batch):
    """Nearest, never bilinear: a blend of class 3 and class 8 is not class 5."""
    label = torch.randint(0, 19, (batch.shape[0], *batch.shape[-2:]))
    _, label_out = warp_image_and_label(batch, label, rotate_matrix(batch, 0.4))
    assert set(label_out.unique().tolist()) - {LABEL_IGNORE_INDEX} <= set(label.unique().tolist())


def test_warp_image_and_label_fills_out_of_frame_with_the_ignore_index(batch):
    """kornia's default fill 0 would relabel every out-of-frame pixel as class 0."""
    label = torch.full((batch.shape[0], *batch.shape[-2:]), 7)
    _, label_out = warp_image_and_label(batch, label, translate_x_matrix(batch, 1.0))
    values = set(label_out.unique().tolist())
    assert LABEL_IGNORE_INDEX in values and 0 not in values


def test_warp_image_and_label_preserves_label_rank(batch):
    b, _, h, w = batch.shape
    matrix = shear_y_matrix(batch, -0.3)
    _, out3 = warp_image_and_label(batch, torch.randint(0, 19, (b, h, w)), matrix)
    _, out4 = warp_image_and_label(batch, torch.randint(0, 19, (b, 1, h, w)), matrix)
    assert out3.shape == (b, h, w)
    assert out4.shape == (b, 1, h, w)


def test_matrix_in_frame_is_a_no_op_for_matching_shapes(batch):
    matrix = rotate_matrix(batch, 0.5)
    assert _matrix_in_frame(matrix, (32, 32), (32, 32)) is matrix


@pytest.mark.parametrize("builder", [rotate_matrix, shear_x_matrix, translate_x_matrix],
                         ids=["rotate", "shear_x", "translate_x"])
def test_label_at_another_resolution_is_warped_by_the_same_visual_transform(builder):
    """mmseg val pipelines resize the image but load the label at native size.
    A label at 2x the image's resolution, warped via warp_image_and_label, must
    match the image's own label warped at the image's resolution (up to the
    nearest-neighbour edges the 2x grid resolves differently)."""
    torch.manual_seed(0)
    h, w = 32, 48
    images = torch.rand(1, 3, h, w)
    small = torch.zeros(1, h, w, dtype=torch.long)
    small[:, : h // 2, w // 2 :] = 1
    small[:, h // 2 :, : w // 2] = 2
    big = small.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)  # (1, 2h, 2w)

    matrix = builder(images, 0.7)
    _, warped_small = warp_image_and_label(images, small, matrix)
    _, warped_big = warp_image_and_label(images, big, matrix)
    agreement = (warped_big[:, ::2, ::2] == warped_small).float().mean()
    assert agreement > 0.95

    # And the uncorrected matrix (the pre-_matrix_in_frame behaviour) does not.
    from vocab.diff_ops import LABEL_IGNORE_INDEX as fill
    naive = warp_affine(
        big.unsqueeze(1).float(), matrix, dsize=big.shape[-2:], mode="nearest",
        padding_mode="fill", fill_value=torch.tensor([float(fill)]),
    ).round().long().squeeze(1)
    assert (naive[:, ::2, ::2] == warped_small).float().mean() < agreement
