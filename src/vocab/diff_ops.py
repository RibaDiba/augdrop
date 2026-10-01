"""Differentiable (autograd-compatible), GPU-batched op implementations, image + label.

Ported from Sensaug (third_party/Sensaug), numerics unchanged:

    sensaug/dataset/differentiable_augmentations.py
        the RGB/HSV rail ops, blur and noise -- itself a port of
        azshue/adversarial_data_augmentation's diffaug.py (Shu, Shen, Lin,
        Goldstein, "Adversarial Differentiable Data Augmentation for Autonomous
        Systems", ICRA 2021)
    sensaug/dataset/differentiable_augmentations_aa.py
        the AutoAugment-family ops built on kornia, and the label-safe warp
        (affine_matrix, _matrix_in_frame, warp_image_and_label)

tests/vocab/test_sensaug_parity.py checks these against the originals. What
changed is structure only: each geometric op is defined by its MATRIX builder
(`rotate_matrix`, ...), and both the image-only op and the image+label path
warp with that one matrix. Sensaug kept the calibration constants in two places
(the op wrappers and `geometric_affine_matrix`), which is where image and label
could have drifted apart.

The op functions here take a SIGNED magnitude (`delta`); the families bake the
direction into the op name (lighter_R / darker_R, rotate_pos / rotate_neg) and
pass an unsigned magnitude in [0, 1]. `photometric` and `geometric` at the
bottom adapt an op to the vocab's (images, magnitude, label) -> (images, label)
call.

Tensor contract (all public functions):
    - dtype: torch.float32
    - shape: (B, 3, H, W)          <- BCHW, batched
    - value range: [0, 1]
    - channel order: RGB (0=R, 1=G, 2=B)

Magnitude contract: every op takes its magnitude as EITHER a scalar (one
magnitude for the whole batch) OR a length-B tensor (one magnitude per image).
The per-image form is what lets one autograd pass measure d loss / d magnitude
separately for each image. That is also why no magnitude is ever rebuilt with
`torch.tensor([[...]])`: that silently DETACHES a tensor argument (it returns
requires_grad=False rather than raising), which would zero the very gradient the
probe exists to collect. Magnitudes are assembled with torch.stack throughout.
"""

import math
import warnings
from typing import Optional, Tuple, Union

import kornia.color
import torch
import torch.nn.functional as F
from kornia.enhance import adjust_brightness, adjust_contrast, adjust_saturation, sharpness
from kornia.geometry.transform import get_affine_matrix2d, warp_affine

Tensor = torch.Tensor

# --- channel index constants -------------------------------------------------
R, G, B = 0, 1, 2  # index into the RGB channel dim
H, S, V = 0, 1, 2  # index into kornia's HSV channel dim

RGB_RAIL_MAX = 1.0
# kornia.color.rgb_to_hsv returns hue in RADIANS [0, 2*pi), not normalized [0, 1].
_HSV_RAIL_MAX = {H: 2 * math.pi, S: 1.0, V: 1.0}
# Darkening V rails toward a nonzero floor instead of pure black (matches
# Sensaug's CPU HSVPerturbation).
V_DARKEN_FLOOR = 10.0 / 255.0

# Default Gaussian blur support, in taps. Sized to the [0, 1] magnitude contract:
# for `blur` the magnitude IS sigma, so 13 taps span +-6 sigma at the largest
# sigma the contract allows. Against Sensaug's previous 67-tap default the
# difference is exactly zero for sigma <= 0.5 and float32 round-off at sigma = 1,
# at 1/26th of the cost per pixel. Callers needing a larger sigma must pass a
# larger kernel_size explicitly; `blur` warns when sigma outruns its kernel.
BLUR_KERNEL_SIZE = (13, 13)

# --- AutoAugment-family calibration ------------------------------------------
#
# Every constant below is MIRRORED from the corresponding Sensaug CPU
# (LEGACY20_OPS) class, so a magnitude m in [0, 1] means the same amount of
# perturbation here as it does there. None of them are chosen freely.
#
# TODO(calibration), carried over from Sensaug: mirroring the constant is
# necessary but NOT sufficient. Measured against the CPU classes on a synthetic
# scene at m=1.0 (mean absolute difference on the 0-255 scale):
#
#     op            |legacy-ours|   legacy moved   ours moved   verdict
#     translate_x/y      0.00          62-101        62-101      exact
#     sharpness          0.00-0.09      1.0-7.5       1.0-7.6    matches
#     shear_x/y          4.7-5.5       36-48         36-48       matches
#     rotate             8.4-8.5       91            88          matches
#     color_neg         20.8          18.4          22.1         ~20% strong
#     color_pos         23.7          33.2          19.8         ~40% WEAK
#     brightness_neg    21.8          79.4         101.2         ~27% strong
#     contrast_pos      51.0          17.7          49.4         ~2.8x strong
#     brightness_pos    73.7          49.4         123.1         ~2.5x strong
#     contrast_neg      80.5          47.6          79.3         ~1.7x strong
#
# The geometric ops and sharpness are calibrated (their residual is
# bilinear-vs-INTER_NEAREST resampling). brightness, contrast and color are NOT,
# and they differ in FORM, not only in scale:
#   1. brightness -- torchvision MULTIPLIES (img * factor); kornia ADDS (img + factor).
#   2. contrast   -- torchvision blends toward the grayscale mean; kornia is a pure multiply.
#   3. color      -- torchvision blends toward grayscale; kornia scales S in HSV.
# Until those three are refit, their rows of R are on a different footing from
# the rest. The torchvision-faithful versions of these ops are the
# `randaug_default` / `trivialaug_wide` families, not these.

# Rotate: angle_deg = magnitude * 45. kornia takes degrees directly.
ROTATE_MAX_DEG = 45.0

# Shear: shear_factor = magnitude * 0.3, where 0.3 is a raw matrix ELEMENT
# ([[1, x, 0], [y, 1, 0]]). kornia takes a shear ANGLE and writes -tan(angle) in
# that slot, so the conversion is -atan(factor) (see _shear_angle).
SHEAR_MAX_FACTOR = 0.3

# Translate: tx = magnitude * (extent / 4), expressed as the fraction.
TRANSLATE_MAX_FRACTION = 0.25

# The photometric CPU classes are ASYMMETRIC: the negative direction carries its
# own smaller scale rather than being a sign flip of the positive one.
#
#   BrightnessTransform      factor = 1.0 + m         -> [1.0, 2.0]
#   NegativeBrightnessTransform      1.0 - 0.6*m      -> [1.0, 0.4]
#   ContrastTransform / NegativeContrastTransform     same 1.0 / -0.6 pair
#   SharpnessTransform / NegativeSharpnessTransform   same 1.0 / -0.6 pair
#   ColorTransform           factor = 1.0 + 2.0*m     -> [1.0, 3.0]
#   NegativeColorTransform          1.0 + 2.0*(-0.3*m) = 1.0 - 0.6*m -> [1.0, 0.4]
#
# Selected ELEMENTWISE on the sign of the magnitude, because with a per-image
# magnitude the sign can differ across the batch.
BRIGHTNESS_POS_SCALE = 1.0
CONTRAST_POS_SCALE = 1.0
SHARPNESS_POS_SCALE = 1.0
COLOR_POS_SCALE = 2.0
PHOTOMETRIC_NEG_SCALE = 0.6

# Out-of-frame label fill for geometric ops. 255 is mmseg's ignore index. kornia's
# default fill of 0 would silently relabel every out-of-frame pixel as class 0
# (road, on Cityscapes) and train against it.
LABEL_IGNORE_INDEX = 255


# --- magnitude helpers -------------------------------------------------------


def _as_delta(delta: Union[float, Tensor], reference: Tensor) -> Tensor:
    """Normalize a python float or a (possibly requires_grad) tensor onto
    `reference`'s dtype/device, preserving the autograd graph if `delta` is
    already a tensor."""
    if torch.is_tensor(delta):
        return delta.to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(float(delta), dtype=reference.dtype, device=reference.device)


def _broadcast_delta(delta: Tensor, batch_size: int) -> Tensor:
    """Reshape a scalar (0-dim) or per-image (B,) delta to (n, 1, 1, 1), n in
    {1, B}, so it broadcasts against a (B, C, H, W) batch. The per-image form
    makes autograd return a length-B gradient instead of one batch-averaged
    number."""
    d = delta.reshape(-1, 1, 1, 1)
    assert d.shape[0] in (1, batch_size), (
        f"delta must be a scalar or a length-{batch_size} vector (one per "
        f"image), got shape {tuple(delta.shape)}"
    )
    return d


def _batch_delta(delta: Union[float, Tensor], images: Tensor) -> Tensor:
    """Normalize a scalar-or-per-image magnitude to a length-B tensor, PRESERVING
    the autograd graph. kornia's affine builders want every parameter shaped
    (B,). A scalar is expanded rather than repeated: expand is a view, so the
    gradient of all B uses accumulates back onto the single scalar."""
    d = _as_delta(delta, images).reshape(-1)
    b = images.shape[0]
    if d.shape[0] == 1:
        return d.expand(b)
    assert d.shape[0] == b, (
        f"magnitude must be a scalar or a length-{b} vector (one per image), "
        f"got shape {tuple(d.shape)}"
    )
    return d


def _signed_scale(delta: Tensor, pos_scale: float, neg_scale: float) -> Tensor:
    """Scale the positive and negative directions of `delta` differently,
    elementwise. `neg_scale` is a positive number multiplying an already-negative
    delta, so `delta = -m` maps to `-neg_scale * m`."""
    return torch.where(delta >= 0, delta * pos_scale, delta * neg_scale)


# --- RGB / HSV rail ops ------------------------------------------------------


def _as_rail(rail: Union[float, Tensor], reference: Tensor) -> Tensor:
    """Rails are constants (no autograd) but may be per-image tensors -- see
    hsv_channel's V floor."""
    if torch.is_tensor(rail):
        return rail.to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(rail, dtype=reference.dtype, device=reference.device)


def _weighted_rail_perturb(
    channel: Tensor,
    delta: Tensor,
    rail_min: Union[float, Tensor],
    rail_max: Union[float, Tensor],
) -> Tensor:
    """out = channel * (1 - |delta|) + rail * |delta|, rail = rail_max if
    delta >= 0 else rail_min. Differentiable in `channel` and in the magnitude of
    `delta`; the SIGN of delta only selects the rail (a hard branch). With a
    per-image delta the sign can differ across the batch, so the rail is
    selected elementwise."""
    d = _broadcast_delta(delta, channel.shape[0])
    rail = torch.where(d >= 0, _as_rail(rail_max, d), _as_rail(rail_min, d))
    abs_delta = d.abs()
    return channel * (1.0 - abs_delta) + rail * abs_delta


def _replace_channel(tensor: Tensor, channel: int, num_channels: int, new_value: Tensor) -> Tensor:
    """Out-of-place `tensor[:, channel] = new_value`, via torch.cat. The in-place
    form corrupts the autograd graph when `tensor` is the output of a
    differentiable op whose backward reads that same slice (it did, on kornia's
    rgb_to_hsv / hsv_to_rgb)."""
    parts = [new_value if c == channel else tensor[:, c : c + 1] for c in range(num_channels)]
    return torch.cat(parts, dim=1)


def color_channel(images: Tensor, channel: int, delta: Union[float, Tensor]) -> Tensor:
    """Push one RGB channel toward 0 (delta<0) or 1 (delta>0)."""
    assert channel in (R, G, B), "channel must be R, G, or B (0, 1, 2)"
    delta_t = _as_delta(delta, images)
    new_value = _weighted_rail_perturb(
        images[:, channel : channel + 1], delta_t, rail_min=0.0, rail_max=RGB_RAIL_MAX
    )
    return _replace_channel(images, channel, 3, new_value)


def hsv_channel(images: Tensor, channel: int, delta: Union[float, Tensor]) -> Tensor:
    """Push one HSV channel toward its rail; RGB->HSV->RGB via kornia."""
    assert channel in (H, S, V), "channel must be H, S, or V (0, 1, 2)"
    delta_t = _as_delta(delta, images)
    hsv = kornia.color.rgb_to_hsv(images)
    rail_max = _HSV_RAIL_MAX[channel]
    rail_min: Union[float, Tensor] = 0.0
    if channel == V:
        # Darkening V rails toward a nonzero floor, not pure black. The sign can
        # vary across the batch, so this is an elementwise select.
        d = _broadcast_delta(delta_t, images.shape[0])
        rail_min = torch.where(d < 0, torch.full_like(d, V_DARKEN_FLOOR), torch.zeros_like(d))
    new_value = _weighted_rail_perturb(
        hsv[:, channel : channel + 1], delta_t, rail_min=rail_min, rail_max=rail_max
    )
    hsv_out = _replace_channel(hsv, channel, 3, new_value)
    return kornia.color.hsv_to_rgb(hsv_out).clamp(0.0, 1.0)


# --- blur / noise ------------------------------------------------------------


def _gaussian_kernel1d(size: int, sigma: Tensor) -> Tensor:
    """Hand-built 1D Gaussian kernel stack, shape (n, size), n = 1 for a scalar
    sigma and n = B for a per-image sigma. Hand-built (not kornia's
    gaussian_blur2d) so it is guaranteed differentiable w.r.t. sigma.

    One dimension, not two: an isotropic 2D Gaussian is the outer product of two
    1D ones, and sum(outer(a, b)) == sum(a) * sum(b), so normalizing each factor
    by its own sum reproduces the normalized 2D kernel exactly. `blur` convolves
    with the two factors in sequence (kh + kw taps instead of kh * kw)."""
    s = sigma.reshape(-1)  # (n,) -- a 0-dim sigma becomes (1,)
    ax = torch.arange(size, dtype=sigma.dtype, device=sigma.device) - (size - 1) / 2.0
    var = (2 * s**2 + 1e-12).view(-1, 1)  # (n, 1)
    gauss = torch.exp(-(ax**2).view(1, -1) / var)  # (n, size)
    return gauss / gauss.sum(dim=1, keepdim=True)


def _warn_if_undersupported(sigma: Tensor, kernel_size: Tuple[int, int]) -> None:
    """Warn when `kernel_size` is too small to resolve `sigma` (the Gaussian is
    then truncated and the blur weaker than requested). Reads sigma under
    no_grad: a diagnostic on the value, not part of what autograd differentiates."""
    with torch.no_grad():
        sigma_max = float(sigma.max())
    # The kernel's half-width (k - 1) / 2 taps must cover 3 sigma.
    supported = (min(kernel_size) - 1) / 6.0
    if sigma_max > supported:
        warnings.warn(
            f"blur: sigma up to {sigma_max:.3g} exceeds what kernel_size="
            f"{kernel_size} resolves (max ~{supported:.3g} at 3 sigma); the "
            f"Gaussian is truncated, so the blur is weaker than requested. Pass "
            f"a larger kernel_size (>= {2 * math.ceil(3 * sigma_max) + 1}).",
            RuntimeWarning,
            stacklevel=3,
        )


def blur(
    images: Tensor,
    sigma: Union[float, Tensor],
    kernel_size: Tuple[int, int] = BLUR_KERNEL_SIZE,
) -> Tensor:
    """Gaussian blur; magnitude == sigma directly (unsigned).

    kernel_size is deliberately NOT derived from sigma: for a per-image sigma the
    whole batch would share a size chosen by its largest sigma, and image i's
    output would then depend on the other images' magnitudes -- exactly the
    cross-image coupling the probe must exclude.

    NOTE: not numerically equivalent to cv2-based blurs, which derive the kernel
    size from the magnitude; magnitudes do not transfer between the two.
    """
    sigma_t = _as_delta(sigma, images).clamp(min=1e-3)
    _warn_if_undersupported(sigma_t, kernel_size)
    b, c = images.shape[0], images.shape[1]
    kh, kw = kernel_size
    kernel_y = _gaussian_kernel1d(kh, sigma_t).to(images.dtype)
    kernel_x = _gaussian_kernel1d(kw, sigma_t).to(images.dtype)

    if kernel_y.shape[0] == 1:
        # Scalar sigma: one kernel, depthwise over channels.
        kernel_y = kernel_y.view(1, 1, kh, 1).repeat(c, 1, 1, 1)
        kernel_x = kernel_x.view(1, 1, 1, kw).repeat(c, 1, 1, 1)
        out = F.conv2d(images, kernel_y, padding=(kh // 2, 0), groups=c)
        return F.conv2d(out, kernel_x, padding=(0, kw // 2), groups=c)

    assert kernel_y.shape[0] == b, (
        f"sigma must be a scalar or a length-{b} vector, got {kernel_y.shape[0]}"
    )
    # Per-image sigma: fold the batch into the channel dim and group by B*C so no
    # kernel sees another image. repeat_interleave, NOT repeat: the folded layout
    # is (b0c0, b0c1, b0c2, b1c0, ...), so image b's kernel must repeat C times
    # CONSECUTIVELY. `repeat` would tile, silently blurring one image with
    # another's sigma -- same shapes, no error, wrong numbers.
    kernel_y = kernel_y.view(b, 1, kh, 1).repeat_interleave(c, dim=0)  # (b*c, 1, kh, 1)
    kernel_x = kernel_x.view(b, 1, 1, kw).repeat_interleave(c, dim=0)  # (b*c, 1, 1, kw)
    folded = images.reshape(1, b * c, *images.shape[2:])
    out = F.conv2d(folded, kernel_y, padding=(kh // 2, 0), groups=b * c)
    out = F.conv2d(out, kernel_x, padding=(0, kw // 2), groups=b * c)
    return out.reshape(b, c, *out.shape[2:])


def gaussian_noise(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Additive Gaussian noise scaled by delta; scalar or length-B.

    Reparameterized: the gradient flows to the scale, not the sample. The sample
    is drawn fresh on every call, so two calls with the same delta differ --
    callers that need comparability across calls (the probe does) must fix the
    RNG themselves."""
    delta_t = _as_delta(delta, images)
    d = _broadcast_delta(delta_t, images.shape[0])
    noise = torch.randn(images.shape, dtype=images.dtype, device=images.device)
    return torch.clamp(images + noise * d, 0.0, 1.0)


# --- affine matrices ---------------------------------------------------------


def affine_matrix(
    images: Tensor,
    *,
    angle: Union[float, Tensor] = 0.0,
    tx: Union[float, Tensor] = 0.0,
    ty: Union[float, Tensor] = 0.0,
    sx: Union[float, Tensor] = 0.0,
    sy: Union[float, Tensor] = 0.0,
) -> Tensor:
    """The (B, 2, 3) affine matrix for a batch, about each image's center.

    Arguments are in kornia's NATIVE units: `angle` in degrees, `tx`/`ty` in
    PIXELS, `sx`/`sy` as shear angles in radians. Each may be a scalar or a
    length-B tensor and stays differentiable either way.
    """
    b, _, h, w = images.shape
    angle_t = _batch_delta(angle, images)
    tx_t = _batch_delta(tx, images)
    ty_t = _batch_delta(ty, images)
    sx_t = _batch_delta(sx, images)
    sy_t = _batch_delta(sy, images)

    # torch.stack, never torch.tensor([[...]]), which would detach the magnitude.
    translations = torch.stack([tx_t, ty_t], dim=-1)  # (B, 2)
    center = torch.stack(
        [torch.full_like(angle_t, w / 2.0), torch.full_like(angle_t, h / 2.0)], dim=-1
    )  # (B, 2)
    scale = torch.ones(b, 2, dtype=images.dtype, device=images.device)

    # get_affine_matrix2d returns (B, 3, 3) (checked on kornia 0.8.2; the shape
    # has moved between versions); warp_affine wants the top two rows.
    matrix = get_affine_matrix2d(translations, center, scale, angle_t, sx=sx_t, sy=sy_t)
    return matrix[:, :2, :]


def _shear_angle(delta: Union[float, Tensor], images: Tensor) -> Tensor:
    """[0, 1] magnitude -> the kornia shear angle that reproduces the shear
    matrix element `delta * SHEAR_MAX_FACTOR`. Negated because kornia writes
    -tan(angle) into that slot; atan because it writes the TANGENT."""
    factor = _batch_delta(delta, images) * SHEAR_MAX_FACTOR
    return -torch.atan(factor)


def rotate_matrix(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Rotate about the image center; delta=+1 -> +45 degrees."""
    return affine_matrix(images, angle=_batch_delta(delta, images) * ROTATE_MAX_DEG)


def shear_x_matrix(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Horizontal shear about the image center; delta=+1 -> factor 0.3."""
    return affine_matrix(images, sx=_shear_angle(delta, images))


def shear_y_matrix(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Vertical shear about the image center; delta=+1 -> factor 0.3."""
    return affine_matrix(images, sy=_shear_angle(delta, images))


def translate_x_matrix(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Horizontal translation; delta=+1 -> a quarter of the image WIDTH."""
    width = images.shape[-1]
    return affine_matrix(images, tx=_batch_delta(delta, images) * (width * TRANSLATE_MAX_FRACTION))


def translate_y_matrix(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Vertical translation; delta=+1 -> a quarter of the image HEIGHT."""
    height = images.shape[-2]
    return affine_matrix(images, ty=_batch_delta(delta, images) * (height * TRANSLATE_MAX_FRACTION))


def _warp(images: Tensor, matrix: Tensor) -> Tensor:
    """Bilinear (kornia's default), not the CPU classes' INTER_NEAREST: nearest is
    piecewise constant, so its derivative w.r.t. the magnitude is zero almost
    everywhere. Zero padding matches the CPU classes' borderValue=0."""
    h, w = images.shape[-2:]
    return warp_affine(images, matrix, dsize=(h, w))


# --- geometric ops (image only) ----------------------------------------------


def rotate_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    return _warp(images, rotate_matrix(images, delta))


def shear_x_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    return _warp(images, shear_x_matrix(images, delta))


def shear_y_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    return _warp(images, shear_y_matrix(images, delta))


def translate_x_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    return _warp(images, translate_x_matrix(images, delta))


def translate_y_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    return _warp(images, translate_y_matrix(images, delta))


# --- AutoAugment-family photometric ops --------------------------------------
#
# `1.0 + delta` centers "no change" at delta=0; valid because factor=1.0 is
# kornia's identity for contrast, saturation and sharpness. brightness is the
# exception: kornia's is additive, so delta is added directly.
#
# Every op clamps to [0, 1] afterwards. The clamp zeroes the gradient for pixels
# driven out of range, which is intended: a saturated pixel does not respond to
# more magnitude.


def brightness_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Brightness. kornia's adjust_brightness is ADDITIVE -- see TODO(calibration)."""
    d = _signed_scale(_batch_delta(delta, images), BRIGHTNESS_POS_SCALE, PHOTOMETRIC_NEG_SCALE)
    return adjust_brightness(images, d).clamp(0.0, 1.0)


def contrast_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Contrast. kornia's adjust_contrast is a pure multiply -- see TODO(calibration)."""
    d = _signed_scale(_batch_delta(delta, images), CONTRAST_POS_SCALE, PHOTOMETRIC_NEG_SCALE)
    return adjust_contrast(images, 1.0 + d).clamp(0.0, 1.0)


def sharpness_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Sharpness."""
    d = _signed_scale(_batch_delta(delta, images), SHARPNESS_POS_SCALE, PHOTOMETRIC_NEG_SCALE)
    return sharpness(images, 1.0 + d).clamp(0.0, 1.0)


def color_op(images: Tensor, delta: Union[float, Tensor]) -> Tensor:
    """Saturation ("color" in AutoAugment's naming). Positive scale is 2.0."""
    d = _signed_scale(_batch_delta(delta, images), COLOR_POS_SCALE, PHOTOMETRIC_NEG_SCALE)
    return adjust_saturation(images, 1.0 + d).clamp(0.0, 1.0)


# --- label-safe warp ---------------------------------------------------------


def _matrix_in_frame(matrix: Tensor, src_hw, dst_hw) -> Tensor:
    """Re-express a (B, 2, 3) affine written in `src_hw` PIXEL coordinates so it
    means the same visual transform in `dst_hw` pixel coordinates.

    The matrix bakes in the frame it was built for: the rotation/shear center is
    (w/2, h/2) and translations are in that frame's pixels. mmseg val pipelines
    Resize the image but load the label at its ORIGINAL size, so image and label
    routinely reach warp_image_and_label at different resolutions.

    The change of frame is M' = S M S^-1 with S = diag(w_dst/w_src, h_dst/h_src):
    the linear part's diagonal is invariant, the off-diagonals pick up sx/sy and
    sy/sx, and the translation column scales by (sx, sy). Identical shapes return
    `matrix` itself.
    """
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    if (src_h, src_w) == (dst_h, dst_w):
        return matrix

    sx = dst_w / src_w
    sy = dst_h / src_h
    out = matrix.clone()
    out[:, 0, 1] = out[:, 0, 1] * (sx / sy)
    out[:, 1, 0] = out[:, 1, 0] * (sy / sx)
    out[:, 0, 2] = out[:, 0, 2] * sx
    out[:, 1, 2] = out[:, 1, 2] * sy
    return out


def warp_image_and_label(
    images: Tensor,
    label: Tensor,
    matrix: Tensor,
    label_fill: float = LABEL_IGNORE_INDEX,
) -> Tuple[Tensor, Tensor]:
    """Apply one affine matrix to an image batch AND its label map.

    Four things must hold at once, and none is the default:
    - the IMAGE warp is bilinear and keeps the graph, so d loss / d magnitude flows;
    - the LABEL warp is nearest and takes `matrix.detach()`: no gradient through
      class ids, and no interpolated ids (a blend of class 3 and 8 is not class 5);
    - out-of-frame label pixels get the ignore index, not kornia's 0;
    - the label is warped by the matrix expressed in the LABEL's pixel frame
      (`_matrix_in_frame`), since the two rasters can differ in size.

    `label` may be (B, H, W) or (B, 1, H, W); the returned label matches the
    input's rank and is integral (long).
    """
    h, w = images.shape[-2:]
    images_out = warp_affine(images, matrix, dsize=(h, w), mode="bilinear")

    squeezed = label.dim() == 3
    label_in = label.unsqueeze(1) if squeezed else label
    label_hw = label_in.shape[-2:]
    label_out = warp_affine(
        label_in.to(images.dtype),
        _matrix_in_frame(matrix.detach(), (h, w), label_hw),
        dsize=label_hw,
        mode="nearest",
        padding_mode="fill",
        fill_value=torch.tensor([float(label_fill)], dtype=images.dtype, device=images.device),
    )
    label_out = label_out.round().long()
    return images_out, label_out.squeeze(1) if squeezed else label_out


# --- adapters to the vocab call ----------------------------------------------
#
# A family declares each op as a signed pair with the direction in the name and
# an unsigned magnitude, so `sign` is baked in here: the op sees
# `sign * magnitude`.


def photometric(op, sign: float = 1.0):
    """(images, delta) -> images, as (images, magnitude, label) -> (images, label).
    Photometric ops move no pixels, so the label passes through unchanged."""

    def fn(images: Tensor, magnitude, label: Optional[Tensor] = None):
        return op(images, sign * _as_delta(magnitude, images)), label

    return fn


def geometric(matrix_fn, sign: float = 1.0):
    """(images, delta) -> matrix, as (images, magnitude, label) -> (images, label).
    Image and label are warped by the same matrix, so they cannot drift apart."""

    def fn(images: Tensor, magnitude, label: Optional[Tensor] = None):
        matrix = matrix_fn(images, sign * _as_delta(magnitude, images))
        if label is None:
            return _warp(images, matrix), None
        return warp_image_and_label(images, label, matrix)

    return fn
