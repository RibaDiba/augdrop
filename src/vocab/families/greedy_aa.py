"""Greedy AutoAugment's ops, differentiable, each signed op split into _pos/_neg.

Naghizadeh, Abavisani & Metaxas, "Greedy AutoAugment" (Pattern Recognition Letters 2020,
arXiv 1908.00704), Table 1: 20 techniques, "the discretization setup, which is introduced in
AutoAugment". The released code (github.com/arnaghizadeh/greedy_auto_augment, no license
file) applies them through TensorFlow's AutoAugment CIFAR ``augmentation_transforms.py``
(Apache-2.0, its ``ALL_TRANSFORMS``, in that order), on 32 x 32 images. A level L in 0..9
(``PARAMETER_MAX`` = 10) maps to a magnitude by ``float_parameter(L, maxval) = L maxval / 10``
or ``int_parameter`` (the same, truncated); hosts/greedy_autoaugment.py holds the map. The
ranges below are each op's at ``PARAMETER_MAX``, the 32 px ones as fractions of the image:

    op                  mag_range       units                       the TF code (PIL, 8-bit, 32 px)
    FlipLR              none                                        transpose(FLIP_LEFT_RIGHT)
    FlipUD              none                                        transpose(FLIP_TOP_BOTTOM)
    AutoContrast        none                                        ImageOps.autocontrast
    Equalize            none                                        ImageOps.equalize
    Invert              none                                        ImageOps.invert
    Rotate_pos/neg      [0, 30]         degrees                     rotate(int_parameter(L, 30)), random sign
    Posterize           [0, 8]          bits; identity 8            posterize(4 - int_parameter(L, 4))
    CropBilinear        [0, 10/32]      crop per side, of W / H     crop((L, L, 32 - L, 32 - L)), resize back
    Solarize            [0, 256]        threshold; identity 256     solarize(256 - int_parameter(L, 256))
    Color_, Contrast_,
    Brightness_,        [0, 0.9]        factor 1 +/- m              <op>.enhance(float_parameter(L, 1.8) + 0.1)
    Sharpness_pos/neg
    ShearX/Y_pos/neg    [0, 0.3]        shear factor                AFFINE (1, v, 0, 0, 1, 0), random sign
    TranslateX/Y_       [0, 10/32]      fraction of W / H           AFFINE (1, 0, int_parameter(L, 10), ...),
      pos/neg                                                       random sign
    Cutout              [0, 20/32]      square side, of W           a int_parameter(L, 20) px square, gray
    Blur                none                                        filter(ImageFilter.BLUR)
    Smooth              none                                        filter(ImageFilter.SMOOTH)

Posterize's identity (8 bits) is outside what GAA ever applies (4 to 1 bits at levels 0..9,
0 at ``PARAMETER_MAX``), but an op's range must hold its identity, so it runs to 8.

How the ops are computed:

- **Rotate, ShearX/Y, TranslateX/Y and the enhance ops** are ``deepaa``'s (``split_specs``):
  the TF code's PIL calls are the same as DeepAA's ``augmentation.py`` (shear about the
  top-left pixel, rotate about the centre, PIL's AFFINE data with +v moving the content left
  / up), sampled bilinearly where PIL uses NEAREST so the magnitude differentiates, fill 0,
  the label moved; the enhance ops are PIL's ImageEnhance blend in float. The TF code draws a
  random sign for the geometric ops; here each direction is its own op (the user's choice,
  2026-11-07, in the search too). An enhance op's direction is its level (factor below 1 at
  L <= 4, the identity at L = 5, above 1 after), so it splits the same way.
- **Solarize, Posterize, AutoContrast, Equalize** are ``randaug_default``'s and **Invert**
  ``autoaugment``'s (torchvision on 8-bit images, which equals PIL's; ``deepaa`` docstring).
  ``grad_mode="none"``.
- **FlipLR** is ``madaug.flip``; **FlipUD** flips top to bottom, the label with it.
- **CropBilinear** is the crop box ``(c W, c H, W - c W, H - c H)`` resized back to (H, W),
  PIL's ``resize`` map (an output pixel's centre at ``left + (x + 0.5) box_w / W``, as
  ``deepaa.random_resized_crop`` builds it), bilinear as PIL's BILINEAR is when it enlarges.
  The box is not rounded to whole pixels (the TF code's L is a whole pixel at 32 px), so the
  magnitude differentiates. Geometric: the label moves (nearest).
- **Cutout** is the TF code's ``_cutout_pil_impl`` geometry: a centre uniform on the pixel
  grid (``randint(0, H)``, ``randint(0, W)``), the box ``[c - s // 2, c + s // 2)`` clipped
  at the border (exclusive, so it shrinks there), s = ``magnitude * W`` pixels, filled (125,
  122, 113). 0 is the identity (``size <= 0``). One centre per image in one ``torch.rand(B,
  2)`` on the CPU generator. It leaves the mask as it is (the user's choice for segmentation,
  as for Faster AutoAugment). ``grad_mode="none"``.
- **Blur and Smooth** are PIL's kernels at pixel scale (not scaled to the image size): BLUR's
  5 x 5 ring / 16, SMOOTH's 3 x 3 (1, 5 at the centre) / 13. PIL leaves the outer 2 (BLUR)
  or 1 (SMOOTH) pixels as they are and rounds the rest to 8 bits; this keeps the border and
  does not round. Smooth is ``trivialaug_diff.degenerate("Sharpness", .)``, torchvision's
  SMOOTH (PIL's Sharpness degenerate is that filter). No magnitude, ``grad_mode="none"``.

Which ops differentiate: the 18 split ops and CropBilinear, through autograd.
"""

import dataclasses
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..diff_ops import _batch_delta, _warp, warp_image_and_label
from ..op import OpSpec
from . import madaug
from .deepaa import exact_spec, forward_matrix, op_name, split_specs
from .randaug_default import _per_image
from .trivialaug_diff import degenerate

FAMILY = "greedy_aa"
SOURCE = "Greedy AutoAugment (Naghizadeh et al., 2020); TF AutoAugment augmentation_transforms.py"

Tensor = torch.Tensor

#: The TF code's ops are stated for 32 x 32 images.
IMAGE_SIZE = 32.0

#: ``augmentation_transforms.ALL_TRANSFORMS``, in its order.
OP_ORDER = (
    "FlipLR",
    "FlipUD",
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "CropBilinear",
    "Solarize",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Cutout",
    "Blur",
    "Smooth",
)

#: transform id -> the split op's mag_range upper end, at PARAMETER_MAX (the enhance ops'
#: 0.1 + 1.8 as 1 +/- 0.9; translates a fraction of the image).
SPLIT_RANGES: Dict[str, float] = {
    "Rotate": 30.0,
    "Color": 0.9,
    "Contrast": 0.9,
    "Brightness": 0.9,
    "Sharpness": 0.9,
    "ShearX": 0.3,
    "ShearY": 0.3,
    "TranslateX": 10.0 / IMAGE_SIZE,
    "TranslateY": 10.0 / IMAGE_SIZE,
}

#: CropBilinear's largest crop per side and Cutout's largest side, fractions of the image.
CROP_MAX = 10.0 / IMAGE_SIZE
CUTOUT_MAX = 20.0 / IMAGE_SIZE

#: ``_cutout_pil_impl``'s fill, in 8-bit RGB.
CUTOUT_FILL = (125, 122, 113)

#: PIL's ``ImageFilter.BLUR``: a 5 x 5 ring of ones over 16.
BLUR_KERNEL = (
    (1, 1, 1, 1, 1),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (1, 1, 1, 1, 1),
)
BLUR_SCALE = 16.0


def flip_ud(images: Tensor, magnitude=None, label: Optional[Tensor] = None):
    """``transpose(FLIP_TOP_BOTTOM)``: top to bottom, the label with it."""
    out = torch.flip(images, dims=(-2,))
    return out, (None if label is None else torch.flip(label, dims=(-2,)))


def crop_bilinear(images: Tensor, magnitude, label: Optional[Tensor] = None):
    """``_crop_impl``: the box ``(c W, c H, W - c W, H - c H)`` resized back to (H, W),
    c = ``magnitude`` (B,), through PIL's resize map. Graph kept."""
    c = _batch_delta(magnitude, images)
    h, w = images.shape[-2:]
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    scale = one - 2.0 * c
    data = torch.stack((scale, zero, c * w, zero, scale, c * h), dim=-1)
    matrix = forward_matrix(images, data)
    if label is None:
        return _warp(images, matrix), None
    return warp_image_and_label(images, label, matrix)


def cutout(images: Tensor, magnitude, label: Optional[Tensor] = None):
    """``_cutout_pil_impl`` per image (module docstring): a square of side ``magnitude *
    W`` pixels, ``size // 2`` either side of a centre on the pixel grid, exclusive and
    clipped, filled (125, 122, 113)."""
    b, _, h, w = images.shape
    mags = _per_image(magnitude, b, images.dtype)
    centres = torch.rand(b, 2)
    fill = torch.tensor(CUTOUT_FILL, dtype=images.dtype, device=images.device).view(3, 1, 1) / 255.0
    out = images.clone()
    for i, m in enumerate(mags):
        half = int(m * w) // 2
        if half <= 0:
            continue
        y = min(h - 1, int(float(centres[i, 0]) * h))
        x = min(w - 1, int(float(centres[i, 1]) * w))
        y0, y1 = max(0, y - half), min(h, y + half)
        x0, x1 = max(0, x - half), min(w, x + half)
        out[i, :, y0:y1, x0:x1] = fill
    return out, label


def _filter_kept_border(images: Tensor, kernel: Tensor) -> Tensor:
    """``kernel`` (k, k) over each channel, the outer ``k // 2`` pixels left as they are
    (PIL's ``ImagingFilter``)."""
    k = kernel.shape[-1]
    r = k // 2
    channels = images.shape[1]
    weight = kernel.to(images.dtype).to(images.device).expand(channels, 1, k, k)
    inner = F.conv2d(images, weight, groups=channels)
    out = images.clone()
    out[..., r:-r, r:-r] = inner
    return out


def blur(images: Tensor, magnitude=None, label: Optional[Tensor] = None):
    """``filter(ImageFilter.BLUR)`` in float, the 2-pixel border kept."""
    kernel = torch.tensor(BLUR_KERNEL, dtype=torch.float64) / BLUR_SCALE
    return _filter_kept_border(images, kernel), label


def smooth(images: Tensor, magnitude=None, label: Optional[Tensor] = None):
    """``filter(ImageFilter.SMOOTH)`` in float, the 1-pixel border kept: torchvision's
    Sharpness degenerate, the same kernel."""
    return degenerate("Sharpness", images), label


def _no_magnitude(name: str, kind: str, fn) -> OpSpec:
    return OpSpec(
        name=op_name(FAMILY, name),
        family=FAMILY,
        kind=kind,
        source=SOURCE,
        mag_range=None,
        identity_at=None,
        grad_mode="none",
        warps_label=kind == "geometric",
        fn=fn,
    )


def _specs():
    for transform_id in OP_ORDER:
        if transform_id in SPLIT_RANGES:
            yield from split_specs(FAMILY, SOURCE, transform_id, SPLIT_RANGES[transform_id])
        elif transform_id == "FlipLR":
            yield _no_magnitude("FlipLR", "geometric", madaug.flip)
        elif transform_id == "FlipUD":
            yield _no_magnitude("FlipUD", "geometric", flip_ud)
        elif transform_id == "Posterize":
            # randaug_default's Posterize at GAA's bits (4 to 1), its range run down to 0.
            yield dataclasses.replace(exact_spec(FAMILY, "Posterize"), source=SOURCE, mag_range=(0.0, 8.0))
        elif transform_id in ("AutoContrast", "Equalize", "Invert", "Solarize"):
            yield dataclasses.replace(exact_spec(FAMILY, transform_id), source=SOURCE)
        elif transform_id == "CropBilinear":
            yield OpSpec(
                name=op_name(FAMILY, "CropBilinear"),
                family=FAMILY,
                kind="geometric",
                source=SOURCE,
                mag_range=(0.0, CROP_MAX),
                identity_at=0.0,
                grad_mode="autograd",
                warps_label=True,
                fn=crop_bilinear,
            )
        elif transform_id == "Cutout":
            yield OpSpec(
                name=op_name(FAMILY, "Cutout"),
                family=FAMILY,
                kind="photometric",
                source=SOURCE,
                mag_range=(0.0, CUTOUT_MAX),
                identity_at=0.0,
                grad_mode="none",
                warps_label=False,
                fn=cutout,
            )
        elif transform_id == "Blur":
            yield _no_magnitude("Blur", "photometric", blur)
        elif transform_id == "Smooth":
            yield _no_magnitude("Smooth", "photometric", smooth)
        else:
            raise AssertionError(transform_id)


OPS = tuple(_specs())
