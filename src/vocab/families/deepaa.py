"""Deep AutoAugment's ops, differentiable, each signed op split into _pos/_neg.

Zheng, Zhang, Yan & Zhang, "Deep AutoAugment" (ICLR 2022, arXiv 2203.06172), Appendix A
Table 7, and its code (github.com/AIoT-MLSys-Lab/DeepAA, vendored as the submodule
``third_party/DeepAA``): ``augmentation.py`` (the search's ops) and
``DeepAA_evaluate/deep_autoaugment.py`` (the trainer's). The search space is the ImageNet
list (``DeepAA_utils.aug_op_imagenet_list``), the one the published ImageNet policies
(``policy_port/policy_DeepAA_imagenet_{1,2}.npz``) were searched over: 18 ops, 139
(op, magnitude) transforms per layer. Identity is not an op here: it is the host's
``NO_OP`` (hosts/deep_autoaugment.py), as everywhere in dropaug.

    op                  mag_range   units                     DeepAA computes (PIL, 8-bit)
    ShearX/Y_pos/neg    [0, 0.3]    shear factor v            img.transform(AFFINE, (1, v, 0, 0, 1, 0))
                                                              (Y: (1, 0, 0, v, 1, 0))
    TranslateX/Y_       [0, 0.45]   fraction of W / H         AFFINE (1, 0, v W, 0, 1, 0) (Y: v H)
      pos/neg
    Rotate_pos/neg      [0, 30]     degrees                   img.rotate(v), about (W/2, H/2)
    AutoContrast        none                                  ImageOps.autocontrast
    Invert              none                                  ImageOps.invert
    Equalize            none                                  ImageOps.equalize
    Solarize            [0, 256]    threshold; identity 256   ImageOps.solarize(img, v)
    Posterize           [4, 8]      bits, int(v); identity 8  ImageOps.posterize(img, int(v))
    Contrast_, Color_,
    Brightness_,        [0, 0.9]    factor 1 +/- m            ImageEnhance.<op>(img).enhance(v),
    Sharpness_pos/neg                                         v in [0.1, 1.9]
    Flip                none                                  ImageOps.mirror (RandFlip's p = 0.5
                                                              coin is the host's)
    Cutout              none        a 16/224 W square         RandCutout: a 16 px square, (125, 123, 114)
    RandomResizedCrop   none                                  RandResizeCrop_imagenet

``_pos`` is DeepAA's positive v: the PIL data above with +m. For the translates that
moves the content left / up, for Rotate counter-clockwise, and for the enhance ops it is a
factor above 1. Both signs are in DeepAA's search space (its grid runs over [-0.3, 0.3],
[0.1, 1.9], ...), so the policy itself says which direction it applies; the host maps
each signed grid value to its op.

How the ops are computed:

- **The geometric ops are PIL's maps.** ``pil_data`` builds the AFFINE data PIL is
  given: an output pixel's continuous coordinates (its centre at index + 0.5) mapped to
  the input's. ``forward_matrix`` inverts it into kornia's forward matrix in index
  coordinates (``diff_ops._warp``, ``warp_image_and_label``). PIL samples with NEAREST
  (``transform`` / ``rotate`` defaults), which has no derivative in v; this samples
  bilinearly so the magnitude differentiates. The fill is 0, PIL's default. The label
  moves (nearest, out-of-frame -> 255, in the label's own frame).
- **The enhance ops** are ``trivialaug_diff.enhance_op``: torchvision's float blend
  ``f x + (1 - f) d`` with PIL's degenerate images (black; the grayscale at 0.299 /
  0.587 / 0.114; its mean; the 3x3 smooth with the border kept). PIL also rounds the
  gray image and the blend to 8 bits; that rounding is not reproduced (it would zero the
  gradient).
- **Solarize, Posterize, AutoContrast, Equalize** are ``randaug_default``'s, **Invert**
  is ``autoaugment``'s: torchvision on 8-bit images, which equals PIL pixel for pixel
  (Solarize's fractional thresholds included: DeepAA's are 256 m / 12), except
  AutoContrast, one 8-bit level off on about 1.6% of values (torchvision builds its
  lookup in float32, PIL in float64). ``grad_mode="none"``.
- **Flip** is ``madaug.flip``.
- **Cutout is RandCutout, not RandCutout60.** The repo's ImageNet list
  (``aug_op_imagenet_list``) names RandCutout60, but both published ImageNet policies
  name op 16 ``RandCutout``, so that is the op they were searched with:
  ``augmentation.RandCutout``, a 16 px square (v // 2 either side of a uniform centre,
  clipped at the border, PIL's rectangle inclusive) on 224 x 224 images. Here the side is
  16/224 of the width. The released trainer's ``RandCutout`` calls ``Cutout_default``,
  which asserts 32 x 32 images, so as released it cannot apply this op to ImageNet; the
  search's op is the one ported.
- **RandomResizedCrop** is ``RandResizeCrop_imagenet``: scale (0.08, 1), ratio (3/4,
  4/3), the box resized back. Two departures: the output keeps the input's size (DeepAA
  resizes to 224, the ImageNet training size; here the crop size is the recipe's), and it
  resamples bilinearly where PIL uses bicubic. Kept as DeepAA has it: its ``get_params``
  falls back to the central crop after one failed attempt (the fallback is indented
  inside the 10-attempt loop), where torchvision tries 10 times.

Cutout and RandomResizedCrop draw their randomness per image (``torch.rand`` on the CPU
generator, so the framework's seed governs it) and have no magnitude: they are
``stochastic`` (vocab/op.py). Cutout leaves the mask as it is, as Faster AutoAugment's
does (the user's choice for segmentation); RandomResizedCrop moves it.

Which ops differentiate: the 18 split ops, through autograd. The rest are
``grad_mode="none"``.
"""

import dataclasses
import math
from typing import Dict, Optional

import torch

from ..diff_ops import _batch_delta, _warp, warp_image_and_label
from ..op import OpSpec
from . import autoaugment, madaug, randaug_default
from .faster_aa import _matrix
from .randaug_default import _per_image
from .trivialaug_diff import enhance_op

FAMILY = "deepaa"
SOURCE = "Deep AutoAugment (Zheng et al., ICLR 2022), augmentation.py"

Tensor = torch.Tensor

#: DeepAA's ImageNet list (``aug_op_imagenet_list``) without Identity, in its order.
OP_ORDER = (
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Rotate",
    "AutoContrast",
    "Invert",
    "Equalize",
    "Solarize",
    "Posterize",
    "Contrast",
    "Color",
    "Brightness",
    "Sharpness",
    "Flip",
    "Cutout",
    "RandomResizedCrop",
)

#: transform id -> the split op's mag_range upper end: DeepAA's range [-hi, hi] (the
#: enhance ops' [0.1, 1.9] as 1 +/- 0.9).
SPLIT_RANGES: Dict[str, float] = {
    "ShearX": 0.3,
    "ShearY": 0.3,
    "TranslateX": 0.45,
    "TranslateY": 0.45,
    "Rotate": 30.0,
    "Contrast": 0.9,
    "Color": 0.9,
    "Brightness": 0.9,
    "Sharpness": 0.9,
}

GEOMETRIC_IDS = frozenset({"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"})

DIRECTIONS = (("pos", 1.0), ("neg", -1.0))

#: The torchvision ops that equal PIL's, reused under this family's names.
EXACT_SOURCES = {
    "AutoContrast": "randaug_default.AutoContrast",
    "Invert": "autoaugment.Invert",
    "Equalize": "randaug_default.Equalize",
    "Solarize": "randaug_default.Solarize",
    "Posterize": "randaug_default.Posterize",
}

#: DeepAA's (and Fast AutoAugment's) Cutout fill, in 8-bit RGB.
CUTOUT_FILL = (125, 123, 114)

#: RandCutout's side, 16 px on DeepAA's 224 x 224 ImageNet images, as a fraction of W.
CUTOUT_SIDE = 16.0 / 224.0

#: RandResizeCrop_imagenet's area scale and aspect ratio ranges (torchvision's).
CROP_SCALE = (0.08, 1.0)
CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)


# --- geometric: PIL's affine maps ---------------------------------------------------


def pil_data(transform_id: str, v: Tensor, h: int, w: int) -> Tensor:
    """(B, 6): the AFFINE data ``(a, b, c, d, e, f)`` PIL is given for a geometric op at
    signed magnitude ``v`` (B,): the input point ``(a x + b y + c, d x + e y + f)`` an
    output point ``(x, y)`` samples, in PIL's continuous coordinates (pixel i spans
    [i, i + 1)). Graph kept."""
    one = torch.ones_like(v)
    zero = torch.zeros_like(v)
    if transform_id == "ShearX":
        rows = (one, v, zero, zero, one, zero)
    elif transform_id == "ShearY":
        rows = (one, zero, zero, v, one, zero)
    elif transform_id == "TranslateX":
        rows = (one, zero, v * w, zero, one, zero)
    elif transform_id == "TranslateY":
        rows = (one, zero, zero, zero, one, v * h)
    elif transform_id == "Rotate":
        # Image.rotate(v): angle = -radians(v), matrix (cos, sin, 0, -sin, cos, 0) about
        # the centre (w / 2, h / 2).
        theta = -torch.deg2rad(v)
        cos, sin = torch.cos(theta), torch.sin(theta)
        cx, cy = w / 2.0, h / 2.0
        rows = (cos, sin, -cos * cx - sin * cy + cx, -sin, cos, sin * cx - cos * cy + cy)
    else:
        raise ValueError(f"{transform_id} is not a geometric op")
    return torch.stack(rows, dim=-1)


def forward_matrix(images: Tensor, data: Tensor) -> Tensor:
    """(B, 2, 3): the forward matrix, in pixel-index coordinates (kornia's
    ``warp_affine`` frame, a pixel's centre at its index), of the PIL map ``data``
    (B, 6). Index u is PIL's continuous u + 0.5, so the inverse map in index
    coordinates is ``A u + (t + A 1/2 - 1/2)``; this inverts it. Graph kept."""
    a, b, c, d, e, f = data.unbind(-1)
    c = c + 0.5 * (a + b) - 0.5
    f = f + 0.5 * (d + e) - 0.5
    det = a * e - b * d
    ia, ib, ic, id_ = e / det, -b / det, -d / det, a / det
    return _matrix(images, ia, ib, ic, id_, -(ia * c + ib * f), -(ic * c + id_ * f))


def geometric_op(transform_id: str, sign: float):
    def fn(images: Tensor, magnitude, label: Optional[Tensor] = None):
        v = sign * _batch_delta(magnitude, images)
        h, w = images.shape[-2:]
        matrix = forward_matrix(images, pil_data(transform_id, v, h, w))
        if label is None:
            return _warp(images, matrix), None
        return warp_image_and_label(images, label, matrix)

    return fn


# --- the random ops ----------------------------------------------------------------


def cutout(images: Tensor, magnitude, label: Optional[Tensor] = None):
    """DeepAA's ``RandCutout`` / ``Cutout_default`` geometry (AutoAugment's too), per
    image: a square of side ``v = magnitude * W`` pixels about a centre uniform on
    [0, W) x [0, H), corners ``int(clip(c -/+ v // 2))``, filled (125, 123, 114) both
    ends inclusive (PIL's rectangle), so it shrinks at the border. 0 is the identity
    (``Cutout``'s ``v <= 0`` guard). One centre per image whatever its magnitude, drawn
    in one ``torch.rand(B, 2)`` on the CPU generator, so the framework's seed governs it
    and bumping one image's magnitude leaves the others alone."""
    b, _, h, w = images.shape
    mags = _per_image(magnitude, b, images.dtype)
    centres = torch.rand(b, 2)
    fill = torch.tensor(CUTOUT_FILL, dtype=images.dtype, device=images.device).view(3, 1, 1) / 255.0
    out = images.clone()
    for i, m in enumerate(mags):
        if m <= 0.0:
            continue
        half = (m * w) // 2
        x = float(centres[i, 0]) * w
        y = float(centres[i, 1]) * h
        x0, x1 = int(min(w, max(0.0, x - half))), int(min(w, max(0.0, x + half)))
        y0, y1 = int(min(h, max(0.0, y - half))), int(min(h, max(0.0, y + half)))
        out[i, :, y0 : y1 + 1, x0 : x1 + 1] = fill
    return out, label


def rand_cutout(images: Tensor, magnitude=None, label: Optional[Tensor] = None):
    """``RandCutout``: ``cutout`` at its one size, 16/224 of the width."""
    return cutout(images, CUTOUT_SIDE, label)


def crop_box(u: Tensor, h: int, w: int):
    """``RandResizeCrop_imagenet.get_params`` from four uniforms ``u``: ``(left, top,
    width, height)``. One attempt, then the central crop: DeepAA's fallback sits inside
    its 10-attempt loop."""
    area = float(w * h)
    log_lo, log_hi = math.log(CROP_RATIO[0]), math.log(CROP_RATIO[1])
    target = area * (CROP_SCALE[0] + (CROP_SCALE[1] - CROP_SCALE[0]) * float(u[0]))
    aspect = math.exp(log_lo + (log_hi - log_lo) * float(u[1]))
    cw = round(math.sqrt(target * aspect))
    ch = round(math.sqrt(target / aspect))
    if 0 < cw <= w and 0 < ch <= h:
        # random.randint(0, n) is inclusive of n.
        return int(float(u[3]) * (w - cw + 1)), int(float(u[2]) * (h - ch + 1)), cw, ch
    in_ratio = float(w) / float(h)
    if in_ratio < min(CROP_RATIO):
        cw, ch = w, round(w / min(CROP_RATIO))
    elif in_ratio > max(CROP_RATIO):
        cw, ch = round(h * max(CROP_RATIO)), h
    else:
        cw, ch = w, h
    return (w - cw) // 2, (h - ch) // 2, cw, ch


def random_resized_crop(images: Tensor, magnitude=None, label: Optional[Tensor] = None):
    """``RandResizeCrop_imagenet``, per image: a ``crop_box`` resized back to (H, W),
    PIL's ``resize(box=...)`` map (an output pixel's centre at ``left + (x + 0.5) cw /
    W``). Four uniforms per image in one ``torch.rand(B, 4)`` on the CPU generator."""
    b, _, h, w = images.shape
    draws = torch.rand(b, 4)
    rows = []
    for i in range(b):
        left, top, cw, ch = crop_box(draws[i], h, w)
        rows.append((cw / w, 0.0, float(left), 0.0, ch / h, float(top)))
    data = torch.tensor(rows, dtype=images.dtype, device=images.device)
    matrix = forward_matrix(images, data)
    if label is None:
        return _warp(images, matrix), None
    return warp_image_and_label(images, label, matrix)


# --- the specs ----------------------------------------------------------------------


def op_name(family: str, transform_id: str, direction: Optional[str] = None) -> str:
    return f"{family}.{transform_id}" if direction is None else f"{family}.{transform_id}_{direction}"


def split_specs(family: str, source: str, transform_id: str, hi: float):
    """``<transform_id>_pos`` / ``_neg`` over [0, hi], identity 0, autograd: PIL's map
    (geometric) or PIL's ImageEnhance blend (``trivialaug_diff.enhance_op``)."""
    geometric = transform_id in GEOMETRIC_IDS
    for direction, sign in DIRECTIONS:
        yield OpSpec(
            name=op_name(family, transform_id, direction),
            family=family,
            kind="geometric" if geometric else "photometric",
            source=source,
            mag_range=(0.0, hi),
            identity_at=0.0,
            grad_mode="autograd",
            warps_label=geometric,
            fn=geometric_op(transform_id, sign) if geometric else enhance_op(transform_id, sign),
        )


def exact_spec(family: str, transform_id: str) -> OpSpec:
    """The torchvision op that equals PIL's, renamed into ``family``."""
    source_name = EXACT_SOURCES[transform_id]
    module = randaug_default if source_name.startswith("randaug_default.") else autoaugment
    spec = next(s for s in module.OPS if s.name == source_name)
    return dataclasses.replace(spec, name=op_name(family, transform_id), family=family)


def _specs():
    for transform_id in OP_ORDER:
        if transform_id in SPLIT_RANGES:
            yield from split_specs(FAMILY, SOURCE, transform_id, SPLIT_RANGES[transform_id])
        elif transform_id in EXACT_SOURCES:
            yield exact_spec(FAMILY, transform_id)
        elif transform_id == "Flip":
            yield OpSpec(
                name=op_name(FAMILY, "Flip"),
                family=FAMILY,
                kind="geometric",
                source=SOURCE,
                mag_range=None,
                identity_at=None,
                grad_mode="none",
                warps_label=True,
                fn=madaug.flip,
            )
        elif transform_id == "Cutout":
            yield OpSpec(
                name=op_name(FAMILY, "Cutout"),
                family=FAMILY,
                kind="photometric",
                source=SOURCE,
                mag_range=None,
                identity_at=None,
                grad_mode="none",
                warps_label=False,
                fn=rand_cutout,
                stochastic=True,
            )
        elif transform_id == "RandomResizedCrop":
            yield OpSpec(
                name=op_name(FAMILY, "RandomResizedCrop"),
                family=FAMILY,
                kind="geometric",
                source=SOURCE,
                mag_range=None,
                identity_at=None,
                grad_mode="none",
                warps_label=True,
                fn=random_resized_crop,
                stochastic=True,
            )
        else:
            raise AssertionError(transform_id)


OPS = tuple(_specs())
