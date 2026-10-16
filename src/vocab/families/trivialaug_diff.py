"""TrivialAugment's wide op list, differentiable, each signed op split into _pos/_neg.

The ops of `trivialaug_wide` (Müller & Hutter 2021, Table 8; torchvision's
TrivialAugmentWide) at its ranges, built so autograd differentiates the magnitude,
for training and probing one method with one set of ops. Like the DIFF32 families,
the magnitude is unsigned and the direction is in the name, so every image is probed
in both directions and no row of R mixes the two:

    op               mag_range   units                         torchvision computes
    Rotate_pos/neg   [0, 135]    degrees, +/-                  rotate(angle=+/-m)
    ShearX/Y_pos/neg [0, 0.99]   shear rate, about the corner  affine(shear=deg(atan(+/-m)), center=[0, 0])
    TranslateX/Y_    [0, 32]     pixels, absolute              affine(translate=int(+/-m))
    Brightness_,
    Color_, Contrast_,
    Sharpness_pos/neg [0, 0.99]  factor 1 +/- m                adjust_*(factor=1 +/- m)
    Posterize        [2, 8]      bits             } torchvision's own op (trivialaug_wide's),
    Solarize         [0, 256]    threshold        } no derivative in the magnitude: the probe
    AutoContrast     none                         } fades them in (configs/host/
    Equalize         none                         } trivialaugment_diff.yaml)

Where these differ from torchvision. The geometric ops are torchvision's affine,
written in torch so the matrix carries the graph: the same matrix (torchvision's
`_get_inverse_affine_matrix`: rotate about the center, shear about the top-left
corner, translate), applied by kornia's bilinear warp instead of NEAREST, and the
translate is not truncated to whole pixels. The enhance ops are torchvision's float
blend, `clamp(f * x + (1 - f) * d)` with `d` = the op at factor 0, with no uint8
round trip. So on 8-bit input they equal torchvision up to the rounding of the
output (enhance) or the resampling (geometric); tests/vocab/test_trivialaug_diff.py
pins both. The label moves with the geometric ops (nearest, out-of-frame -> 255,
in the label's own frame), as in `geometric`.

`BINS` gives the magnitudes TrivialAugmentWide draws from, read off its
`_AUGMENTATION_SPACE`, unsigned: the method's random sign is the choice between
`_pos` and `_neg`.
"""

import dataclasses
from typing import Callable, Dict, Optional, Tuple

import torch
from torchvision.transforms.v2 import TrivialAugmentWide
from torchvision.transforms.v2 import functional as TF

from ..diff_ops import _as_delta, _batch_delta, _broadcast_delta, _warp, warp_image_and_label
from ..op import OpSpec
from . import trivialaug_wide
from .randaug_default import ENHANCE_IDS, OP_ORDER

FAMILY = "trivialaug_diff"
SOURCE = "torchvision TrivialAugmentWide (Müller & Hutter, 2021), differentiable in torch"

Tensor = torch.Tensor

#: transform id -> mag_range upper end (unsigned, identity at 0), for the split ops.
SPLIT_RANGES: Dict[str, float] = {
    "Rotate": 135.0,
    "Color": 0.99,
    "Contrast": 0.99,
    "Brightness": 0.99,
    "Sharpness": 0.99,
    "ShearX": 0.99,
    "ShearY": 0.99,
    "TranslateX": 32.0,
    "TranslateY": 32.0,
}

#: The ops kept as torchvision's own (`trivialaug_wide`'s implementation).
EXACT_IDS = ("AutoContrast", "Equalize", "Solarize", "Posterize")

GEOMETRIC_IDS = frozenset({"Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY"})

DIRECTIONS = (("pos", 1.0), ("neg", -1.0))


# --- geometric: torchvision's affine matrix, in torch -------------------------


def _affine_matrix(
    images: Tensor,
    *,
    angle: Optional[Tensor] = None,
    shear_x: Optional[Tensor] = None,
    shear_y: Optional[Tensor] = None,
    tx: Optional[Tensor] = None,
    ty: Optional[Tensor] = None,
    corner: bool = False,
) -> Tensor:
    """The (B, 2, 3) forward matrix, in pixel-index coordinates (kornia's warp_affine
    frame), of torchvision's affine with these parameters.

    torchvision's forward matrix is ``RSS (p - C) + C + T``, with ``RSS`` built from
    the rotation ``r`` and shear angles ``sx``, ``sy`` (radians) as in
    ``_get_inverse_affine_matrix`` (determinant 1). torchvision's coordinates are
    centered on the image; in index coordinates its center ``C`` = 0 is the image
    center ``((w - 1) / 2, (h - 1) / 2)`` and ``center=[0, 0]`` is the corner of the
    first pixel, (-0.5, -0.5). Every argument is (B,) degrees / rate / pixels and
    keeps the graph."""
    b, _, h, w = images.shape
    zero = torch.zeros(b, dtype=images.dtype, device=images.device)
    r = zero if angle is None else torch.deg2rad(angle)
    sx = zero if shear_x is None else torch.atan(shear_x)
    sy = zero if shear_y is None else torch.atan(shear_y)

    a = torch.cos(r - sy) / torch.cos(sy)
    bb = -torch.cos(r - sy) * torch.tan(sx) / torch.cos(sy) - torch.sin(r)
    c = torch.sin(r - sy) / torch.cos(sy)
    d = -torch.sin(r - sy) * torch.tan(sx) / torch.cos(sy) + torch.cos(r)

    cx, cy = (-0.5, -0.5) if corner else ((w - 1) / 2.0, (h - 1) / 2.0)
    tx = zero if tx is None else tx
    ty = zero if ty is None else ty
    row0 = torch.stack([a, bb, cx + tx - a * cx - bb * cy], dim=-1)
    row1 = torch.stack([c, d, cy + ty - c * cx - d * cy], dim=-1)
    return torch.stack([row0, row1], dim=1)


def _geometric_matrix(transform_id: str) -> Callable[[Tensor, Tensor], Tensor]:
    """(images, signed magnitude (B,)) -> forward matrix, as torchvision's
    TrivialAugmentWide dispatches ``transform_id``."""
    if transform_id == "Rotate":
        # F.rotate(angle=m) passes -m to the matrix: a positive angle turns counter-clockwise.
        return lambda images, m: _affine_matrix(images, angle=-m)
    if transform_id == "ShearX":
        return lambda images, m: _affine_matrix(images, shear_x=m, corner=True)
    if transform_id == "ShearY":
        return lambda images, m: _affine_matrix(images, shear_y=m, corner=True)
    if transform_id == "TranslateX":
        return lambda images, m: _affine_matrix(images, tx=m)
    if transform_id == "TranslateY":
        return lambda images, m: _affine_matrix(images, ty=m)
    raise ValueError(f"{transform_id} is not a geometric op")


def geometric_op(transform_id: str, sign: float):
    matrix_fn = _geometric_matrix(transform_id)

    def fn(images: Tensor, magnitude, label: Optional[Tensor] = None):
        matrix = matrix_fn(images, sign * _batch_delta(magnitude, images))
        if label is None:
            return _warp(images, matrix), None
        return warp_image_and_label(images, label, matrix)

    return fn


# --- enhance: torchvision's float blend ----------------------------------------

_DEGENERATE = {
    "Brightness": TF.adjust_brightness,
    "Color": TF.adjust_saturation,
    "Contrast": TF.adjust_contrast,
    "Sharpness": TF.adjust_sharpness,
}


def degenerate(transform_id: str, images: Tensor) -> Tensor:
    """torchvision's ``d``, the op at factor 0 on float images: black, the grayscale
    image, its mean gray, the 3x3-smoothed image with the border kept."""
    return _DEGENERATE[transform_id](images, 0.0)


def enhance_op(transform_id: str, sign: float):
    def fn(images: Tensor, magnitude, label: Optional[Tensor] = None):
        f = 1.0 + sign * _broadcast_delta(_as_delta(magnitude, images), images.shape[0])
        d = degenerate(transform_id, images)
        return (f * images + (1.0 - f) * d).clamp(0.0, 1.0), label

    return fn


# --- the family -------------------------------------------------------------------


def _split_specs(transform_id: str):
    geometric = transform_id in GEOMETRIC_IDS
    for direction, sign in DIRECTIONS:
        yield OpSpec(
            name=f"{FAMILY}.{transform_id}_{direction}",
            family=FAMILY,
            kind="geometric" if geometric else "photometric",
            source=SOURCE,
            mag_range=(0.0, SPLIT_RANGES[transform_id]),
            identity_at=0.0,
            grad_mode="autograd",
            warps_label=geometric,
            fn=geometric_op(transform_id, sign) if geometric else enhance_op(transform_id, sign),
        )


def _exact_spec(transform_id: str) -> OpSpec:
    """trivialaug_wide's op (torchvision's own) under this family's name."""
    spec = next(s for s in trivialaug_wide.OPS if s.name == f"{trivialaug_wide.FAMILY}.{transform_id}")
    return dataclasses.replace(spec, name=f"{FAMILY}.{transform_id}", family=FAMILY)


def _build_ops() -> Tuple[OpSpec, ...]:
    specs = []
    for transform_id in OP_ORDER:
        if transform_id == "Identity":
            continue
        if transform_id in EXACT_IDS:
            specs.append(_exact_spec(transform_id))
        else:
            specs.extend(_split_specs(transform_id))
    return tuple(specs)


def _split_table(transform_id: str, hi: float):
    magnitudes_fn, _ = TrivialAugmentWide._AUGMENTATION_SPACE[transform_id]
    name = f"{FAMILY}.{transform_id}"

    def table(num_bins):
        # Translate is absolute in TrivialAugmentWide, so the image size is unused.
        values = [abs(float(v)) for v in magnitudes_fn(num_bins, 331, 331)]
        outside = [v for v in values if v > hi * (1 + 1e-6)]
        if outside:
            raise ValueError(f"{name}: torchvision's bins {outside} are outside [0, {hi}]")
        return tuple(min(v, hi) for v in values), False

    return table


def _build_bins() -> Dict[str, Callable]:
    tables = {}
    for spec in OPS:
        base = spec.name.split(".", 1)[1]
        if base in EXACT_IDS:
            if spec.has_magnitude:
                tables[spec.name] = trivialaug_wide.BINS[f"{trivialaug_wide.FAMILY}.{base}"]
            continue
        transform_id = base.rsplit("_", 1)[0]
        tables[spec.name] = _split_table(transform_id, SPLIT_RANGES[transform_id])
    return tables


assert set(SPLIT_RANGES) | set(EXACT_IDS) == set(OP_ORDER) - {"Identity"}
assert ENHANCE_IDS <= set(SPLIT_RANGES)

OPS = _build_ops()
BINS = _build_bins()
