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
from .randaug_default import AA_IMAGE_SIZE, ENHANCE_IDS, OP_ORDER

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


#: The image axis a translate moves along (width for X, height for Y).
_TRANSLATE_AXIS = {"TranslateX": -1, "TranslateY": -2}


def geometric_op(transform_id: str, sign: float, translate_ref: Optional[float] = None):
    """``translate_ref`` sets a translate's units: None, pixels (TrivialAugment's
    absolute range); otherwise the image extent the range is stated for, scaled to the
    image's own extent as torchvision does (RandAugment's 331, AugMix's 1, a fraction of
    the image)."""
    matrix_fn = _geometric_matrix(transform_id)
    axis = None if translate_ref is None else _TRANSLATE_AXIS.get(transform_id)

    def fn(images: Tensor, magnitude, label: Optional[Tensor] = None):
        m = sign * _batch_delta(magnitude, images)
        if axis is not None:
            m = m * (images.shape[axis] / translate_ref)
        matrix = matrix_fn(images, m)
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


# --- a split family -----------------------------------------------------------------
#
# Shared with `randaug_diff` and `augmix_diff`, which build RandAugment's and AugMix's
# ops the same way at their own ranges.


def split_specs(family: str, source: str, transform_id: str, hi: float, translate_ref: Optional[float] = None):
    """``<transform_id>_pos`` and ``_neg``: unsigned magnitude in [0, hi], identity at 0."""
    geometric = transform_id in GEOMETRIC_IDS
    for direction, sign in DIRECTIONS:
        yield OpSpec(
            name=f"{family}.{transform_id}_{direction}",
            family=family,
            kind="geometric" if geometric else "photometric",
            source=source,
            mag_range=(0.0, hi),
            identity_at=0.0,
            grad_mode="autograd",
            warps_label=geometric,
            fn=geometric_op(transform_id, sign, translate_ref) if geometric else enhance_op(transform_id, sign),
        )


def split_table(family: str, method_cls, transform_id: str, hi: float, translate_ref: Optional[float] = None):
    """The bins ``method_cls`` draws ``transform_id``'s magnitude from, read off its
    ``_AUGMENTATION_SPACE``, unsigned and in the split op's units (``translate_ref`` as
    in ``geometric_op``). The table is evaluated on a 331x331 image; a relative
    translate comes back in units of ``translate_ref``. torchvision's float32 can
    overshoot ``hi`` by round-off, which is clipped; more raises."""
    magnitudes_fn, _ = method_cls._AUGMENTATION_SPACE[transform_id]
    name = f"{family}.{transform_id}"
    size = AA_IMAGE_SIZE
    scale = 1.0 if translate_ref is None or transform_id not in _TRANSLATE_AXIS else translate_ref / size

    def table(num_bins):
        values = [abs(float(v)) * scale for v in magnitudes_fn(num_bins, int(size), int(size))]
        outside = [v for v in values if v > hi * (1 + 1e-6)]
        if outside:
            raise ValueError(f"{name}: torchvision's bins {outside} are outside [0, {hi}]")
        return tuple(min(v, hi) for v in values), False

    return table


def build_split_family(
    family: str,
    source: str,
    method_cls,
    split_ranges: Dict[str, float],
    exact: Dict[str, Tuple[OpSpec, Optional[Callable]]],
    translate_ref: Optional[float] = None,
) -> Tuple[Tuple[OpSpec, ...], Dict[str, Callable]]:
    """``(OPS, BINS)`` of a split family over RandAugment's 13 ops (``OP_ORDER`` minus
    Identity): each op in ``split_ranges`` (transform id -> ``hi``) as a ``_pos`` /
    ``_neg`` pair with ``method_cls``'s bins, and each op in ``exact`` (transform id ->
    ``(torchvision's OpSpec, its BinTable or None)``) as that op under this family's
    name."""
    if set(split_ranges) | set(exact) != set(OP_ORDER) - {"Identity"} or set(split_ranges) & set(exact):
        raise ValueError(f"{family}: split {sorted(split_ranges)} and exact {sorted(exact)} must partition the ops")
    specs, tables = [], {}
    for transform_id in OP_ORDER:
        if transform_id == "Identity":
            continue
        if transform_id in exact:
            spec, table = exact[transform_id]
            spec = dataclasses.replace(spec, name=f"{family}.{transform_id}", family=family)
            specs.append(spec)
            if table is not None:
                tables[spec.name] = table
            continue
        hi = split_ranges[transform_id]
        table = split_table(family, method_cls, transform_id, hi, translate_ref)
        for spec in split_specs(family, source, transform_id, hi, translate_ref):
            specs.append(spec)
            tables[spec.name] = table
    return tuple(specs), tables


def _exact(transform_id: str):
    """trivialaug_wide's op (torchvision's own) and its bins."""
    name = f"{trivialaug_wide.FAMILY}.{transform_id}"
    spec = next(s for s in trivialaug_wide.OPS if s.name == name)
    return spec, trivialaug_wide.BINS.get(name)


assert ENHANCE_IDS <= set(SPLIT_RANGES)

# Translate is absolute in TrivialAugmentWide (translate_ref None).
OPS, BINS = build_split_family(
    FAMILY, SOURCE, TrivialAugmentWide, SPLIT_RANGES, {t: _exact(t) for t in EXACT_IDS}
)
