"""Faster AutoAugment's ops, differentiable, each signed op split into _pos/_neg.

Hataya, Zdenek, Yoshizoe & Nakayama, "Faster AutoAugment: Learning Augmentation
Strategies using Backpropagation" (ECCV 2020, arXiv 1911.06987), Table 2: 17 ops. The
implementations are the authors' (github.com/moskomule/dda, ``dda/functional.py`` and
``dda/operations.py``, MIT), in their units: dda learns mu in [0, 1] and multiplies it by
each op's ``magnitude_scale``, and the 9 ops with ``flip_magnitude`` take a random sign.
Here the sign is the op: ``_pos`` and ``_neg`` are separate ops, in the vocab and in the
policy search (hosts/faster_autoaugment), so the search can learn a direction. That is a
departure from the paper, which learns one mu and p per op and flips its sign at p = 0.5.

    op                 mag_range    units                         dda computes
    ShearX/Y_pos/neg   [0, 0.3]     shear factor, about the       kornia.shear: [[1, m, 0], [0, 1, 0]]
                                    top-left pixel                (Y: [[1, 0, 0], [m, 1, 0]])
    TranslateX/Y_      [0, 0.45]    fraction of the width /       kornia.translate(m * W) (Y: m * H)
      pos/neg                       height
    Rotate_pos/neg     [0, 30]      degrees, about the center     kornia.rotate(m)
    HorizontalFlip     none                                       img.flip(-1)
    Solarize           [0, 256/255] threshold; identity at 256/255  where(img < m, img, 1 - img)
    Posterize          [0, 1]       8 - floor(8 (1 - m)) bits;    see below
                                    identity at 1
    Invert             none                                       1 - img
    Contrast_, Color_,
    Brightness_,       [0, 1]       factor 1 +/- m                blend toward the mean gray / the
    Sharpness_pos/neg                                             gray image / black / the 3x3 smooth
    AutoContrast       none                                       per-channel min-max stretch
    Equalize           none                                       per-channel histogram equalization
    Cutout             [0, 0.2]     square side, fraction of W    (not in dda; Fast AutoAugment's)
    SamplePairing      [0, 1]       blend weight of a partner     (1 - m) x + m x[randperm]

``_pos`` is the direction a factor or angle grows in (factor 1 + m, +m degrees, +m
shear, +m translate); dda draws both with p = 0.5, so the naming is ours.

What dda ships and the paper does not, and what this keeps:

- **TranslateX.** dda's op list (``Policy.dda_operations``) has TranslateY twice and no
  TranslateX. The paper lists both; this has both, TranslateX being dda's
  ``translate_x``.
- **Cutout and SamplePairing.** In the paper's Table 2, not in dda's op list (dda's
  ``cutout`` raises NotImplementedError; the paper used a CUDA kernel it did not
  release). The paper says its search space is Fast AutoAugment's, so Cutout is Fast
  AutoAugment's ``Cutout`` (v in [0, 0.2], ``CutoutAbs``), which `madaug` already
  ports exactly; it is used from there. SamplePairing is dda's ``sample_pairing`` (the
  partner is the image a ``randperm`` of the batch puts in its place, which can be the
  image itself, a no-op; mu in [0, 1]); the paper's examples
  (Fig. 5) show mu up to 0.8, so it is not Fast AutoAugment's [0, 0.4].
- **Posterize is fixed.** dda's shifts left, then right
  (``(x << shift) >> shift``), which on int64 is the identity: its Posterize only
  truncates to 8 bits whatever mu is. This shifts right, then left, which zeroes the
  low ``shift`` bits, as posterize means, keeping dda's mu -> shift map
  (``shift = floor(8 (1 - mu))``: mu = 1 keeps 8 bits, mu = 0 keeps none).
- **Solarize's range runs to 256/255.** dda's threshold is mu in [0, 1], and at mu = 1 a
  pixel of exactly 1.0 is still inverted (``img < m``), so no mu is the identity. The op's
  range ends at 256/255, a threshold above every pixel and its exact identity (as
  torchvision's Solarize runs to 256); the search still maps mu to dda's [0, 1]
  (``POLICY_SCALES``), so a searched policy applies exactly dda's thresholds.
- **Kept as dda has them.** The gray weights are 0.299, 0.587, 0.110 (the blue weight
  should be 0.114). Contrast's mean is the 8-bit-rounded mean gray, as PIL's.
- **The warp.** dda used kornia 0.3.1, whose ``warp_affine`` normalized the matrix as if
  ``align_corners=True`` and then sampled with ``align_corners=False``; this uses
  kornia 0.8.2's ``warp_affine`` (``diff_ops._warp``), consistent at
  ``align_corners=True``. The matrices are dda's. The label moves with the geometric
  ops (nearest, out-of-frame -> 255, in the label's own frame), as in `geometric`.

Cutout and SamplePairing do not move the label (the user's choice for segmentation;
both papers are classification-only). SamplePairing couples the images of a batch
(``couples_images``), so the probe gives it an ``unprobed`` NaN row.

Which ops differentiate: the 18 split ops and SamplePairing through autograd.
Solarize, Posterize and Cutout step with the magnitude, so they are ``grad_mode="none"``:
the probe fades them in (configs/host/faster_autoaugment.yaml), and the policy search
uses dda's straight-through estimator for them (hosts/faster_autoaugment/policy.py).
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from kornia.geometry.transform import get_rotation_matrix2d

from ..diff_ops import _as_delta, _batch_delta, _broadcast_delta, _warp, warp_image_and_label
from ..op import OpSpec
from .madaug import cutout, flip

FAMILY = "faster_aa"
SOURCE = "Faster AutoAugment (Hataya et al., 2020), dda/functional.py"

Tensor = torch.Tensor

#: The paper's Table 2, in its order: the order the ops become axes of R, and the order
#: the search lays them out in.
OP_ORDER = (
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Rotate",
    "HorizontalFlip",
    "Solarize",
    "Posterize",
    "Invert",
    "Contrast",
    "Color",
    "Brightness",
    "Sharpness",
    "AutoContrast",
    "Equalize",
    "Cutout",
    "SamplePairing",
)

#: transform id -> dda's ``magnitude_scale``: the split op's mag_range upper end.
SPLIT_RANGES: Dict[str, float] = {
    "ShearX": 0.3,
    "ShearY": 0.3,
    "TranslateX": 0.45,
    "TranslateY": 0.45,
    "Rotate": 30.0,
    "Contrast": 1.0,
    "Color": 1.0,
    "Brightness": 1.0,
    "Sharpness": 1.0,
}

GEOMETRIC_IDS = frozenset({"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"})

DIRECTIONS = (("pos", 1.0), ("neg", -1.0))

#: The unsigned ops with a magnitude: ``(mag_range, identity_at)``.
UNSIGNED_RANGES: Dict[str, Tuple[Tuple[float, float], float]] = {
    "Solarize": ((0.0, 256.0 / 255.0), 256.0 / 255.0),
    "Posterize": ((0.0, 1.0), 1.0),
    "Cutout": ((0.0, 0.2), 0.0),
    "SamplePairing": ((0.0, 1.0), 0.0),
}

#: transform id -> dda's ``magnitude_scale``: the op-unit magnitude at mu = 1, the top of
#: what the search can apply. The mag_range's upper end, except Solarize's (its range
#: runs past dda's 1 to its identity).
POLICY_SCALES: Dict[str, float] = {
    **SPLIT_RANGES,
    "Solarize": 1.0,
    "Posterize": 1.0,
    "Cutout": 0.2,
    "SamplePairing": 1.0,
}

#: Ops whose output steps with the magnitude: no autograd derivative. The search uses
#: dda's straight-through estimator for these, and for AutoContrast and Equalize.
STEPPED_IDS = frozenset({"Solarize", "Posterize", "Cutout"})

#: dda's straight-through ops with no magnitude: the estimator passes the image's
#: gradient through unchanged.
STRAIGHT_THROUGH_IDS = frozenset({"AutoContrast", "Equalize"})

#: dda's ``_gray`` weights, the blue one as dda has it (0.114 elsewhere).
GRAY_WEIGHTS = (0.299, 0.587, 0.110)


# --- geometric: dda's kornia matrices -------------------------------------------------


def _matrix(images: Tensor, a, b, c, d, tx, ty) -> Tensor:
    """(B, 2, 3) ``[[a, b, tx], [c, d, ty]]`` from (B,) entries, graph kept."""
    return torch.stack([torch.stack([a, b, tx], dim=-1), torch.stack([c, d, ty], dim=-1)], dim=1)


def geometric_matrix(transform_id: str, images: Tensor, m: Tensor) -> Tensor:
    """The (B, 2, 3) forward matrix, in pixel-index coordinates (kornia's warp_affine
    frame), dda applies for a geometric op at signed magnitude ``m`` (B,) in its units:
    shear factor, translate fraction, degrees."""
    one = torch.ones_like(m)
    zero = torch.zeros_like(m)
    h, w = images.shape[-2:]
    if transform_id == "ShearX":
        return _matrix(images, one, m, zero, one, zero, zero)
    if transform_id == "ShearY":
        return _matrix(images, one, zero, m, one, zero, zero)
    if transform_id == "TranslateX":
        return _matrix(images, one, zero, zero, one, m * w, zero)
    if transform_id == "TranslateY":
        return _matrix(images, one, zero, zero, one, zero, m * h)
    if transform_id == "Rotate":
        # kornia.rotate: about ((W - 1) / 2, (H - 1) / 2), positive counter-clockwise.
        center = torch.stack([torch.full_like(m, (w - 1) / 2.0), torch.full_like(m, (h - 1) / 2.0)], dim=-1)
        scale = torch.ones(m.shape[0], 2, dtype=m.dtype, device=m.device)
        return get_rotation_matrix2d(center, m, scale)
    raise ValueError(f"{transform_id} is not a geometric op")


def geometric_op(transform_id: str, sign: float):
    def fn(images: Tensor, magnitude, label: Optional[Tensor] = None):
        matrix = geometric_matrix(transform_id, images, sign * _batch_delta(magnitude, images))
        if label is None:
            return _warp(images, matrix), None
        return warp_image_and_label(images, label, matrix)

    return fn


# --- photometric: dda's functional bodies ---------------------------------------------


def gray(images: Tensor) -> Tensor:
    """dda's ``_gray``: (B, 1, H, W)."""
    r, g, b = images.chunk(3, dim=1)
    return GRAY_WEIGHTS[0] * r + GRAY_WEIGHTS[1] * g + GRAY_WEIGHTS[2] * b


def blend(images: Tensor, degenerate: Tensor, factor: Tensor) -> Tensor:
    """dda's ``_blend_image``: ``degenerate + factor * (images - degenerate)``, clamped;
    factor 1 is the identity. ``factor`` is (n, 1, 1, 1)."""
    return (degenerate + factor * (images - degenerate)).clamp(0.0, 1.0)


def _sharpness_kernel(images: Tensor) -> Tensor:
    """dda's ``get_sharpness_kernel``: PIL's smooth filter, [[1, 1, 1], [1, 5, 1], [1, 1, 1]] / 13."""
    kernel = torch.ones(3, 3, dtype=images.dtype, device=images.device)
    kernel[1, 1] = 5.0
    return kernel / 13.0


def smooth(images: Tensor) -> Tensor:
    """dda's ``_blur`` with the sharpness kernel: reflect-padded 3x3, per channel."""
    c = images.shape[1]
    kernel = _sharpness_kernel(images).repeat(c, 1, 1, 1)
    return F.conv2d(F.pad(images, (1, 1, 1, 1), "reflect"), kernel, groups=c)


def mean_gray(images: Tensor) -> Tensor:
    """dda's ``contrast`` degenerate: the 8-bit-rounded mean gray, (B, 1, 1, 1)."""
    mean = gray(images * 255.0).flatten(1).mean(dim=1).add(0.5).floor()
    return mean.view(-1, 1, 1, 1) / 255.0


_DEGENERATE: Dict[str, Callable[[Tensor], Tensor]] = {
    "Contrast": mean_gray,
    "Color": gray,
    "Brightness": torch.zeros_like,
    "Sharpness": smooth,
}


def enhance_op(transform_id: str, sign: float):
    degenerate = _DEGENERATE[transform_id]

    def fn(images: Tensor, magnitude, label: Optional[Tensor] = None):
        factor = 1.0 + sign * _broadcast_delta(_as_delta(magnitude, images), images.shape[0])
        return blend(images, degenerate(images), factor), label

    return fn


def solarize(images: Tensor, magnitude, label: Optional[Tensor] = None):
    """dda's ``solarize``: pixels at or above the threshold ``m`` are inverted."""
    m = _broadcast_delta(_as_delta(magnitude, images), images.shape[0])
    return torch.where(images < m, images, 1.0 - images), label


def posterize(images: Tensor, magnitude, label: Optional[Tensor] = None):
    """dda's ``posterize`` with its shift order fixed: keep ``8 - shift`` bits,
    ``shift = floor(8 (1 - m))``."""
    m = _broadcast_delta(_as_delta(magnitude, images), images.shape[0])
    with torch.no_grad():
        shift = ((1.0 - m) * 8).long()
        levels = (images.mul(255).long() >> shift) << shift
    return levels.to(images.dtype) / 255.0, label


def invert(images: Tensor, magnitude=None, label: Optional[Tensor] = None):
    return 1.0 - images, label


def auto_contrast(images: Tensor, magnitude=None, label: Optional[Tensor] = None):
    """dda's ``auto_contrast``: each channel's [min, max] stretched to [0, 255]."""
    with torch.no_grad():
        reshaped = images.flatten(0, 1).flatten(1, 2).clamp(0, 1) * 255
        low, _ = reshaped.min(dim=1, keepdim=True)
        high, _ = reshaped.max(dim=1, keepdim=True)
        lut = (torch.arange(256, device=images.device, dtype=images.dtype) - low) * (255 / (high - low + 0.1))
        out = lut.floor().gather(1, reshaped.long()).reshape_as(images) / 255
    return out.clamp(0.0, 1.0), label


def equalize(images: Tensor, magnitude=None, label: Optional[Tensor] = None):
    """dda's ``equalize``: PIL's per-channel histogram equalization, batched."""
    with torch.no_grad():
        reshaped = images.clone().flatten(0, 1).clamp_(0, 1) * 255
        size = reshaped.size(0)
        shifted = reshaped + 256 * torch.arange(0, size, device=reshaped.device, dtype=reshaped.dtype).view(-1, 1, 1)
        histogram = shifted.histc(size * 256, 0, size * 256 - 1).view(size, 256)
        cdf = histogram.cumsum(-1)
        step = ((cdf[:, -1] - histogram[:, -1]) / 255).floor_().view(size, 1)
        cdf = torch.cat([cdf.new_zeros((cdf.size(0), 1)), cdf], dim=1)[:, :256] + (step / 2).floor_()
        out = (cdf / (step + 0.1)).floor_().view(-1)[shifted.long()].reshape_as(images) / 255
    return out.clamp(0.0, 1.0), label


def sample_pairing(images: Tensor, magnitude, label: Optional[Tensor] = None):
    """dda's ``sample_pairing``: each image blended with the image a random permutation
    of the batch puts in its place (possibly itself), ``(1 - m) x + m x[randperm]``. The
    label stays the image's own."""
    m = _broadcast_delta(_as_delta(magnitude, images), images.shape[0])
    partner = torch.randperm(images.shape[0]).to(images.device)
    return ((1.0 - m) * images + m * images[partner]).clamp(0.0, 1.0), label


#: The unsigned ops' implementations, (images, magnitude, label) -> (images, label).
UNSIGNED_FNS: Dict[str, Callable] = {
    "HorizontalFlip": flip,
    "Solarize": solarize,
    "Posterize": posterize,
    "Invert": invert,
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Cutout": cutout,
    "SamplePairing": sample_pairing,
}


def op_name(transform_id: str, direction: Optional[str] = None) -> str:
    return f"{FAMILY}.{transform_id}" if direction is None else f"{FAMILY}.{transform_id}_{direction}"


def _specs():
    for transform_id in OP_ORDER:
        if transform_id in SPLIT_RANGES:
            geometric = transform_id in GEOMETRIC_IDS
            for direction, sign in DIRECTIONS:
                yield OpSpec(
                    name=op_name(transform_id, direction),
                    family=FAMILY,
                    kind="geometric" if geometric else "photometric",
                    source=SOURCE,
                    mag_range=(0.0, SPLIT_RANGES[transform_id]),
                    identity_at=0.0,
                    grad_mode="autograd",
                    warps_label=geometric,
                    fn=geometric_op(transform_id, sign) if geometric else enhance_op(transform_id, sign),
                )
            continue
        mag_range, identity_at = UNSIGNED_RANGES.get(transform_id, (None, None))
        geometric = transform_id == "HorizontalFlip"
        yield OpSpec(
            name=op_name(transform_id),
            family=FAMILY,
            kind="geometric" if geometric else "photometric",
            source=SOURCE,
            mag_range=mag_range,
            identity_at=identity_at,
            grad_mode="autograd" if transform_id == "SamplePairing" else "none",
            warps_label=geometric,
            fn=UNSIGNED_FNS[transform_id],
            couples_images=transform_id == "SamplePairing",
        )


OPS = tuple(_specs())

#: vocab op name -> (transform id, sign), in OPS order: +1 / -1 for a split op, None for
#: an unsigned one.
TRANSFORMS: Dict[str, Tuple[str, Optional[float]]] = {}
for _t in OP_ORDER:
    if _t in SPLIT_RANGES:
        for _direction, _sign in DIRECTIONS:
            TRANSFORMS[op_name(_t, _direction)] = (_t, _sign)
    else:
        TRANSFORMS[op_name(_t)] = (_t, None)

assert tuple(TRANSFORMS) == tuple(s.name for s in OPS)
