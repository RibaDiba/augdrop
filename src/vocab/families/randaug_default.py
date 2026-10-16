"""RandAugment's default op list, applied by torchvision's own implementation.

Op list: the 14 transformations of Cubuk et al., "RandAugment: Practical
automated data augmentation with a reduced search space" (2020), Fig. 2, in that
order. torchvision.transforms.v2.RandAugment implements the same 14.

Units: the RandAugment paper puts every op on AutoAugment's shared linear
magnitude scale ("Following [5] ... an integer scale from 0 to 10 where a value
of 10 indicates the maximum scale"), so each op's range is the one in
AutoAugment's Table 6 (Cubuk et al., 2019), stated there for 331x331 images:

    ShearX/Y        [-0.3, 0.3]   shear rate
    TranslateX/Y    [-150, 150]   pixels, at 331x331
    Rotate          [-30, 30]     degrees
    Solarize        [0, 256]      threshold; pixels >= threshold are inverted
    Posterize       [4, 8]        bits kept per channel
    Color, Contrast,
    Brightness,
    Sharpness       [0.1, 1.9]    PIL ImageEnhance factor, 1.0 = unchanged
    Identity,
    AutoContrast,
    Equalize        no magnitude

`mag_range` and the op's magnitude argument are in exactly those units. The
conversion to what torchvision's dispatcher takes (factor - 1 for the enhance
ops, the translate scaled to the actual image as 150/331 of its extent,
threshold as a fraction of 255) happens inside the op and is pinned against
torchvision's public functionals in tests/vocab/test_families.py.

Only the ops are here, not the method: sampling N ops at a shared magnitude M
is RandAugment the policy, and belongs with the other methods. (Note for later:
torchvision's default magnitude=9 indexes 31 bins, i.e. 9/30 of each range, not
the paper's 9/10.) What the ops do know is the bins a method draws their magnitude
from: `BINS` reads them off torchvision's own table (`_AUGMENTATION_SPACE`) in these
units, so a host can say "bin 9 of 31" and the probe can apply exactly that.

Implementation: each op calls torchvision.transforms.v2's own per-op dispatcher
(`_apply_image_or_video_transform`), the code path RandAugment itself runs,
with torchvision's defaults (NEAREST interpolation, fill 0). Like the originals
it runs on 8-bit images: the [0, 1] float input is rounded to uint8, augmented,
and returned as float in [0, 1], so it is exact on 8-bit-representable input and
carries no gradient (grad_mode "none"). Geometric ops replay the same transform
on the label, nearest-neighbour, out-of-frame -> 255, as Sensaug's
label_safe_transforms.py does.

The four enhance ops also carry an exact `tangent` (`enhance_tangent`), which the
probe differentiates through instead of taking finite differences. The images are
still torchvision's; only the derivative's direction is computed here.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import RandAugment
from torchvision.transforms.v2.functional import rgb_to_grayscale

from ..diff_ops import LABEL_IGNORE_INDEX
from ..op import OpSpec

FAMILY = "randaug_default"
SOURCE = "torchvision.transforms.v2.RandAugment (Cubuk et al., 2020)"

GEOMETRIC_TRANSFORM_IDS = frozenset({"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"})

# AutoAugment's Table 6 ranges are stated for 331x331 images; torchvision scales
# the translate range to 150/331 of the actual extent.
AA_IMAGE_SIZE = 331.0

# The RandAugment paper's Fig. 2 order.
OP_ORDER = (
    "Identity", "AutoContrast", "Equalize", "Rotate", "Solarize", "Color", "Posterize",
    "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateX", "TranslateY",
)

ENHANCE_IDS = frozenset({"Color", "Contrast", "Brightness", "Sharpness"})

# The largest threshold torchvision's solarize accepts on uint8. The paper's
# range ends at 256, where no 8-bit pixel is >= the threshold and the op is the
# identity (as PIL's ImageOps.solarize(img, 256) is); torchvision rejects it, so
# the op returns the image itself above this.
_SOLARIZE_MAX_THRESHOLD = 255.0

# transform id -> (mag_range, identity_at), in the paper's units.
RANGES: Dict[str, Optional[Tuple[Tuple[float, float], float]]] = {
    "Identity": None,
    "AutoContrast": None,
    "Equalize": None,
    "Rotate": ((-30.0, 30.0), 0.0),
    "Solarize": ((0.0, 256.0), 256.0),
    "Color": ((0.1, 1.9), 1.0),
    "Posterize": ((4.0, 8.0), 8.0),
    "Contrast": ((0.1, 1.9), 1.0),
    "Brightness": ((0.1, 1.9), 1.0),
    "Sharpness": ((0.1, 1.9), 1.0),
    "ShearX": ((-0.3, 0.3), 0.0),
    "ShearY": ((-0.3, 0.3), 0.0),
    "TranslateX": ((-150.0, 150.0), 0.0),
    "TranslateY": ((-150.0, 150.0), 0.0),
}

ToNative = Callable[[float, int, int], float]

#: num_bins -> (each bin's magnitude in the op's units, signed). A signed op is negated
#: with p = 0.5 by the method, i.e. mirrored about identity_at.
BinTable = Callable[[int], Tuple[Tuple[float, ...], bool]]


def native_magnitude(transform_id: str, translate_ref: Optional[float]) -> ToNative:
    """Paper units -> the magnitude torchvision's dispatcher takes, as a function
    of (magnitude, image height, image width).

    `translate_ref` is the image size the paper's translate range is stated for
    (331 for AutoAugment/RandAugment), or None when the range is absolute pixels
    (TrivialAugment Wide)."""
    if transform_id in ENHANCE_IDS:
        # torchvision applies factor = 1.0 + magnitude.
        return lambda m, h, w: m - 1.0
    if transform_id == "Solarize":
        # torchvision applies threshold = 255 * magnitude on uint8.
        return lambda m, h, w: m / 255.0
    if transform_id == "TranslateX" and translate_ref is not None:
        return lambda m, h, w: m * w / translate_ref
    if transform_id == "TranslateY" and translate_ref is not None:
        return lambda m, h, w: m * h / translate_ref
    # Rotate (degrees), ShearX/Y (rate), Posterize (bits), absolute translates.
    return lambda m, h, w: m


def paper_magnitude(transform_id: str) -> Callable[[float], float]:
    """The inverse of `native_magnitude` on an image of the reference size: the
    magnitude torchvision's dispatcher takes -> the paper's units."""
    if transform_id in ENHANCE_IDS:
        return lambda v: v + 1.0
    if transform_id == "Solarize":
        return lambda v: v * 255.0
    # Rotate, ShearX/Y, Posterize; translates at the reference size, or absolute.
    return lambda v: v


def build_bins(
    family: str,
    method_cls,
    ranges: Dict[str, Optional[Tuple[Tuple[float, float], float]]],
    translate_ref: Optional[float],
    op_order: Sequence[str] = OP_ORDER,
) -> Dict[str, BinTable]:
    """One `BinTable` per op with a magnitude: the bins torchvision's `method_cls`
    draws that op's magnitude from, read off its `_AUGMENTATION_SPACE` (not re-derived)
    and put in the op's units.

    The table is evaluated on a reference-size image, where torchvision's translate
    in pixels equals the paper's (TrivialAugment's translate is absolute and ignores
    the size). torchvision builds the bins in float32, so an end bin can overshoot the
    range by round-off (0.3 is 0.30000001); that is clipped, and anything larger
    raises."""
    size = int(AA_IMAGE_SIZE if translate_ref is None else translate_ref)
    space = method_cls._AUGMENTATION_SPACE
    tables = {}
    for transform_id in op_order:
        if ranges[transform_id] is None:
            continue
        (lo, hi), _ = ranges[transform_id]
        magnitudes_fn, signed = space[transform_id]
        to_paper = paper_magnitude(transform_id)
        name = f"{family}.{transform_id}"

        def table(num_bins, magnitudes_fn=magnitudes_fn, signed=signed, to_paper=to_paper, lo=lo, hi=hi, name=name):
            values = [to_paper(float(v)) for v in magnitudes_fn(num_bins, size, size)]
            slack = 1e-6 * (hi - lo)
            outside = [v for v in values if not lo - slack <= v <= hi + slack]
            if outside:
                raise ValueError(f"{name}: torchvision's bins {outside} are outside mag_range [{lo}, {hi}]")
            return tuple(min(max(v, lo), hi) for v in values), bool(signed)

        tables[name] = table
    return tables


def _per_image(magnitude, batch_size: int, dtype: torch.dtype) -> Sequence[float]:
    """One python float per image, taken at the images' precision, as the
    differentiable ops take theirs (`diff_ops._as_delta`). Otherwise a python
    float and a float32 tensor holding it could land either side of a rounding
    boundary in the 8-bit output, and a scalar magnitude would not equal the same
    value given per image."""
    if torch.is_tensor(magnitude):
        m = magnitude.detach().to("cpu", dtype).reshape(-1)
    else:
        m = torch.as_tensor(magnitude, dtype=dtype).reshape(-1)
    if m.numel() == 1:
        m = m.expand(batch_size)
    if m.numel() != batch_size:
        raise ValueError(
            f"magnitude must be a scalar or a length-{batch_size} vector (one per "
            f"image), got {m.numel()} values"
        )
    return [float(v) for v in m]


def _fill_of(method):
    """The fill dict torchvision's dispatcher takes: `method._fill` in recent
    torchvision (0.27), `method.fill` in 0.15 (the sensaug env on Nexus). Same `{type: fill}`
    shape in both."""
    try:
        return method._fill
    except AttributeError:
        return method.fill


def _to_pixels(images: torch.Tensor) -> torch.Tensor:
    """[0, 1] float images -> the uint8 images torchvision's ops run on."""
    return (images.detach() * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)


def enhance_tangent(method, transform_id: str):
    """The exact d op / d m of an enhance op, as the vocab's (images, magnitude) ->
    (B, C, H, W) tangent, in [0, 1] image units per unit of the paper's factor.

    torchvision computes each enhance op as a blend, clamp(f * x + (1 - f) * d) cast
    to uint8, with the factor f = m and a `d` that doesn't depend on f. Before the
    clamp and the cast the derivative is x - d, and where the clamp holds it is 0.
    `d` is torchvision's own: the op at factor 0 (black, the floored grayscale, the
    3x3-blurred image with its border kept), except for Contrast, whose factor-0
    output truncates the grayscale mean to uint8; there `d` is the float mean of
    `rgb_to_grayscale`, which is what adjust_contrast blends toward. Checked by hand
    on all four ops at factors 0.01 to 2.0: clamp(f * x + (1 - f) * d), cast to
    uint8, equals torchvision's output pixel for pixel.
    """
    if transform_id not in ENHANCE_IDS:
        raise ValueError(f"{transform_id} is not an enhance op {sorted(ENHANCE_IDS)}")

    def tangent(images, magnitude):
        mags = _per_image(magnitude, images.shape[0], images.dtype)
        pixels = _to_pixels(images)
        out = []
        for i, factor in enumerate(mags):
            x = pixels[i]
            if transform_id == "Contrast":
                d = rgb_to_grayscale(x).float().mean(dim=(-3, -2, -1), keepdim=True)
            else:
                # torchvision applies factor = 1.0 + magnitude, so factor 0 is -1.
                d = method._apply_image_or_video_transform(
                    x, transform_id, -1.0, method.interpolation, _fill_of(method)
                ).float()
            x = x.float()
            blended = x.mul(factor).add_(d, alpha=1.0 - factor)
            inside = (blended >= 0.0) & (blended <= 255.0)
            out.append(torch.where(inside, x - d, 0.0) / 255.0)
        return torch.stack(out).to(images.dtype)

    return tangent


def torchvision_op(method, transform_id: str, to_native: Optional[ToNative]):
    """Wrap one op of a torchvision v2 auto-augment `method` instance as the
    vocab's (images, magnitude, label) -> (images, label) call.

    Images are processed one at a time with that image's own magnitude, exactly
    as the method would see them, so no image can depend on another."""
    geometric = transform_id in GEOMETRIC_TRANSFORM_IDS

    def fn(images, magnitude, label=None):
        b, _, h, w = images.shape
        if geometric and label is not None and tuple(label.shape[-2:]) != (h, w):
            # Translate magnitudes are drawn in the image's pixels, so a label at
            # another resolution would be shifted by the wrong amount.
            raise ValueError(
                f"{transform_id}: label {tuple(label.shape[-2:])} and image {(h, w)} "
                "differ in size; the warp would not line up"
            )
        mags = _per_image(magnitude, b, images.dtype) if to_native is not None else [None] * b
        pixels = _to_pixels(images)

        out_images, out_labels = [], []
        for i in range(b):
            native = 0.0 if to_native is None else to_native(mags[i], h, w)
            image = pixels[i]
            if not (transform_id == "Solarize" and mags[i] > _SOLARIZE_MAX_THRESHOLD):
                image = method._apply_image_or_video_transform(
                    image, transform_id, native, method.interpolation, _fill_of(method)
                )
            out_images.append(image)

            if geometric and label is not None:
                seg = label[i] if label.dim() == 4 else label[i].unsqueeze(0)
                seg = method._apply_image_or_video_transform(
                    seg,
                    transform_id,
                    native,
                    InterpolationMode.NEAREST,
                    {type(seg): LABEL_IGNORE_INDEX},
                )
                out_labels.append(seg if label.dim() == 4 else seg.squeeze(0))

        images_out = torch.stack(out_images).to(images.dtype) / 255.0
        label_out = torch.stack(out_labels) if out_labels else label
        return images_out, label_out

    return fn


def build_ops(
    family: str,
    method,
    ranges: Dict[str, Optional[Tuple[Tuple[float, float], float]]],
    translate_ref: Optional[float],
    source: str,
    op_order: Sequence[str] = OP_ORDER,
) -> Tuple[OpSpec, ...]:
    """One OpSpec per op of a torchvision auto-augment method, in `op_order`."""
    specs = []
    for transform_id in op_order:
        entry = ranges[transform_id]
        mag_range, identity_at = entry if entry is not None else (None, None)
        to_native = None if entry is None else native_magnitude(transform_id, translate_ref)
        geometric = transform_id in GEOMETRIC_TRANSFORM_IDS
        specs.append(
            OpSpec(
                name=f"{family}.{transform_id}",
                family=family,
                kind="geometric" if geometric else "photometric",
                source=source,
                mag_range=mag_range,
                identity_at=identity_at,
                grad_mode="none",
                warps_label=geometric,
                fn=torchvision_op(method, transform_id, to_native),
                tangent=enhance_tangent(method, transform_id) if transform_id in ENHANCE_IDS else None,
            )
        )
    return tuple(specs)


OPS = build_ops(FAMILY, RandAugment(), RANGES, AA_IMAGE_SIZE, SOURCE)
BINS = build_bins(FAMILY, RandAugment, RANGES, AA_IMAGE_SIZE)
