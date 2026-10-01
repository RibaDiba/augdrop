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
the paper's 9/10.)

Implementation: each op calls torchvision.transforms.v2's own per-op dispatcher
(`_apply_image_or_video_transform`), the code path RandAugment itself runs,
with torchvision's defaults (NEAREST interpolation, fill 0). Like the originals
it runs on 8-bit images: the [0, 1] float input is rounded to uint8, augmented,
and returned as float in [0, 1], so it is exact on 8-bit-representable input and
carries no gradient (grad_mode "none"). Geometric ops replay the same transform
on the label, nearest-neighbour, out-of-frame -> 255, as Sensaug's
label_safe_transforms.py does.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import RandAugment

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
        pixels = (images.detach() * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)

        out_images, out_labels = [], []
        for i in range(b):
            native = 0.0 if to_native is None else to_native(mags[i], h, w)
            image = pixels[i]
            if not (transform_id == "Solarize" and mags[i] > _SOLARIZE_MAX_THRESHOLD):
                image = method._apply_image_or_video_transform(
                    image, transform_id, native, method.interpolation, method._fill
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
            )
        )
    return tuple(specs)


OPS = build_ops(FAMILY, RandAugment(), RANGES, AA_IMAGE_SIZE, SOURCE)
