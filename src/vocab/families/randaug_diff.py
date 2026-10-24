"""RandAugment's op list, differentiable, each signed op split into _pos/_neg.

The ops of `randaug_default` (Cubuk et al. 2020, Fig. 2; torchvision's RandAugment) at
AutoAugment's Table 6 ranges, built the way `trivialaug_diff` builds TrivialAugment's
(its `build_split_family`), so autograd differentiates the magnitude and one set of ops
is both trained and probed. The magnitude is unsigned and the direction is in the name:

    op               mag_range   units                          torchvision computes
    Rotate_pos/neg   [0, 30]     degrees, +/-                   rotate(angle=+/-m)
    ShearX/Y_pos/neg [0, 0.3]    shear rate, about the corner   affine(shear=deg(atan(+/-m)), center=[0, 0])
    TranslateX/Y_    [0, 150]    pixels at 331x331, scaled to   affine(translate=int(+/-m * w / 331))
      pos/neg                    the image as torchvision does
    Brightness_,
    Color_, Contrast_,
    Sharpness_pos/neg [0, 0.9]   factor 1 +/- m                 adjust_*(factor=1 +/- m)
    Posterize        [4, 8]      bits             } randaug_default's own op (torchvision's),
    Solarize         [0, 256]    threshold        } no derivative in the magnitude: the probe
    AutoContrast     none                         } fades them in (configs/host/
    Equalize         none                         } randaugment_diff.yaml)

Where these differ from torchvision is `trivialaug_diff`'s list: kornia's bilinear warp
instead of NEAREST, the translate not truncated to whole pixels, the enhance blend with
no uint8 round trip. Labels move with the geometric ops.

`BINS` gives the magnitudes RandAugment applies, unsigned. Up to 31 bins they are
torchvision RandAugment's own table (`_AUGMENTATION_SPACE`), whose default grid of 31
bins puts the paper's level M (0 to 10, 10 = Table 6's maximum) at bin 3M. The paper
also publishes levels above 10 (CIFAR-100 Wide-ResNet-28-10: M = 14; EfficientNet-B5 /
B7: 17 / 28), extending each range linearly past its end (its footnote 2: "We retain the
same scale as [5] for a value of 10"). torchvision can't express those, so a table of
more than 31 bins is the 31-bin grid continued: bin b >= 31 is level b / 3 on the same
linear scale, Posterize by torchvision's own formula (8 - round(b / 7.5) bits), Solarize's
threshold 255 (1 - b / 30). Each stops where the op stops meaning anything: a threshold
at 0 (every pixel inverted), 0 bits, and an enhance `_neg` at m = 1 (factor 0; a negative
factor is not an enhance and torchvision refuses it).
"""

from typing import Callable, Dict

from torchvision.transforms.v2 import RandAugment

from . import randaug_default
from .randaug_default import AA_IMAGE_SIZE, ENHANCE_IDS
from .trivialaug_diff import build_split_family

FAMILY = "randaug_diff"
SOURCE = "torchvision RandAugment (Cubuk et al., 2020), differentiable in torch"

#: transform id -> mag_range upper end (unsigned, identity at 0), for the split ops:
#: AutoAugment's Table 6, as randaug_default states it.
SPLIT_RANGES: Dict[str, float] = {
    "Rotate": 30.0,
    "Color": 0.9,
    "Contrast": 0.9,
    "Brightness": 0.9,
    "Sharpness": 0.9,
    "ShearX": 0.3,
    "ShearY": 0.3,
    "TranslateX": 150.0,
    "TranslateY": 150.0,
}

#: The ops kept as torchvision's own (`randaug_default`'s implementation).
EXACT_IDS = ("AutoContrast", "Equalize", "Solarize", "Posterize")

#: torchvision RandAugment's default number of bins. Bin b is the paper's level b / 3.
GRID_BINS = 31
BINS_PER_LEVEL = 3


def _beyond(name: str) -> Callable[[int], float]:
    """Bin b >= GRID_BINS of op ``name``: the paper's linear scale past level 10."""
    base = name.split(".", 1)[1]
    last = GRID_BINS - 1
    if base == "Posterize":
        return lambda b: float(max(8 - round(b / (last / 4)), 0))
    if base == "Solarize":
        return lambda b: max(255.0 * (1.0 - b / last), 0.0)
    transform_id, direction = base.rsplit("_", 1)
    hi = SPLIT_RANGES[transform_id]
    cap = 1.0 if transform_id in ENHANCE_IDS and direction == "neg" else float("inf")
    return lambda b: min(hi * b / last, cap)


def _on_grid(name: str, table):
    """``table`` (torchvision's) up to GRID_BINS bins, the grid continued past it."""
    beyond = _beyond(name)

    def grid(num_bins):
        if num_bins <= GRID_BINS:
            return table(num_bins)
        values, signed = table(GRID_BINS)
        return values + tuple(beyond(b) for b in range(GRID_BINS, num_bins)), signed

    return grid


def _exact(transform_id: str):
    name = f"{randaug_default.FAMILY}.{transform_id}"
    spec = next(s for s in randaug_default.OPS if s.name == name)
    return spec, randaug_default.BINS.get(name)


OPS, _TORCHVISION_BINS = build_split_family(
    FAMILY,
    SOURCE,
    RandAugment,
    SPLIT_RANGES,
    {t: _exact(t) for t in EXACT_IDS},
    translate_ref=AA_IMAGE_SIZE,
)
BINS = {name: _on_grid(name, table) for name, table in _TORCHVISION_BINS.items()}
