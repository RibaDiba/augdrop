"""SmartAugment's op list, differentiable, each signed op split into _pos/_neg.

Negassi, Wagner & Reiterer, "Smart(Sampling)Augment: Optimal and Efficient Data
Augmentation for Semantic Segmentation" (arXiv 2111.00487; Algorithms 15(5):165, 2022),
Table 2: "the same augmentations as in RandAugment paper", split into a color and a
geometric list (SmartAugment samples N_C of the first and N_G of the second), at these
ranges:

    Color ops      Range          Geometric ops   Range
    Sharpness      (0.1, 1.9)     Rotate          (0, 30)
    AutoContrast   (0, 1)         ShearX          (0.0, 0.3)
    Equalize       (0, 1)         ShearY          (0.0, 0.3)
    Solarize       (0, 256)       TranslateX      (0.0, 0.33)
    Color          (0.1, 1.9)     TranslateY      (0.0, 0.33)
    Contrast       (0.1, 1.9)
    Brightness     (0.1, 1.9)

The ops are built the way `trivialaug_diff` builds TrivialAugment's (its `split_specs`),
so autograd differentiates the magnitude and one set of ops is both trained and probed.
Units are `augmix_diff`'s, whose ranges these are (AugMix's code also stops translate at
a third of the image):

    op               mag_range   units                          applies
    Sharpness_,
    Color_, Contrast_,
    Brightness_pos/neg [0, 0.9]  factor 1 +/- m                 the float blend at 1 +/- m
    Rotate_pos/neg   [0, 30]     degrees, +/-                   rotate(angle=+/-m)
    ShearX/Y_pos/neg [0, 0.3]    shear rate, about the corner   affine(shear=deg(atan(+/-m)), center=[0, 0])
    TranslateX/Y_    [0, 0.33]   fraction of the image width /  affine(translate=+/-m * w)
      pos/neg                    height
    Solarize         [0, 256]    threshold        } randaug_default's own op (torchvision's),
    AutoContrast     none                         } no derivative in the magnitude: the probe
    Equalize         none                         } fades them in (configs/host/smartsampling_diff.yaml)

Where these differ from torchvision is `trivialaug_diff`'s list: kornia's bilinear warp,
the translate not truncated to whole pixels, the enhance blend with no uint8 round trip.
Labels move with the geometric ops.

**The levels (`BINS`) are a reconstruction.** No code was released
(github.com/mvg-inatech/SmartAugment holds only a README), and the paper gives the ranges
but not the map from its level to them. Its level is on a 0 to 30 scale ("magnitude in
the range of [0, 30]", Sec. 4.1.2; SmartSamplingAugment draws it from [5, 30]). Table 2's
(lo, hi) pairs are those of the original AutoAugment / RandAugment / AugMix code, which all
apply level L of L_max as ``lo + (hi - lo) * L / L_max`` and negate a geometric op with
p = 0.5. So bin b (level b of 30, at 31 bins) is:

    geometric _pos / _neg   hi * b / 30 (the method's random sign is the _pos/_neg choice)
    enhance                 factor 0.1 + 1.8 * b / 30: below level 15 it is _neg at
                            m = 1 - factor, from level 15 _pos at m = factor - 1, and level
                            15 is the identity (_pos at m = 0). Not torchvision's 1 +/- m
                            with a random sign: the direction follows the level
    Solarize                threshold 256 * b / 30, as RandAugment's own code computes it
                            (weaker as the level grows). AutoAugment's and AugMix's code use
                            256 - 256 * b / 30. Not a SmartSamplingAugment op; flagged for a
                            SmartAugment host

A table of another size is the same scale on ``num_bins`` evenly spaced levels.
"""

import dataclasses
from typing import Callable, Dict, Tuple

from ..op import OpSpec
from . import randaug_default
from .trivialaug_diff import GEOMETRIC_IDS, split_specs

FAMILY = "smartaug_diff"
SOURCE = "SmartAugment (Negassi et al., 2022), Table 2, differentiable in torch"

#: Table 2's two lists, in its order.
COLOR_IDS = ("Sharpness", "AutoContrast", "Equalize", "Solarize", "Color", "Contrast", "Brightness")
GEOMETRIC_ORDER = ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY")
OP_ORDER = COLOR_IDS + GEOMETRIC_ORDER

#: transform id -> mag_range upper end (unsigned, identity at 0), for the split ops:
#: Table 2's ranges, the enhance factor's (0.1, 1.9) as 1 +/- 0.9.
SPLIT_RANGES: Dict[str, float] = {
    "Sharpness": 0.9,
    "Color": 0.9,
    "Contrast": 0.9,
    "Brightness": 0.9,
    "Rotate": 30.0,
    "ShearX": 0.3,
    "ShearY": 0.3,
    "TranslateX": 0.33,
    "TranslateY": 0.33,
}

#: The ops kept as torchvision's own (`randaug_default`'s implementation).
EXACT_IDS = ("AutoContrast", "Equalize", "Solarize")

#: A translate's magnitude is a fraction of the image's extent, as in `augmix_diff`.
TRANSLATE_REF = 1.0

#: The top of the paper's level scale: level 30 is the end of each Table 2 range.
PARAMETER_MAX = 30

#: The level at which an enhance op's factor is 1.0 (0.1 + 1.8 * 15 / 30).
IDENTITY_LEVEL = 15

#: Table 2's enhance factor range.
ENHANCE_RANGE = (0.1, 1.9)

#: Table 2's Solarize range (the threshold).
SOLARIZE_RANGE = (0.0, 256.0)


def _levels(num_bins: int) -> Tuple[float, ...]:
    """The levels of a ``num_bins``-bin table, evenly spaced on [0, PARAMETER_MAX]."""
    if num_bins < 2:
        raise ValueError(f"a level table needs at least 2 bins, got {num_bins}")
    return tuple(PARAMETER_MAX * b / (num_bins - 1) for b in range(num_bins))


def _split_table(transform_id: str, direction: str) -> Callable[[int], Tuple[Tuple[float, ...], bool]]:
    """Bin b of a split op: its unsigned magnitude at level b (module docstring)."""
    hi = SPLIT_RANGES[transform_id]
    if transform_id in GEOMETRIC_IDS:

        def magnitude(level):
            return hi * level / PARAMETER_MAX

    else:
        lo_f, hi_f = ENHANCE_RANGE
        sign = 1.0 if direction == "pos" else -1.0

        def magnitude(level):
            factor = lo_f + (hi_f - lo_f) * level / PARAMETER_MAX
            return sign * (factor - 1.0)

    def table(num_bins):
        return tuple(min(max(magnitude(level), 0.0), hi) for level in _levels(num_bins)), False

    return table


def _solarize_table(num_bins):
    lo, hi = SOLARIZE_RANGE
    return tuple(lo + (hi - lo) * level / PARAMETER_MAX for level in _levels(num_bins)), False


def _build() -> Tuple[Tuple[OpSpec, ...], Dict[str, Callable]]:
    specs, tables = [], {}
    for transform_id in OP_ORDER:
        if transform_id in EXACT_IDS:
            name = f"{randaug_default.FAMILY}.{transform_id}"
            spec = next(s for s in randaug_default.OPS if s.name == name)
            spec = dataclasses.replace(spec, name=f"{FAMILY}.{transform_id}", family=FAMILY)
            specs.append(spec)
            if transform_id == "Solarize":
                tables[spec.name] = _solarize_table
            continue
        for spec in split_specs(FAMILY, SOURCE, transform_id, SPLIT_RANGES[transform_id], TRANSLATE_REF):
            specs.append(spec)
            tables[spec.name] = _split_table(transform_id, spec.name.rsplit("_", 1)[1])
    return tuple(specs), tables


assert set(SPLIT_RANGES) | set(EXACT_IDS) == set(OP_ORDER) and not set(SPLIT_RANGES) & set(EXACT_IDS)
assert set(GEOMETRIC_ORDER) == set(GEOMETRIC_IDS)

OPS, BINS = _build()
