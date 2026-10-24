"""AugMix's op list as Sensaug trained it, differentiable, each signed op split into
_pos/_neg.

Sensaug's AugMix arm (`AugMixTransform` in laurayuzheng/SensAug's
sensaug/dataset/augmentations.py) is torchvision's `AugMix` with `all_ops=True`: the 13
ops of its `_AUGMENTATION_SPACE`, which are RandAugment's 13 (no Identity), at AugMix's
own ranges. (The AugMix paper's default leaves out Color, Contrast, Brightness and
Sharpness; Sensaug's runs did not.) They are built the way `trivialaug_diff` builds
TrivialAugment's (its `build_split_family`), so autograd differentiates the magnitude:

    op               mag_range   units                          torchvision computes
    Rotate_pos/neg   [0, 30]     degrees, +/-                   rotate(angle=+/-m)
    ShearX/Y_pos/neg [0, 0.3]    shear rate, about the corner   affine(shear=deg(atan(+/-m)), center=[0, 0])
    TranslateX/Y_    [0, 1/3]    fraction of the image width /  affine(translate=int(+/-m * w))
      pos/neg                    height (torchvision: w / 3 px at the last bin)
    Brightness_,
    Color_, Contrast_,
    Sharpness_pos/neg [0, 0.9]   factor 1 +/- m                 adjust_*(factor=1 +/- m)
    Posterize        [0, 8]      bits             } torchvision's own op, through AugMix's
    Solarize         [0, 256]    threshold        } dispatcher, no derivative in the
    AutoContrast     none                         } magnitude: the probe fades them in
    Equalize         none                         } (configs/host/augmix_diff.yaml)

AugMix's bins (`BINS`) are its table at its `_PARAMETER_MAX` = 10 bins, unsigned; it
draws a magnitude from the first `severity` of them (hosts/augmix.py). Posterize's bins
are 4 - round(b / 2.25) bits, so its range runs down to 0 bits, below RandAugment's 4.

Where these differ from torchvision is `trivialaug_diff`'s list: kornia's bilinear warp
(Sensaug's AugMix used BILINEAR too) with the translate not truncated to whole pixels,
and the enhance blend with no uint8 round trip. The ops move a label like every
geometric vocab op (the probe needs that). Training does not hand them one: Sensaug's
AugMix augments the image only and leaves the mask as it is (tasks/augment.py).
"""

from typing import Dict, Optional, Tuple

from torchvision.transforms.v2 import AugMix

from .randaug_default import build_bins, build_ops
from .trivialaug_diff import build_split_family

FAMILY = "augmix_diff"
SOURCE = "torchvision AugMix (Hendrycks et al., 2020), all_ops as Sensaug ran it, differentiable in torch"

#: transform id -> mag_range upper end (unsigned, identity at 0), for the split ops:
#: the last of torchvision AugMix's bins.
SPLIT_RANGES: Dict[str, float] = {
    "Rotate": 30.0,
    "Color": 0.9,
    "Contrast": 0.9,
    "Brightness": 0.9,
    "Sharpness": 0.9,
    "ShearX": 0.3,
    "ShearY": 0.3,
    "TranslateX": 1.0 / 3.0,
    "TranslateY": 1.0 / 3.0,
}

#: The ops kept as torchvision's own, through AugMix's dispatcher, as
#: transform id -> (mag_range, identity_at) in randaug_default's units.
EXACT_RANGES: Dict[str, Optional[Tuple[Tuple[float, float], float]]] = {
    "AutoContrast": None,
    "Equalize": None,
    "Solarize": ((0.0, 256.0), 256.0),
    "Posterize": ((0.0, 8.0), 8.0),
}
EXACT_IDS = tuple(EXACT_RANGES)

#: A translate's magnitude is a fraction of the image's extent: pixels "at a 1-pixel image".
TRANSLATE_REF = 1.0

_EXACT_OPS = build_ops(FAMILY, AugMix(), EXACT_RANGES, None, SOURCE, op_order=EXACT_IDS)
_EXACT_BINS = build_bins(FAMILY, AugMix, EXACT_RANGES, None, op_order=EXACT_IDS)

OPS, BINS = build_split_family(
    FAMILY,
    SOURCE,
    AugMix,
    SPLIT_RANGES,
    {spec.name.split(".", 1)[1]: (spec, _EXACT_BINS.get(spec.name)) for spec in _EXACT_OPS},
    translate_ref=TRANSLATE_REF,
)
