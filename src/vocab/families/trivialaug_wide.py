"""TrivialAugment's wide op list, applied by torchvision's own implementation.

Müller & Hutter, "TrivialAugment: Tuning-free Yet State-of-the-Art Data
Augmentation" (ICCV 2021), Table 8: "The Wide augmentation space ... has the
same operations as RA, but uses the strength ranges in parentheses." So the op
list is RandAugment's 14 (see `randaug_default`), and the ranges are:

    ShearX/Y        [-0.99, 0.99]  shear rate
    TranslateX/Y    [-32, 32]      pixels, absolute (not scaled to the image)
    Rotate          [-135, 135]    degrees
    Solarize        [0, 256]       threshold; pixels >= threshold are inverted
    Posterize       [2, 8]         bits kept per channel
    Color, Contrast,
    Brightness,
    Sharpness       [0.01, 2.0]    PIL ImageEnhance factor, 1.0 = unchanged
    Identity,
    AutoContrast,
    Equalize        no magnitude

Table 8 lists shear and translate as unsigned strengths (0 - 0.99, 0 - 32); the
method draws their sign at random, as it does for rotate, so the op's own
argument range is signed. The enhance range is the paper's (0.01 - 2.);
torchvision's bins stop at 1.99 (1 + 0.99), which the op still accepts.

Implementation, units conversion and label handling are shared with
`randaug_default`; each op calls TrivialAugmentWide's own dispatcher.
"""

from typing import Dict, Optional, Tuple

from torchvision.transforms.v2 import TrivialAugmentWide

from .randaug_default import build_ops

FAMILY = "trivialaug_wide"
SOURCE = "torchvision.transforms.v2.TrivialAugmentWide (Müller & Hutter, 2021)"

RANGES: Dict[str, Optional[Tuple[Tuple[float, float], float]]] = {
    "Identity": None,
    "AutoContrast": None,
    "Equalize": None,
    "Rotate": ((-135.0, 135.0), 0.0),
    "Solarize": ((0.0, 256.0), 256.0),
    "Color": ((0.01, 2.0), 1.0),
    "Posterize": ((2.0, 8.0), 8.0),
    "Contrast": ((0.01, 2.0), 1.0),
    "Brightness": ((0.01, 2.0), 1.0),
    "Sharpness": ((0.01, 2.0), 1.0),
    "ShearX": ((-0.99, 0.99), 0.0),
    "ShearY": ((-0.99, 0.99), 0.0),
    "TranslateX": ((-32.0, 32.0), 0.0),
    "TranslateY": ((-32.0, 32.0), 0.0),
}

OPS = build_ops(FAMILY, TrivialAugmentWide(), RANGES, None, SOURCE)
