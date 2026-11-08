"""AdaAug's ops, differentiable, each signed op split into _pos/_neg.

Cheung & Yeung, "AdaAug: Learning Class- and Instance-adaptive Data Augmentation
Policies" (ICLR 2022), and its code (github.com/jamestszhim/adaptive_augment, MIT):
``ada_aug/operation.py`` ("adapted from Fast AutoAugment and AutoAugment"), whose
``AUGMENT_LIST`` is 16 ops and Identity (the host's no-op, as everywhere in dropaug),
each applied at ``level * (high - low) + low`` for the policy's level in [0, 1]:

    op                  AUGMENT_LIST    mag_range   units                     PIL call (8-bit)
    ShearX/Y_pos/neg    [-0.3, 0.3]     [0, 0.3]    shear factor v            AFFINE (1, v, 0, 0, 1, 0)
    TranslateX/Y_       [-0.45, 0.45]   [0, 0.45]   fraction of W / H         AFFINE (1, 0, v W, 0, 1, 0)
      pos/neg
    Rotate_pos/neg      [-30, 30]       [0, 30]     degrees                   img.rotate(v)
    AutoContrast        none                                                  ImageOps.autocontrast
    Invert              none                                                  ImageOps.invert
    Equalize            none                                                  ImageOps.equalize
    Solarize            [0, 256]        [0, 256]    threshold; identity 256   ImageOps.solarize(img, v)
    Posterize           [4, 8]          [4, 8]      bits, int(v); identity 8  ImageOps.posterize(img, int(v))
    Contrast_, Color_,
    Brightness_,        [0.1, 1.9]      [0, 0.9]    factor 1 +/- m            ImageEnhance.<op>(img).enhance(v)
    Sharpness_pos/neg
    Cutout              [0, 0.2]        [0, 0.2]    square side, fraction of W  CutoutAbs(img, v W), (125, 123, 114)
    Flip                none                                                  ImageOps.mirror

These are Fast AutoAugment's ops at Fast AutoAugment's ranges, which is also the lineage
and the ranges of DeepAA's ``augmentation.py``, so the bodies are ``deepaa``'s: the
geometric ops are PIL's AFFINE maps sampled bilinearly (the label moves), the enhance
ops PIL's ImageEnhance blend in float, and Solarize, Posterize, AutoContrast, Invert and
Equalize the torchvision ops that equal PIL's (``deepaa`` docstring). Cutout and Flip are
``madaug``'s: MADAug's ``operation.py`` is AdaAug's, and ``madaug.cutout`` is its
``CutoutAbs`` exactly (side ``v * width`` for both sides, the centre ``np.random.uniform(w)``
as numpy reads it); it leaves the mask as it is (photometric). Flip moves the label.

``_pos`` is AdaAug's positive v (the PIL data with +v; a factor above 1). AdaAug's
``random_mirror`` negates the five geometric ops' v with p = 0.5, so for them the
direction is a coin; the enhance ops have no mirror, so their direction is the level's
side of 0.5 (below it the factor is under 1: ``_neg``). The host maps a level to its op
(``hosts/adaaug.py``).

Which ops differentiate: the 18 split ops, through autograd. Solarize, Posterize and
Cutout step with the magnitude and the other four have none: ``grad_mode="none"``. AdaAug
itself never differentiates an op: its search passes every op's gradient in the level
straight through (``stop_gradient``), and so does ``src/adaaug/``. The autograd rows are
the probe's.
"""

import dataclasses
from typing import Dict

from . import madaug
from .deepaa import exact_spec, op_name, split_specs

FAMILY = "adaaug"
SOURCE = "AdaAug (Cheung & Yeung, ICLR 2022), operation.py"

#: AdaAug's ``config.OPS_NAMES`` without Identity, in its order.
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
    "Cutout",
    "Flip",
)

#: transform id -> the split op's mag_range upper end: AUGMENT_LIST's [-hi, hi] (the
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

#: The ops ``madaug`` already holds as AdaAug has them.
MADAUG_IDS = ("Cutout", "Flip")


def _specs():
    madaug_ops = {spec.name: spec for spec in madaug.OPS}
    for transform_id in OP_ORDER:
        if transform_id in SPLIT_RANGES:
            yield from split_specs(FAMILY, SOURCE, transform_id, SPLIT_RANGES[transform_id])
        elif transform_id in MADAUG_IDS:
            spec = madaug_ops[f"{madaug.FAMILY}.{transform_id}"]
            yield dataclasses.replace(spec, name=op_name(FAMILY, transform_id), family=FAMILY, source=SOURCE)
        else:
            yield exact_spec(FAMILY, transform_id)


OPS = tuple(_specs())
