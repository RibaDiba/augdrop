"""AutoAugment's one op that RandAugment dropped: Invert, applied by torchvision.

AutoAugment (Cubuk et al., "AutoAugment: Learning Augmentation Policies from
Data", 2019) lists 16 ops in its Table 6. RandAugment kept 13 of them, at the
same ranges, and added Identity. Those 13 are `randaug_default`'s ops, and
AutoAugment uses them under those names (see hosts/autoaugment.py). Of the other
three, torchvision's AutoAugment implements only Invert (not Cutout or
SamplePairing), so this family holds Invert alone:

    Invert    "Invert the pixels of the image."    no magnitude

Implementation, 8-bit handling and grad_mode as in `randaug_default`; the op
calls AutoAugment's own dispatcher.
"""

from torchvision.transforms.v2 import AutoAugment

from .randaug_default import AA_IMAGE_SIZE, build_ops

FAMILY = "autoaugment"
SOURCE = "torchvision.transforms.v2.AutoAugment (Cubuk et al., 2019)"

RANGES = {"Invert": None}

OPS = build_ops(FAMILY, AutoAugment(), RANGES, AA_IMAGE_SIZE, SOURCE, op_order=("Invert",))
