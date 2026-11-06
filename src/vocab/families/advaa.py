"""Adversarial AutoAugment's ops, differentiable, each signed op split into _pos/_neg.

Zhang, Wang, Zhang & Zhong, "Adversarial AutoAugment" (ICLR 2020, arXiv 1912.11188),
§3.2: "There are totally 16 image operations in our search space, including
ShearX/Y, TranslateX/Y, Rotate, AutoContrast, Invert, Equalize, Solarize, Posterize,
Contrast, Color, Brightness, Sharpness, Cutout and Sample Pairing", with "the
parameter setting" deferred to AutoAugment (footnote 1). So the ranges are AutoAugment's
Table 6 (Cubuk et al., 2019), stated for 331 x 331 images. No code was released for
the paper.

    op                  mag_range       units                         AutoAugment Table 6
    ShearX/Y_pos/neg    [0, 0.3]        shear rate                    [-0.3, 0.3]
    TranslateX/Y_       [0, 150/331]    fraction of W / H             [-150, 150] px at 331
      pos/neg
    Rotate_pos/neg      [0, 30]         degrees                       [-30, 30]
    AutoContrast        none
    Invert              none
    Equalize            none
    Solarize            [0, 256]        threshold; identity 256       [0, 256]
    Posterize           [4, 8]          bits, int(v); identity 8      [4, 8]
    Contrast_, Color_,
    Brightness_,        [0, 0.9]        factor 1 +/- m                [0.1, 1.9]
    Sharpness_pos/neg
    Cutout              [0, 60/331]     square side, fraction of W    [0, 60] px at 331, gray
    SamplePairing       [0, 0.4]        blend weight of a partner     [0, 0.4]

The implementations are `deepaa`'s: AutoAugment's reference ops are PIL's, and
DeepAA's ``augmentation.py`` is that same lineage (through Fast AutoAugment) at the
same ranges, so ShearX/Y, TranslateX/Y and Rotate are PIL's AFFINE maps, sampled
bilinearly; the enhance ops are PIL's ImageEnhance blend in float; Solarize,
Posterize, AutoContrast, Equalize and Invert are the torchvision ops that equal PIL's
(``deepaa`` docstring). The translates scale to the image as torchvision's do: 150/331
of its width / height.

- **Cutout** is ``deepaa.cutout`` at a magnitude: "Set a random square patch of
  side-length magnitude pixels to gray", 60/331 of the width at most, about a uniform
  centre, clipped at the border, filled (125, 123, 114). It leaves the mask as it is (the
  user's choice for segmentation, as for Faster AutoAugment).
- **SamplePairing** is ``faster_aa.sample_pairing``: "Linearly add the image with
  another image (selected at random from the same mini-batch) with weight magnitude,
  without changing the label", ``(1 - m) x + m x[randperm]``. The mini-batch it pairs
  within is the whole augmented batch, the M copies included. It couples the batch's
  images (``couples_images``).

Each signed op is split: ``_pos`` is the positive end of AutoAugment's range (PIL's
data with +m; a factor above 1). The policy's magnitude bins span the signed range
(hosts/adversarial_autoaugment.py), so the sign comes from the bin.

Which ops differentiate: the 18 split ops and SamplePairing, through autograd.
Solarize, Posterize and Cutout step with the magnitude and the other three have none:
``grad_mode="none"``.
"""

from typing import Dict

from ..op import OpSpec
from .deepaa import exact_spec, op_name, split_specs
from .deepaa import cutout as _cutout
from .faster_aa import sample_pairing

FAMILY = "advaa"
SOURCE = "Adversarial AutoAugment (Zhang et al., ICLR 2020); AutoAugment Table 6"

#: AutoAugment's ranges are stated for 331 x 331 images.
AA_IMAGE_SIZE = 331.0

#: The paper's 16 ops, in its §3.2 order.
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
    "SamplePairing",
)

#: transform id -> the split op's mag_range upper end: Table 6's [-hi, hi] (the
#: enhance ops' [0.1, 1.9] as 1 +/- 0.9), translates as a fraction of the image.
SPLIT_RANGES: Dict[str, float] = {
    "ShearX": 0.3,
    "ShearY": 0.3,
    "TranslateX": 150.0 / AA_IMAGE_SIZE,
    "TranslateY": 150.0 / AA_IMAGE_SIZE,
    "Rotate": 30.0,
    "Contrast": 0.9,
    "Color": 0.9,
    "Brightness": 0.9,
    "Sharpness": 0.9,
}

#: Cutout's largest side, 60 px at 331, as a fraction of the width.
CUTOUT_MAX = 60.0 / AA_IMAGE_SIZE

#: SamplePairing's largest blend weight.
SAMPLE_PAIRING_MAX = 0.4


def _specs():
    for transform_id in OP_ORDER:
        if transform_id in SPLIT_RANGES:
            yield from split_specs(FAMILY, SOURCE, transform_id, SPLIT_RANGES[transform_id])
        elif transform_id == "Cutout":
            yield OpSpec(
                name=op_name(FAMILY, "Cutout"),
                family=FAMILY,
                kind="photometric",
                source=SOURCE,
                mag_range=(0.0, CUTOUT_MAX),
                identity_at=0.0,
                grad_mode="none",
                warps_label=False,
                fn=_cutout,
            )
        elif transform_id == "SamplePairing":
            yield OpSpec(
                name=op_name(FAMILY, "SamplePairing"),
                family=FAMILY,
                kind="photometric",
                source=SOURCE,
                mag_range=(0.0, SAMPLE_PAIRING_MAX),
                identity_at=0.0,
                grad_mode="autograd",
                warps_label=False,
                fn=sample_pairing,
                couples_images=True,
            )
        else:
            yield exact_spec(FAMILY, transform_id)


OPS = tuple(_specs())
