"""The 8 PIL/AutoAugment photometric ops: brightness, contrast, sharpness, color,
each as a _pos/_neg pair.

Sensaug's photometric AutoAugment-family ops (differentiable_augmentations_aa.py),
same names, same order as in Sensaug's DIFF32_OPS. Built on kornia.enhance.

Units, as Sensaug defines them: an unsigned magnitude in [0, 1], direction in the
name. The enhance factor is 1 + m (positive) or 1 - 0.6 m (negative); color's
positive direction is 1 + 2 m.

These are the Sensaug versions, NOT the PIL/torchvision originals: brightness,
contrast and color differ from them in form (see TODO(calibration) in
`vocab.diff_ops`). The torchvision-faithful ops are in `randaug_default` and
`trivialaug_wide`.
"""

from ..diff_ops import brightness_op, color_op, contrast_op, photometric, sharpness_op
from ..op import OpSpec

FAMILY = "pil_autoaug"
SOURCE = "sensaug differentiable_augmentations_aa.py (kornia.enhance)"

_OPS = (
    ("brightness", brightness_op),
    ("contrast", contrast_op),
    ("sharpness", sharpness_op),
    ("color", color_op),
)

OPS = tuple(
    OpSpec(
        name=f"{base}_{direction}",
        family=FAMILY,
        kind="photometric",
        source=SOURCE,
        mag_range=(0.0, 1.0),
        identity_at=0.0,
        grad_mode="autograd",
        warps_label=False,
        fn=photometric(op, sign),
    )
    for base, op in _OPS
    for direction, sign in (("pos", +1.0), ("neg", -1.0))
)
