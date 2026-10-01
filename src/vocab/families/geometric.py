"""The 10 geometric ops: rotate, shear x/y, translate x/y, each as a _pos/_neg pair.

Sensaug's geometric AutoAugment-family ops (differentiable_augmentations_aa.py),
same names, same order as in Sensaug's DIFF32_OPS. Built on kornia's affine warp,
bilinear so the magnitude stays differentiable.

Units, as Sensaug defines them: an unsigned magnitude in [0, 1], direction in the
name, mirrored from Sensaug's CPU classes so m=1.0 means
    rotate      45 degrees about the image center
    shear       matrix element 0.3, about the image center
    translate   a quarter of the image extent

Every op here warps the label with the same matrix as the image (nearest,
out-of-frame -> 255, re-expressed in the label's own pixel frame). Calling the
image-only op and leaving the label alone would measure misalignment, not
sensitivity; `OpSpec.apply(images, m, label)` is the entry point.
"""

from ..diff_ops import (
    geometric,
    rotate_matrix,
    shear_x_matrix,
    shear_y_matrix,
    translate_x_matrix,
    translate_y_matrix,
)
from ..op import OpSpec

FAMILY = "geometric"
SOURCE = "sensaug differentiable_augmentations_aa.py (kornia affine warp)"

_MATRICES = (
    ("rotate", rotate_matrix),
    ("shear_x", shear_x_matrix),
    ("shear_y", shear_y_matrix),
    ("translate_x", translate_x_matrix),
    ("translate_y", translate_y_matrix),
)

OPS = tuple(
    OpSpec(
        name=f"{base}_{direction}",
        family=FAMILY,
        kind="geometric",
        source=SOURCE,
        mag_range=(0.0, 1.0),
        identity_at=0.0,
        grad_mode="autograd",
        warps_label=True,
        fn=geometric(matrix_fn, sign),
    )
    for base, matrix_fn in _MATRICES
    for direction, sign in (("pos", +1.0), ("neg", -1.0))
)
