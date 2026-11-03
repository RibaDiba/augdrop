"""The two MADAug ops dropaug had no op for: Cutout and Flip.

Hou, Zhang & Zhou, "When to Learn What: Model-Adaptive Data Augmentation Curriculum"
(ICCV 2023), §4.1, and its code (github.com/JackHck/MADAug, ``operation.py``, which is
AdaAug's). MADAug's op list is Fast AutoAugment's 15 PIL ops plus Flip and Identity.
13 of them are AutoAugment's Table 6 ops at the same ranges, which are vocab's
``randaug_default`` ops, and Invert is ``autoaugment.Invert``; the MADAug vocab
(configs/vocab/madaug.yaml) uses those. That leaves:

    Cutout   ``Cutout(img, v)``, v in [0, 0.2]: a square of side v * width (the width,
             for both sides), filled with (125, 123, 114), centred at random. 0 is the
             identity
    Flip     ``PIL.ImageOps.mirror``: a horizontal flip. No magnitude. ("not from the
             paper" in AdaAug's code: AutoAugment had no flip)

Both are torch, batched, and carry no gradient in the magnitude (grad_mode "none"):
Cutout's square has a hard edge, so its output steps with the side length in pixels.

Cutout follows ``CutoutAbs`` exactly: corners ``int(max(0, c - side / 2))``, far corner
``min(size, corner + side)``, and PIL's rectangle fills both ends inclusive, at integer
(truncated) coordinates. The centre is ``np.random.uniform(w)`` there, which numpy reads
as low = w, high = 1, so it is uniform on (1, w], not [0, w). That is ported as it is. The
B centres are drawn in one ``torch.rand(B, 2)`` call on the CPU generator, so the probe's
per-op seed (``probe.apply_op``) and the audit's seeded run give every call the same
squares, and bumping one image's magnitude leaves the others alone.

The label: Cutout moves no pixels, so it is photometric and the label stays as it is
(MADAug is classification-only; its cut-out area keeps its class). Flip moves the label
with the image.
"""

import torch

from ..op import OpSpec
from .randaug_default import _per_image

FAMILY = "madaug"
SOURCE = "MADAug / AdaAug operation.py (Hou et al., 2023)"

#: CutoutAbs's fill, in 8-bit RGB.
CUTOUT_FILL = (125, 123, 114)


def cutout(images, magnitude, label=None):
    """MADAug's ``Cutout``, per image: a square of side ``magnitude * width``."""
    b, _, h, w = images.shape
    mags = _per_image(magnitude, b, images.dtype)
    # One draw per image whatever its magnitude, so the squares don't depend on which
    # images are cut. u -> w + (1 - w) u is numpy's uniform(low=w, high=1).
    centres = torch.rand(b, 2)
    fill = torch.tensor(CUTOUT_FILL, dtype=images.dtype, device=images.device).view(3, 1, 1) / 255.0
    out = images.clone()
    for i, v in enumerate(mags):
        if v <= 0.0:
            continue
        side = v * w
        cx = w + (1.0 - w) * float(centres[i, 0])
        cy = h + (1.0 - h) * float(centres[i, 1])
        x0 = int(max(0.0, cx - side / 2.0))
        y0 = int(max(0.0, cy - side / 2.0))
        x1 = min(float(w), x0 + side)
        y1 = min(float(h), y0 + side)
        out[i, :, y0 : int(y1) + 1, x0 : int(x1) + 1] = fill
    return out, label


def flip(images, magnitude=None, label=None):
    """``PIL.ImageOps.mirror``: left-right, the label with it."""
    out = torch.flip(images, dims=(-1,))
    return out, (None if label is None else torch.flip(label, dims=(-1,)))


OPS = (
    OpSpec(
        name=f"{FAMILY}.Cutout",
        family=FAMILY,
        kind="photometric",
        source=SOURCE,
        mag_range=(0.0, 0.2),
        identity_at=0.0,
        grad_mode="none",
        warps_label=False,
        fn=cutout,
    ),
    OpSpec(
        name=f"{FAMILY}.Flip",
        family=FAMILY,
        kind="geometric",
        source=SOURCE,
        mag_range=None,
        identity_at=None,
        grad_mode="none",
        warps_label=True,
        fn=flip,
    ),
)
