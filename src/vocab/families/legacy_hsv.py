"""The 14 legacy photometric ops: RGB / HSV lighter-darker, blur, noise.

Sensaug's `DIFFERENTIABLE_PERTURBATIONS`, same names, same order (the first 14
axes of every Sensaug R). Implementations are in `vocab.diff_ops`; the lineage is
Shu et al., "Adversarial Differentiable Data Augmentation for Autonomous
Systems" (ICRA 2021), via Sensaug's differentiable_augmentations.py.

Units, as Sensaug defines them: an unsigned magnitude in [0, 1], direction baked
into the name. For the rail ops the magnitude is the blend weight toward the
rail; for `blur` it is sigma in pixels; for `noise` it is the std of the added
Gaussian noise. These are not commensurable with each other: at 0.5, `blur`
moves a pixel by at most ~78/255 and `noise` by up to ~251/255.
"""

from ..diff_ops import B, G, H, R, S, V, blur, color_channel, gaussian_noise, hsv_channel, photometric
from ..op import OpSpec

FAMILY = "legacy_hsv"
SOURCE = "sensaug differentiable_augmentations.py (Shu et al., ICRA 2021)"


def _spec(name, fn):
    return OpSpec(
        name=name,
        family=FAMILY,
        kind="photometric",
        source=SOURCE,
        mag_range=(0.0, 1.0),
        identity_at=0.0,
        grad_mode="autograd",
        warps_label=False,
        fn=fn,
    )


def _rgb(channel, sign):
    return photometric(lambda images, d: color_channel(images, channel, d), sign)


def _hsv(channel, sign):
    return photometric(lambda images, d: hsv_channel(images, channel, d), sign)


OPS = (
    _spec("lighter_R", _rgb(R, +1.0)),
    _spec("darker_R", _rgb(R, -1.0)),
    _spec("lighter_G", _rgb(G, +1.0)),
    _spec("darker_G", _rgb(G, -1.0)),
    _spec("lighter_B", _rgb(B, +1.0)),
    _spec("darker_B", _rgb(B, -1.0)),
    _spec("lighter_H", _hsv(H, +1.0)),
    _spec("darker_H", _hsv(H, -1.0)),
    _spec("lighter_S", _hsv(S, +1.0)),
    _spec("darker_S", _hsv(S, -1.0)),
    _spec("lighter_V", _hsv(V, +1.0)),
    _spec("darker_V", _hsv(V, -1.0)),
    _spec("blur", photometric(lambda images, d: blur(images, sigma=d))),
    _spec("noise", photometric(lambda images, d: gaussian_noise(images, delta=d))),
)
