"""Finite-difference fallback for ops autograd can't differentiate.

Sensaug only ever probed its 32 autograd ops. dropaug also probes vocab ops with
``grad_mode="none"``, e.g. the torchvision RandAugment/TrivialAugment families, which
round to uint8 and take the magnitude as a python float, so autograd has no path to
it. For those, the per-image sensitivity is the central difference of the per-image
loss:

    g_i = (L_i(m_i + h) - L_i(m_i - h)) / 2h

All images are perturbed at once, two forwards per op rather than 2B. That is only
valid because image i's loss depends on image i's magnitude alone, which is the same
eval()-mode independence the autograd path rests on (see grad_collect).

Two caveats travel with every finite-difference row:

- ``evaluate`` must use COMMON RANDOM NUMBERS. A stochastic op has to draw the same
  sample on both sides of the difference, or the row measures the sample's noise
  divided by 2h. grad_collect reseeds the op before every evaluation.
- For a quantized op this is a secant slope at scale ``h``, not a derivative. A
  step that falls inside one bin (Posterize's integer bits) returns exactly zero, so
  such an op needs a wider step than the default. Every row records whether it came
  from autograd or from here, because the two are only interchangeable for smooth
  ops.

Pure numpy, so the numerics are testable without a model.
"""

from typing import Callable

import numpy as np

# The default step, as a fraction of an op's mag_range width. Ops carry their own
# units (degrees, pixels, enhance factors), so an absolute default would mean
# something different for every family. On Sensaug's [0, 1] ops it is 0.05.
DEFAULT_STEP = 0.05


def fd_points(magnitude: np.ndarray, step: float, lo: float = 0.0, hi: float = 1.0):
    """The two magnitudes each image is evaluated at, clipped into ``[lo, hi]``.

    Interior images get a central difference over ``2 * step``. An image within
    ``step`` of a rail gets a one-sided difference, because the op is undefined past
    its range (a negative blur sigma, a brightness past the rail).
    """
    if not step > 0:
        raise ValueError(f"finite-difference step must be positive, got {step}")
    if not lo < hi:
        raise ValueError(f"magnitude range must satisfy lo < hi, got ({lo}, {hi})")
    magnitude = np.asarray(magnitude, dtype=np.float64)
    if magnitude.size and (magnitude.min() < lo or magnitude.max() > hi):
        raise ValueError(
            f"magnitudes span [{magnitude.min():.4g}, {magnitude.max():.4g}], "
            f"outside the op's range [{lo}, {hi}]"
        )
    return np.maximum(magnitude - step, lo), np.minimum(magnitude + step, hi)


def fd_grad(
    evaluate: Callable[[np.ndarray], np.ndarray],
    magnitude: np.ndarray,
    *,
    step: float,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Per-image ``d loss / d magnitude`` by finite differences.

    ``evaluate`` maps a (B,) magnitude array to the (B,) per-image losses, with the
    op's randomness pinned (see the module docstring). ``step`` is in the op's own
    units. Returns a (B,) float64 array.
    """
    lower, upper = fd_points(magnitude, step, lo, hi)
    return (evaluate(upper) - evaluate(lower)) / (upper - lower)
