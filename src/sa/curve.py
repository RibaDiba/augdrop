"""The SA curve: for each op, the levels where its cumulative sensitivity crosses
evenly spaced targets.

Ported from Sensaug's ``sensaug/sensitivity_analysis.py``:
``adaptive_sensitivity_analysis_new`` (here ``sa_curve``), ``objective_function`` and
``pchip_interpolator``. The bodies are unchanged. The one change is that ``sa_curve``
takes an ``evaluate(op, level) -> (miou, kid)`` callable instead of an mmengine runner,
so nothing here knows a training framework. ``tasks/mmseg/loops.py`` supplies
``evaluate`` (Sensaug's ``calculate_miou_kid_new``: set the op on the preprocessor's
eval slot, run ``SubsetTestLoop``).

How a curve is found, per op: score the clean model and the op at ``max_level``, seed
the interpolant with ``(min_level, 0)`` and ``(max_level, 2)``, and ask PCHIP where the
objective crosses the ``num_levels - 1`` interior targets of ``linspace(0, 2,
num_levels + 1)``. Evaluate the level whose crossing is least certain, insert it, and
repeat until every crossing's uncertainty is under ``tolerance * max_level``. The
objective is the mIoU drop over the drop at ``max_level``, plus ``2 * level /
max_level``, minus the KID over the KID at ``max_level``.

Kept as in Sensaug, on purpose:

- there is no cap on evaluations. The search ends when the PCHIP crossings settle.
- ``PchipInterpolator.solve(target)[0]`` raises IndexError when the objective never
  reaches a target (it can, since the KID term is subtracted).
"""

import bisect
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

__all__ = ["Evaluate", "objective_function", "pchip_interpolator", "sa_curve"]

#: ``evaluate(op, level) -> (miou, kid_mean)``: the model scored with ``op`` at
#: ``level`` applied to every image of the SA subset.
Evaluate = Callable[[str, float], Tuple[float, float]]


def sa_curve(
    evaluate: Evaluate,
    ops: Iterable[str],
    miou_base: float,
    num_levels: int = 5,
    tolerance: float = 0.05,
    min_level: float = 0.0,
    max_level: float = 1.0,
    log: Optional[Callable[[str], None]] = print,
) -> Dict[str, List[float]]:
    """``{op: levels}``, ``num_levels - 1`` ascending levels per op, as Sensaug's
    ``adaptive_sensitivity_analysis_new`` computed them.

    ``miou_base`` is the clean model's mIoU on the same subset ``evaluate`` scores.
    ``ops`` is the op list already filtered (Sensaug's ``exclude``: the static prune
    and ``remove_H``)."""
    log = log or (lambda message: None)
    predictor = pchip_interpolator
    perturbation_levels: Dict[str, List[float]] = {}

    for perturbation in ops:
        error = tolerance * max_level

        # calculate mean IoU and KID for baseline and max perturbation
        max_miou, max_kid = evaluate(perturbation, max_level)
        points = np.array([(min_level, 0.0), (max_level, 2.0)])

        # target y values that are evenly spaced between 0 and 2
        targets = np.linspace(0, 2, num_levels + 1)[1:-1]

        levels, errors = predictor(points, targets)

        # check if max error is less then tolerance
        index = np.argmax(errors)
        level, max_error = levels[index], errors[index]

        while max_error >= error:
            # calculate mean IoU and KID for perturbation level
            miou, kid = evaluate(perturbation, level)
            value = objective_function(
                level, max_level, miou_base - miou, miou_base - max_miou, kid, max_kid
            )
            points = np.insert(
                points, bisect.bisect(points[:, 0], level), (level, value), axis=0
            )

            levels, errors = predictor(points, targets)

            # check if max error is less then tolerance
            index = np.argmax(errors)
            level, max_error = levels[index], errors[index]

        # final perturbation levels
        final_levels = list(levels)
        perturbation_levels[perturbation] = final_levels

        log(f"{perturbation}: sampled {len(points) - 1} levels")
        log(f"{perturbation}: final perturbation levels: {final_levels}")

    return perturbation_levels


def objective_function(
    level: float,
    max_level: float,
    miou: float,
    max_miou: float,
    kid: float,
    max_kid: float,
) -> float:
    """Objective function that represents cumulative sensitivity.

    Args:
        level: Perturbation level
        max_level: Maximum perturbation level
        miou: Mean IoU (the drop from clean, as ``sa_curve`` passes it)
        max_miou: Maximum mean IoU (the drop at ``max_level``)
        kid: KID
        max_kid: Maximum KID

    Returns:
        Objective value
    """
    return (
        (miou / (max_miou + np.finfo(np.float32).eps))
        + 2 * (level / max_level)
        - (kid / (max_kid + np.finfo(np.float32).eps))
    )


def pchip_interpolator(points: NDArray, targets: NDArray) -> Tuple[NDArray, NDArray]:
    """Estimate function values and errors at target points using PCHIP interpolation.

    Args:
        points: Array of points
        targets: Array of target y values

    Returns:
        Function values
        Errors
    """
    values: List[float] = []
    errors: List[float] = []

    for target in targets:
        interpolator = PchipInterpolator(points[:, 0], points[:, 1], extrapolate=False)
        value = interpolator.solve(target)[0]

        # check if level is already in points
        if np.any(np.abs(points[:, 0] - value) < 1e-10):
            values.append(value)
            errors.append(0)
            continue

        # calculate possible x and y differences
        index = bisect.bisect(points[:, 0], value) - 1
        x_diff = (points[index + 1, 0] - points[index, 0]) / points[-1, 0]
        y_diff = min(
            abs(interpolator(points[index, 0]) - target),
            abs(interpolator(points[index + 1, 0]) - target),
        )

        # lower bound interpolation
        points1 = np.insert(
            points, index + 1, (value, target - y_diff * (0.6 + x_diff / 5)), axis=0
        )
        value1 = PchipInterpolator(
            points1[:, 0], points1[:, 1], extrapolate=False
        ).solve(target)[0]

        # upper bound interpolation
        points2 = np.insert(
            points, index + 1, (value, target + y_diff * (0.6 + x_diff / 5)), axis=0
        )
        value2 = PchipInterpolator(
            points2[:, 0], points2[:, 1], extrapolate=False
        ).solve(target)[0]

        values.append(value)
        errors.append(abs(value1 - value2) / 2)

    return np.array(values), np.array(errors)
