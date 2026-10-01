"""SensAug's sensitivity analysis (SA), with no training framework.

Every SA round, Sensaug's ``RobustValLoop`` scores the model under each op at a few
levels, turns those scores into the training pdf over ``(op, level)`` pairs, and hands
the pdf to training. Every ``SA_CURVE_CADENCE`` rounds it first re-searches the levels
themselves (the SA curve). These are the parts of that loop that are math:

    curve.py       sa_curve: the adaptive per-op level search (PCHIP on mIoU drop,
                   level and KID), from sensitivity_analysis.py
    pdf.py         the three pdf generators (generate_pdf_new with descending_MA,
                   the weighted variant, uniform), from sensaug_loop.py
    magnitudes.py  the SA pdf -> the probe's magnitudes, from corr_magnitudes.py

The loop itself (when to run, what to evaluate, where the pdf goes) is glue:
``tasks/mmseg/loops.py``. Everything here takes plain dicts and callables.
"""

from .curve import Evaluate, objective_function, pchip_interpolator, sa_curve
from .magnitudes import (
    MAGNITUDE_MODES,
    MODE_FIXED,
    MODE_MODAL,
    MODE_SAMPLED_INDEPENDENT,
    MODE_SAMPLED_SHARED,
    conditional_levels,
    modal_magnitude,
    probe_magnitudes,
    resolve_magnitudes,
    sample_magnitudes,
)
from .pdf import (
    NONE_KEY,
    REMOVE_H_NAMES,
    dict_mean,
    generate_pdf_new,
    generate_pdf_new_weighted_aug,
    generate_uniform_pdf,
    level_metrics,
    none_prob_denominator,
    remove_H_perturbations,
)

__all__ = [
    "Evaluate",
    "MAGNITUDE_MODES",
    "MODE_FIXED",
    "MODE_MODAL",
    "MODE_SAMPLED_INDEPENDENT",
    "MODE_SAMPLED_SHARED",
    "NONE_KEY",
    "REMOVE_H_NAMES",
    "conditional_levels",
    "dict_mean",
    "generate_pdf_new",
    "generate_pdf_new_weighted_aug",
    "generate_uniform_pdf",
    "level_metrics",
    "modal_magnitude",
    "none_prob_denominator",
    "objective_function",
    "pchip_interpolator",
    "probe_magnitudes",
    "remove_H_perturbations",
    "resolve_magnitudes",
    "sa_curve",
    "sample_magnitudes",
]
