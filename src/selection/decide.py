"""One analysed probe -> the Selection a training run acts on.

The policy every training loop's glue applies between ``stats.analyze_sweep`` and
``host.set_weights``, in one place, so a loop's callbacks only gate and install:

- a ``"control"`` probe is measured and logged but never becomes a q
- an Analysis that is ``withheld`` (too few images, or the shared-factor alarm)
  never becomes a q
- otherwise R is the Analysis's ``primary`` R, gated by its FDR survivors, and the
  selector decides, declining as ``base`` describes

Every refusal comes back as a declined Selection (``applied`` False, ``reason`` set),
so the caller has one thing to check. ``tasks/mmseg/hooks.py``'s ApplyWeightsHook
predates this and inlines the same steps.

``publication`` is the other half of Sensaug's handoff, for a loop that re-applies the
latest R to every SA round's pdf (``tasks/mmseg/loops.RobustValLoop``): what an
emission does to the R the loop holds. It is Sensaug's ``prune_augmentations``.
"""

from collections.abc import Mapping
from typing import Optional, Tuple

import numpy as np

from stats.redundancy import compute_red

from .base import Redundancy, Selection, Selector, as_relevance, declined

__all__ = ["PUBLISH", "KEEP", "CLEAR", "decide", "publication", "redundancy_of"]

#: ``publication`` actions: replace the held R with this one, leave the held R as it
#: is, or drop it (the SA pdf then trains as generated until an R is published again).
PUBLISH, KEEP, CLEAR = "publish", "keep", "clear"


def redundancy_of(analysis, mask_within_op: bool = True) -> Redundancy:
    """The ``Redundancy`` a ``stats.Analysis`` hands a selector."""
    return Redundancy(
        analysis.names,
        analysis.primary,
        survives=analysis.survives,
        mask_within_op=mask_within_op,
        ci_lo=analysis.ci_lo,
    )


def decide(
    analysis,
    rel: Mapping,
    selector: Selector,
    *,
    role: str = "active",
    budget=None,
    mask_within_op: bool = True,
) -> Selection:
    """``selector``'s Selection on ``analysis``, or a declined one when the probe's
    ``role`` or the analysis rules a q out. ``rel`` is
    ``dict(zip(host.vocabulary(), host.base_probs()))``."""
    names, prior = as_relevance(rel)
    if role != "active":
        return declined(names, prior, f"{role} probe: R is the baseline, not a q")
    if analysis.withheld is not None:
        return declined(names, prior, f"R withheld: {analysis.withheld}")
    return selector.select(redundancy_of(analysis, mask_within_op), rel, budget)


def publication(
    analysis, *, role: str = "active", mode: str = "squared", mask_within_op: bool = True
) -> Tuple[str, Optional[str]]:
    """``(action, reason)``: what one emission does to the R an SA loop re-applies to
    each round's pdf. Sensaug's ``prune_augmentations`` (grad_sens_analysis.py), in its
    order:

    - a control probe, a skipped sweep (fewer than ``n_min`` images) and an R with no
      finite cell: ``KEEP``. Sensaug returned before touching ``runner.corr_redundancy``,
      so the last published R stays in effect
    - the shared-factor alarm, and a red(a) ``compute_red`` calls unusable (no cell
      survives the FDR gate, or no spread): ``CLEAR``. Sensaug set
      ``runner.corr_redundancy = None``, so every later pdf trains unweighted until the
      next usable R, under every down-weighting method, mRMR included
    - otherwise ``PUBLISH``

    ``mode`` and ``mask_within_op`` are how Sensaug's hook scored red(a)
    (``--corr-red-mode``, ``--corr-keep-within-op``). The score is ``compute_red`` over
    the full R, as Sensaug's, not ``measured_red``."""
    if role != "active":
        return KEEP, f"{role} probe: R is the baseline, not a q"
    if analysis.skipped is not None:
        return KEEP, f"R skipped: {analysis.skipped}"
    r = analysis.primary
    if r is None or not np.isfinite(r).any():
        return KEEP, "R has no finite cell"
    if analysis.withheld is not None:
        return CLEAR, f"R withheld: {analysis.withheld}"
    score = compute_red(
        r, analysis.names, mode=mode, mask_within_op=mask_within_op, survivor_mask=analysis.survives
    )
    if not score.usable:
        return CLEAR, f"red(a) unusable: {score.reason}"
    return PUBLISH, None
