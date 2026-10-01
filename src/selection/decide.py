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
"""

from collections.abc import Mapping

from .base import Redundancy, Selection, Selector, as_relevance, declined

__all__ = ["decide", "redundancy_of"]


def redundancy_of(analysis, mask_within_op: bool = True) -> Redundancy:
    """The ``Redundancy`` a ``stats.Analysis`` hands a selector."""
    return Redundancy(
        analysis.names,
        analysis.primary,
        survives=analysis.survives,
        mask_within_op=mask_within_op,
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
