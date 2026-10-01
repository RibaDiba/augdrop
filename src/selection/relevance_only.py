"""Hard: keep the top-k ops by relevance, and ignore R. The ablation of mRMR's
redundancy term.

Not in Sensaug. It answers the question an mRMR result invites alongside the
``random_k`` null: did the redundancy term change what was kept, or would ranking
by the prior alone have kept the same ops? It uses R only to decide which ops are
measured, so its pool, exemptions and budget are exactly mRMR's.

Ties break by op name, never by R, since breaking them on redundancy would bring
back the term this arm exists to remove. When the prior is flat over the measured
ops (every host but SensAug's) there is nothing to rank, and it declines rather
than keep an arbitrary alphabetical prefix. ``random_k`` is the arm for an
arbitrary subset.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import Decline, HardSelector

__all__ = ["RelevanceOnly"]

#: Relative spread below which the prior counts as flat.
_FLAT_RTOL = 1e-12


class RelevanceOnly(HardSelector):
    """Top-k ops by prior mass. ``lam`` derives the budget as in ``MRMR``."""

    name = "relevance_only"

    def __init__(self, *, lam: Optional[float] = None):
        super().__init__(lam=lam)

    def _rank(self, sub: np.ndarray, rel: np.ndarray, budget: int, names: List[str]) -> List[int]:
        if np.ptp(rel) <= _FLAT_RTOL * float(np.abs(rel).max()):
            raise Decline(
                f"relevance is flat across the {len(names)} measured ops, so "
                f"relevance-only has nothing to rank by"
            )
        # names is sorted, so a stable sort breaks ties by name.
        return list(np.argsort(-rel, kind="stable"))
