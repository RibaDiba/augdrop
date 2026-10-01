"""Hard: keep k of the measured ops at random. The null band; always run it.

A ranked selector's result means something only if it beats dropping the same
number of ops at random, and one random draw is not a baseline. Run this at several
seeds with the same budget (``null_band``) and read mRMR, QPFS and DPP against the
spread of outcomes.

Ported from Sensaug's ``null`` random-prune arm (``train.py``:
``draw_random_prune``). The draw is identical: the ops to DROP are
``RandomState(seed).choice(sorted(pool), n_drop, replace=False)``. What changes is
the pool. Sensaug drew once before training, with no R, from the vocabulary minus
the ops the run already excluded. Here the pool is the one every hard selector
ranks: ops with positive prior and a usable row in R. Ops the host excludes have
prior 0, so they are out as before, and ops with no row in R are exempt and kept,
as they are under mRMR. So at the same budget this keeps exactly as many ops as the
ranked selectors, out of the same pool.

The draw uses a private ``RandomState``, never the global numpy RNG, so it can't
shift the training stream's draws. It must be a pure function of the seed: every
DDP rank has to prune the same ops.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, List, Optional

import numpy as np

from .base import HardSelector, Redundancy, Selection

__all__ = ["RandomK", "null_band"]


class RandomK(HardSelector):
    """k measured ops kept uniformly at random.

    seed:  required. A null draw whose seed isn't recorded can't be reproduced.
    lam:   derive the budget as in ``MRMR``, so a null matches an mRMR arm's lam
    """

    name = "random_k"

    def __init__(self, *, seed: int, lam: Optional[float] = None):
        super().__init__(lam=lam)
        if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed < 2**32:
            raise ValueError(f"seed must be an integer in [0, 2**32), got {seed!r}")
        self.seed = int(seed)

    def _rank(self, sub: np.ndarray, rel: np.ndarray, budget: int, names: List[str]) -> List[int]:
        rng = np.random.RandomState(self.seed)
        dropped = set(rng.choice(list(names), size=len(names) - budget, replace=False).tolist())
        return [i for i, name in enumerate(names) if name not in dropped]


def null_band(
    R: Redundancy,
    rel: Mapping,
    budget: Optional[int] = None,
    *,
    seeds: Iterable[int],
    lam: Optional[float] = None,
) -> List[Selection]:
    """``random_k`` at each seed: the band a ranked selector's result has to clear."""
    return [RandomK(seed=seed, lam=lam).select(R, rel, budget) for seed in seeds]
