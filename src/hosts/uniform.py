"""Plain uniform sampling over a vocabulary: the no-policy baseline.

The default reproduces Sensaug's ``--random-aug`` bank: the 32 diff32 ops plus
one ``"none"`` entry, all equally likely, so P(none) = 1/(A+1).
"""

from typing import Iterable, Optional, Sequence

import numpy as np

from .base import AugHost
from .sensaug import DIFF32_OPS

__all__ = ["UniformHost"]


class UniformHost(AugHost):
    name = "uniform"

    def __init__(
        self,
        *,
        vocabulary: Sequence[str] = DIFF32_OPS,
        none_prob: Optional[float] = None,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        n = len(vocabulary)
        super().__init__(
            vocabulary,
            np.full(n, 1.0 / n) if n else np.zeros(0),
            none_prob=1.0 / (n + 1) if none_prob is None else none_prob,
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )
