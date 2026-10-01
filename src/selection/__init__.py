"""DECIDE: turn R into q, the per-op weights a host installs with ``set_weights``.

    R = Redundancy(names, R_scalenorm, survives=survives_in_A)
    rel = dict(zip(host.vocabulary(), host.base_probs()))
    sel = build_selector("mrmr", lam=0.25).select(R, rel)
    if sel.applied:
        host.set_weights(sel.q)

    base.py            the contract (``base`` docstring): Redundancy, Selection,
                       Selector, HardSelector; pairwise, budget_from_lambda,
                       ramp_lambda
    kl_reweight.py     soft: q(a) ~ p(a) exp(-lambda red(a)). Sensaug's Lever 3
    mrmr.py            hard: greedy min-redundancy max-relevance. Sensaug's mRMR arm
    qpfs.py            hard: quadratic-programming feature selection (new)
    dpp.py             hard: greedy MAP of a determinantal point process (new)
    random_k.py        hard: k ops at random. The null band; always run it
    relevance_only.py  hard: top-k by prior, R unused. The ablation of mRMR's
                       redundancy term
    decide.py          decide(analysis, rel, selector, role=...): the training-loop
                       policy (control probes and withheld R never become q)

Every hard selector ranks the same pool (ops with a positive prior and a usable row
in R) and keeps the same number at the same budget, so their results are directly
comparable, with ``random_k`` across seeds (``null_band``) as the baseline.

Named ``selection``, not ``select``: ``select`` is a stdlib module that
``subprocess`` imports, and ``src/`` sits first on ``sys.path``.
"""

from .base import (
    MODES,
    Decline,
    HardSelector,
    Redundancy,
    Selection,
    Selector,
    budget_from_lambda,
    pairwise,
    ramp_lambda,
)
from .decide import decide, redundancy_of
from .dpp import DPP
from .kl_reweight import KLReweight, measured_red, reweight
from .mrmr import MRMR
from .qpfs import QPFS
from .random_k import RandomK, null_band
from .relevance_only import RelevanceOnly

__all__ = [
    "DPP",
    "Decline",
    "HARD_SELECTORS",
    "HardSelector",
    "KLReweight",
    "MODES",
    "MRMR",
    "QPFS",
    "RandomK",
    "Redundancy",
    "RelevanceOnly",
    "SELECTORS",
    "Selection",
    "Selector",
    "budget_from_lambda",
    "build_selector",
    "decide",
    "measured_red",
    "null_band",
    "pairwise",
    "ramp_lambda",
    "redundancy_of",
    "reweight",
]

SELECTORS = {
    cls.name: cls for cls in (KLReweight, MRMR, QPFS, DPP, RandomK, RelevanceOnly)
}

#: The selectors allowed to set an op to exactly 0. Everything else is held to
#: "soft, never zero" by tests/selection.
HARD_SELECTORS = frozenset(name for name, cls in SELECTORS.items() if cls.hard)


def build_selector(name: str, **kwargs) -> Selector:
    """Construct the selector registered under ``name`` with its keyword arguments."""
    try:
        cls = SELECTORS[name]
    except KeyError:
        raise ValueError(f"unknown selector {name!r}; expected one of {sorted(SELECTORS)}") from None
    return cls(**kwargs)
