"""red(a): collapse R into one redundancy score per op.

Ported from Sensaug's ``sensaug/redundancy.py`` (``compute_red`` and what it needs).
Its ``reweight`` / ``ramp_lambda`` are not here: turning a score into sampling
weights is the host's ``set_weights(q)`` (see ``hosts/``).

The score feeds a max-entropy tilt::

    q(a) proportional to  pdf_old(a) * exp(-lambda * red(a))

**red(a) is standardized, and that is load-bearing.** Writing red(a) = c + s(a) with
c the mean across ops, exp(-lambda*red) = exp(-lambda*c) * exp(-lambda*s), and the
constant factor cancels in the normaliser. Only deviations from the mean ever affect
the output. The raw row sums are not comparable across runs -- on Sensaug's logged
matrices their means span -0.71 to +4.08 and their standard deviations 0.24 to 3.69
-- so a lambda tuned against one checkpoint would mean something entirely different
at the next. Dividing by the standard deviation absorbs that into lambda' =
lambda/sigma and makes one lambda portable.

Pure numpy on purpose: no torch, no registries.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

_STD_FLOOR = 1e-8

MODES = ("squared", "abs", "signed")


class RedundancyScore(NamedTuple):
    """What compute_red returns.

    names:    op order, matching the rows of the matrix it was built from
    raw:      row sums before standardization, in the units of `mode`
    std:      the standardized score -- what reweight consumes
    dropped:  ops whose row was entirely unusable (all-NaN, or gated out)
    mode:     which reduction produced it
    reason:   None when the score is usable; otherwise why it is not, and the
              caller must leave the pdf alone. Never silently degrade -- a no-op
              that looks like a real run is the expensive failure here.
    """

    names: List[str]
    raw: np.ndarray
    std: np.ndarray
    dropped: List[str]
    mode: str
    reason: Optional[str]

    def as_dict(self) -> Dict[str, float]:
        """The standardized score keyed by op name, which is how reweight and the
        log both want it."""
        return {name: float(v) for name, v in zip(self.names, self.std)}

    @property
    def usable(self) -> bool:
        return self.reason is None


def within_op_pairs(names: Sequence[str]) -> List[Tuple[int, int]]:
    """Index pairs for the two halves of a single op, e.g. lighter_S/darker_S and
    sharpness_pos/sharpness_neg.

    Excluded from red(a) because they measure a parameterization convention, not
    redundancy between two augmentations someone might have chosen independently.
    The two directions of one op are the same structural relationship whichever
    sign it comes out with, and under the `signed` reduction a strongly negative
    within-op cell would hand that op the strongest protection in the matrix --
    exactly backwards.

    Derived from the names rather than hard-coded so it stays correct for both the
    14-op vocabulary the earlier runs logged and the 32-op one (6 pairs and 15
    pairs respectively).
    """
    index = {name: i for i, name in enumerate(names)}
    pairs = []
    for name, i in index.items():
        if name.startswith("lighter_"):
            partner = "darker_" + name[len("lighter_") :]
        elif name.endswith("_pos"):
            partner = name[: -len("_pos")] + "_neg"
        else:
            continue
        if partner in index:
            pairs.append((i, index[partner]))
    return pairs


def compute_red(
    r,
    names: Sequence[str],
    mode: str = "squared",
    mask_within_op: bool = True,
    survivor_mask=None,
    min_std: float = 1e-6,
) -> RedundancyScore:
    """Per-op redundancy score from the correlation matrix.

    r:             (A, A) correlation matrix. NaN is expected -- `correlate` leaves
                   dropped rows and columns NaN, and the json log writes them null.
    survivor_mask: (A, A) bool from fdr_correct. Cells that did not survive
                   multiplicity correction contribute nothing. Standard hygiene at
                   496 simultaneous tests, and it is also what keeps a `signed`
                   variant defensible if the closure question ever resolves toward
                   the constraint binding: a survivor subset sum stays unconstrained
                   even where the full row sum would not.

    `squared` is the default because it is closure-proof and does not depend on the
    sign convention of any individual op. `abs` and `signed` exist as ablation arms.
    """
    if mode not in MODES:
        raise ValueError(f"unknown red mode {mode!r}, expected one of {MODES}")

    names = list(names)
    matrix = np.asarray(r, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"r must be square, got shape {matrix.shape}")
    if matrix.shape[0] != len(names):
        raise ValueError(
            f"r is {matrix.shape[0]}x{matrix.shape[0]} but {len(names)} names were given"
        )

    n_ops = len(names)
    work = matrix.copy()

    # An op whose whole off-diagonal row is NaN was dropped upstream (constant
    # gradient row -> no correlation defined). Record it before the NaNs are
    # zeroed, otherwise it is indistinguishable from an op that genuinely
    # correlates with nothing.
    off_diagonal = ~np.eye(n_ops, dtype=bool)
    all_nan = np.all(np.isnan(np.where(off_diagonal, work, np.nan)), axis=1)
    dropped = [name for name, is_nan in zip(names, all_nan) if is_nan]

    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(work, 0.0)

    if mask_within_op:
        for i, j in within_op_pairs(names):
            work[i, j] = 0.0
            work[j, i] = 0.0

    if survivor_mask is not None:
        mask = np.asarray(survivor_mask, dtype=bool)
        if mask.shape != work.shape:
            raise ValueError(
                f"survivor_mask is {mask.shape}, expected {work.shape}"
            )
        work = work * mask

    if mode == "squared":
        raw = (work**2).sum(axis=1)
    elif mode == "abs":
        raw = np.abs(work).sum(axis=1)
    else:
        raw = work.sum(axis=1)

    std_dev = float(raw.std())
    reason = None
    if len(dropped) == n_ops:
        reason = "every op was dropped upstream; R carries no usable cell"
    elif not np.any(work):
        reason = (
            "no cell survived masking (within-op exclusion and/or the FDR gate); "
            "red(a) is identically zero"
        )
    elif std_dev < min_std:
        reason = (
            f"red(a) has standard deviation {std_dev:.3g} < {min_std:g}: every op is "
            f"equally redundant, so reweighting would be a no-op"
        )

    standardized = (raw - raw.mean()) / (std_dev + _STD_FLOOR)

    return RedundancyScore(
        names=names,
        raw=raw,
        std=standardized,
        dropped=dropped,
        mode=mode,
        reason=reason,
    )


def summarise(score: RedundancyScore, top: int = 5) -> str:
    """One-line human summary for the training log: which ops this would push down
    hardest and which it would leave alone."""
    if not score.usable:
        return f"red({score.mode}) unusable: {score.reason}"
    order = np.argsort(score.std)
    least = ", ".join(
        f"{score.names[i]}={score.std[i]:+.2f}" for i in order[:top]
    )
    most = ", ".join(
        f"{score.names[i]}={score.std[i]:+.2f}" for i in order[::-1][:top]
    )
    return (
        f"red({score.mode}) over {len(score.names)} ops "
        f"[raw mean {score.raw.mean():+.2f}, sd {score.raw.std():.2f}] -- "
        f"most redundant: {most} | least: {least}"
    )
