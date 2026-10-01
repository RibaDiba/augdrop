"""Exact aliases: two rows of G that are one measurement under two names. Merge them
before anything else is computed on G.

Two vocab ops can be the same function. torchvision's RandAugment and
TrivialAugmentWide dispatch Solarize through the same code over the same range, so
``randaug_default.Solarize`` and ``trivialaug_wide.Solarize``, probed at the same
magnitude, give the same row of G bit for bit. Left in, that one op is counted twice:

- ``shared_image_factor`` is the mean |g| over ops, so the duplicated op weighs double
  in every image's normalizer
- ``closure_null`` is -1/(K-1), and K counts the duplicate
- the pair sits at r = 1 exactly, which raises red(a) for both, makes R singular
  (``diagnostics``, and the QPFS / DPP kernels), and adds a test to the FDR family
  whose answer is known in advance

An alias is a row equal to an earlier row to within ``rtol`` of the larger row's scale
(``ALIAS_RTOL``, loose enough for GPU reduction-order noise, far tighter than two
different ops ever agree on 100+ images). Proportional rows are not aliases: the same
op in two unit systems is a different measurement point. Neither is a row with no
variance (that op is ``dropped``) or an all-NaN row (``unprobed``). The first row of a
group, in G's order, represents it. The others come back from ``analyze`` as NaN rows
and columns of R, listed in ``aliases`` and not in ``dropped``, so the matrix keeps
its shape and an alias is never read as an uncorrelated op.

Downstream, an alias has no row of its own, so a hard selector treats it as exempt and
keeps it even when it prunes the representative. Map it through ``aliases`` when its
fate should follow its representative's. Aliases only arise in vocabularies that mix
families (probe studies), never in one host's.
"""

from typing import Dict, List, NamedTuple, Sequence

import numpy as np

from .corr import _STD_EPS

__all__ = ["ALIAS_RTOL", "Merged", "find_aliases", "merge_aliases"]

#: How close two rows must be, relative to the larger row's max |g|, to be one op.
ALIAS_RTOL = 1e-6


def find_aliases(d_grad: np.ndarray, rtol: float = ALIAS_RTOL) -> Dict[int, int]:
    """``{alias row: representative row}`` over a (A, N) G. The representative is the
    group's first row; rows with a NaN or no variance are never grouped."""
    d_grad = np.asarray(d_grad, dtype=np.float64)
    if d_grad.ndim != 2:
        raise ValueError(f"G must be (A, N), got {d_grad.shape}")
    if d_grad.shape[1] == 0:
        return {}
    candidates = np.flatnonzero(np.isfinite(d_grad).all(axis=1) & (d_grad.std(axis=1) >= _STD_EPS))
    scale = np.abs(d_grad).max(axis=1)

    alias_of: Dict[int, int] = {}
    for pos, i in enumerate(candidates):
        if i in alias_of:
            continue
        rest = np.array([j for j in candidates[pos + 1 :] if j not in alias_of], dtype=int)
        if not rest.size:
            continue
        gap = np.abs(d_grad[rest] - d_grad[i]).max(axis=1)
        for j in rest[gap <= rtol * np.maximum(scale[rest], scale[i])]:
            alias_of[int(j)] = int(i)
    return alias_of


class Merged(NamedTuple):
    """G with one row per group of aliases.

    d_grad   (K, N), G's rows at ``keep``
    names    the K names kept, in G's order
    keep     indices of the kept rows in the original G
    aliases  ``{alias name: representative name}``
    """

    d_grad: np.ndarray
    names: List[str]
    keep: np.ndarray
    aliases: Dict[str, str]


def merge_aliases(d_grad: np.ndarray, names: Sequence[str], rtol: float = ALIAS_RTOL) -> Merged:
    """Drop every alias row of G, keeping each group's representative."""
    d_grad = np.asarray(d_grad, dtype=np.float64)
    names = [str(n) for n in names]
    if d_grad.shape[0] != len(names):
        raise ValueError(f"G has {d_grad.shape[0]} rows but {len(names)} names were given")
    alias_of = find_aliases(d_grad, rtol)
    keep = np.array([i for i in range(len(names)) if i not in alias_of], dtype=int)
    return Merged(
        d_grad=d_grad[keep],
        names=[names[i] for i in keep],
        keep=keep,
        aliases={names[j]: names[i] for j, i in sorted(alias_of.items())},
    )
