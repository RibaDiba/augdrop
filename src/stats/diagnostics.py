"""How much independent structure R holds, read off its spectrum.

    effective_rank       exp(entropy of lambda / sum lambda) (Roy & Vetterli 2007). K
                         mutually uncorrelated ops give K; K copies of one op give 1.
    participation_ratio  (sum lambda)^2 / sum lambda^2. Same endpoints, and weighted
                         toward the large eigenvalues.
    condition_number     lambda_max / lambda_min; inf when R is singular (an exact
                         linear dependence, e.g. an alias ``dedup`` didn't merge).
    vif                  per op, 1 / (1 - R^2 of that op regressed on all the others);
                         inf when the op is in an exact dependence.
    n_signal             eigenvalues above the Marchenko-Pastur upper edge, when the
                         number of images is known.

What to compare them against. Sampling noise alone spreads the spectrum of K
independent ops measured on N images over the Marchenko-Pastur interval
((1 - sqrt(K/N))^2, (1 + sqrt(K/N))^2), which is 0.47 to 1.73 at K/N = 0.1. So the
effective rank of an R of truly independent ops is below K, and only eigenvalues above
the upper edge are evidence of shared structure. A per-image-normalized R also sits at
``closure_null(K)`` off the diagonal rather than at 0, which lifts every statistic
here slightly.

Only R's finite part is read: an op whose off-diagonal row is all NaN (dropped,
unprobed, an alias) is set aside, reported in ``excluded``, and gets a NaN VIF. A
matrix with any other NaN cell raises, because its spectrum is undefined.
"""

import math
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Diagnostics", "diagnose", "effective_rank", "marchenko_pastur_edges", "participation_ratio", "vif"]

#: An eigenvalue below this fraction of the largest counts as 0.
_EIG_RTOL = 1e-10
#: An R^2 within this of 1 is an exact dependence.
_R2_ATOL = 1e-12


def _spectrum(r: np.ndarray) -> np.ndarray:
    """Eigenvalues of a symmetric matrix, ascending, with round-off negatives at 0."""
    eigvals = np.linalg.eigvalsh((r + r.T) / 2.0)
    return np.clip(eigvals, 0.0, None)


def effective_rank(eigvals: np.ndarray) -> float:
    total = float(eigvals.sum())
    if total <= 0:
        return float("nan")
    p = eigvals[eigvals > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eigvals: np.ndarray) -> float:
    squares = float((eigvals**2).sum())
    return float(eigvals.sum() ** 2 / squares) if squares > 0 else float("nan")


def vif(r: np.ndarray) -> np.ndarray:
    """Per-row variance inflation factor of a finite correlation matrix."""
    k = r.shape[0]
    out = np.full(k, np.nan)
    if k < 2:
        return out
    for i in range(k):
        others = np.delete(np.arange(k), i)
        r_io = r[others, i]
        r2 = float(r_io @ np.linalg.pinv(r[np.ix_(others, others)], hermitian=True) @ r_io)
        out[i] = math.inf if r2 >= 1.0 - _R2_ATOL else 1.0 / (1.0 - r2)
    return out


def marchenko_pastur_edges(n_ops: int, n_images: int) -> Tuple[float, float]:
    """The eigenvalue interval of a K x K sample correlation of K independent ops on N
    images, for large K and N."""
    ratio = n_ops / n_images
    return (1.0 - math.sqrt(ratio)) ** 2, (1.0 + math.sqrt(ratio)) ** 2


class Diagnostics(NamedTuple):
    """The statistics above for one R. ``names`` is R's full axis order; ``vif`` is
    aligned to it. ``eigenvalues`` are over the ``used`` ops, descending."""

    names: Tuple[str, ...]
    used: Tuple[str, ...]
    excluded: Tuple[str, ...]
    eigenvalues: np.ndarray
    effective_rank: float
    participation_ratio: float
    condition_number: float
    vif: np.ndarray
    n_images: Optional[int] = None
    mp_edges: Optional[Tuple[float, float]] = None
    n_signal: Optional[int] = None

    def as_dict(self) -> dict:
        """For the logs (``stats.jsonable`` turns inf into null there)."""
        return {
            "used": list(self.used),
            "excluded": list(self.excluded),
            "eigenvalues": self.eigenvalues.tolist(),
            "effective_rank": self.effective_rank,
            "participation_ratio": self.participation_ratio,
            "condition_number": self.condition_number,
            "singular": math.isinf(self.condition_number),
            "vif": {n: float(v) for n, v in zip(self.names, self.vif) if n in self.used},
            "n_images": self.n_images,
            "mp_edges": None if self.mp_edges is None else list(self.mp_edges),
            "n_signal": self.n_signal,
        }


def diagnose(r, names: Sequence[str], n_images: Optional[int] = None) -> Diagnostics:
    """Every statistic above for the correlation matrix ``r`` over ``names``. Give
    ``n_images`` (the N R was estimated on) for the Marchenko-Pastur reference."""
    r = np.asarray(r, dtype=np.float64)
    names = tuple(str(n) for n in names)
    if r.shape != (len(names), len(names)):
        raise ValueError(f"r is {r.shape} but {len(names)} names were given")

    off = np.where(np.eye(len(names), dtype=bool), np.nan, r)
    measured = np.isfinite(off).any(axis=1) if len(names) > 1 else np.isfinite(np.diag(r))
    idx = np.flatnonzero(measured)
    sub = r[np.ix_(idx, idx)]
    if not np.isfinite(sub).all():
        raise ValueError("r has NaN cells between measured ops; its spectrum is undefined")

    eigvals = _spectrum(sub) if idx.size else np.empty(0)
    top = float(eigvals[-1]) if eigvals.size else 0.0
    bottom = float(eigvals[0]) if eigvals.size else 0.0
    condition = math.inf if bottom <= _EIG_RTOL * top else top / bottom

    full_vif = np.full(len(names), np.nan)
    full_vif[idx] = vif(sub)

    edges = n_signal = None
    if n_images is not None and idx.size:
        edges = marchenko_pastur_edges(idx.size, int(n_images))
        n_signal = int((eigvals > edges[1]).sum())

    return Diagnostics(
        names=names,
        used=tuple(names[i] for i in idx),
        excluded=tuple(n for i, n in enumerate(names) if not measured[i]),
        eigenvalues=eigvals[::-1].copy(),
        effective_rank=effective_rank(eigvals) if eigvals.size else float("nan"),
        participation_ratio=participation_ratio(eigvals) if eigvals.size else float("nan"),
        condition_number=condition if eigvals.size else float("nan"),
        vif=full_vif,
        n_images=None if n_images is None else int(n_images),
        mp_edges=edges,
        n_signal=n_signal,
    )
