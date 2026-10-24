"""One G -> everything Sensaug's ``_emit`` derived from it: R_raw, R_scalenorm, the
shared-factor loadings, and the per-cell bootstrap CIs with the BH-FDR survivors.

Ported from ``PerturbationSensitivityAnalysisHookWithGradients._emit`` and
``_emit_bootstrap`` (sensaug/hooks/grad_sens_analysis.py): the same chain, in the same
order, with the mmengine runner, the rank gather and the log writing taken out. The
logs are ``logs.py``; turning the result into q is ``selection``.

One difference from Sensaug. A row of G that is entirely NaN is an op the probe could
not measure (``unprobed``: no magnitude, e.g. AutoContrast, Equalize, Invert). Sensaug's
chain let such a row into ``shared_image_factor``, which then came out NaN in every
column and took R_scalenorm to all NaN with it. Here those rows are set aside before
anything is computed on G, every other op is analysed as if they did not exist, and
they come back as NaN rows and columns of R, listed in ``unprobed`` and not in
``dropped``. On a G with no NaN row the two chains agree exactly.

Four additions, none of which changes R_raw or R_scalenorm on a DIFF32 G:

- exact aliases (``dedup``) are merged before anything else, and come back as NaN rows
  listed in ``aliases``. The DIFF32 ops have none, so there the chain is unchanged.
- ``shrinkage=`` (opt-in) adds a shrunk copy of the primary R (``shrinkage``), which
  then becomes the R a decision is taken on.
- every R gets its ``diagnostics`` (effective rank, VIF, condition number).
- ``residual_rank=`` (opt-in) adds R_residual, the shared factor regressed out and then
  rank-correlated (``residual``), which then becomes the R a decision is taken on. The
  NMS arm uses it with no bootstrap. With ``d_loss`` (the probe's loss increase,
  ``Sweep.loss_matrix()``) every analysis also carries each op's mean loss increase.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .corr import (
    _DIV_EPS,
    _LOADING_ALARM,
    closure_null,
    correlate,
    shared_factor_loadings,
    shared_image_factor,
)
from .dedup import ALIAS_RTOL, find_aliases
from .diagnostics import Diagnostics, diagnose
from .fdr import FDR_METHODS, cluster_bootstrap_ci, fdr_correct
from .residual import residual_rank_corr
from .shrinkage import METHODS as SHRINKAGE_METHODS
from .shrinkage import shrink

__all__ = ["Analysis", "Bootstrap", "analyze", "analyze_sweep"]


@dataclass
class Bootstrap:
    """``_emit_bootstrap``'s output, in the space of the ops it tested: ``names`` are
    the probed, non-dropped ops, and every array is (K, K) over them."""

    names: list
    r: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    p: np.ndarray
    q: np.ndarray
    survives: np.ndarray
    null_r: float
    n_reps: int
    seed: int
    alpha: float
    method: str = "bh"


@dataclass
class Analysis:
    """Everything one G says, labelled.

    names        G's row order, and the axis order of every (A, A) array here
    skipped      why R was not computed at all (fewer than ``n_min`` images), else None
    R_raw        (A, A) Pearson across images, NaN for dropped and unprobed ops
    R_scalenorm  the same after dividing out the per-image shared factor; None when
                 ``normalize_per_image`` is off
    R_shrunk     R_scalenorm (or R_raw) shrunk as ``shrinkage`` records; None unless
                 asked for
    R_residual   Spearman R of G with the shared factor regressed out (``residual``);
                 None unless asked for
    residual_slopes
                 (A,) each op's slope on its leave-one-out factor, with R_residual
    loss_increase
                 (A,) each op's mean loss increase over the probe images, from
                 ``d_loss``; None without it, NaN for an op with no losses
    dropped      ops measured but with no usable row in ``primary``: no variance across
                 images (Sensaug's ``dropped``), or for R_residual also no scale
    unprobed     ops with an all-NaN row in G, never measured
    aliases      ``{alias: representative}``: ops whose row of G duplicated an earlier
                 op's, analysed once under the representative's name
    diagnostics  the spectrum statistics of ``primary`` (``stats.diagnostics``)
    labels       free-form provenance copied into every log record (checkpoint, iter,
                 role, magnitudes, probe set)
    """

    names: list
    n_images: int
    n_probes: int
    normalize_per_image: bool
    skipped: Optional[str] = None
    R_raw: Optional[np.ndarray] = None
    R_scalenorm: Optional[np.ndarray] = None
    dropped: list = field(default_factory=list)
    unprobed: list = field(default_factory=list)
    aliases: dict = field(default_factory=dict)
    shared_factor_loadings: Optional[np.ndarray] = None
    bootstrap: Optional[Bootstrap] = None
    R_shrunk: Optional[np.ndarray] = None
    shrinkage: Optional[dict] = None
    R_residual: Optional[np.ndarray] = None
    residual_slopes: Optional[np.ndarray] = None
    loss_increase: Optional[np.ndarray] = None
    diagnostics: Optional[Diagnostics] = None
    labels: dict = field(default_factory=dict)

    @property
    def primary(self) -> Optional[np.ndarray]:
        """The R a decision is taken on: R_residual or R_shrunk when asked for, else
        R_scalenorm when normalizing, else R_raw."""
        if self.R_residual is not None:
            return self.R_residual
        if self.R_shrunk is not None:
            return self.R_shrunk
        return self.R_scalenorm if self.normalize_per_image else self.R_raw

    @property
    def mean_offdiag(self) -> float:
        """The mean finite off-diagonal cell of ``primary``. For R_residual this is the
        health check: an independent pair sits close to 0 (``residual``), so a mean far
        from it says a shared factor got through the regression."""
        r = self.primary
        if r is None:
            return float("nan")
        cells = r[~np.eye(r.shape[0], dtype=bool)]
        cells = cells[np.isfinite(cells)]
        return float(cells.mean()) if cells.size else float("nan")

    @property
    def max_loading(self) -> float:
        loadings = self.shared_factor_loadings
        if loadings is None or not np.isfinite(loadings).any():
            return float("nan")
        return float(np.nanmax(np.abs(loadings)))

    @property
    def survives(self) -> Optional[np.ndarray]:
        """The FDR survivors scattered back to (A, A). A cell that was never tested
        (a dropped or unprobed op) did not survive. None when no bootstrap ran."""
        if self.bootstrap is None:
            return None
        index = {name: i for i, name in enumerate(self.names)}
        kept = [index[name] for name in self.bootstrap.names]
        full = np.zeros((len(self.names), len(self.names)), dtype=bool)
        full[np.ix_(kept, kept)] = self.bootstrap.survives
        return full

    @property
    def n_survivors(self) -> Optional[int]:
        if self.bootstrap is None:
            return None
        upper = np.triu_indices(len(self.bootstrap.names), k=1)
        return int(self.bootstrap.survives[upper].sum())

    @property
    def n_cells(self) -> Optional[int]:
        if self.bootstrap is None:
            return None
        k = len(self.bootstrap.names)
        return k * (k - 1) // 2

    @property
    def withheld(self) -> Optional[str]:
        """Why this R must not become a q, or None when it may.

        Sensaug withheld its score in two cases: no R at all, and the shared-factor
        alarm. When every op loads >0.9 on the per-image factor, R is ranking which
        images are HARD, and a q built from it would reweight augmentations on the
        strength of image difficulty.
        """
        if self.skipped is not None:
            return self.skipped
        if np.isfinite(self.max_loading) and self.max_loading > _LOADING_ALARM:
            return (
                f"max shared-factor loading {self.max_loading:.2f} > {_LOADING_ALARM}: R is "
                f"tracking which images are hard, not which augmentations are redundant"
            )
        return None

    def summary(self) -> str:
        """Sensaug's one-line ``[grad-corr]`` summary."""
        if self.skipped is not None:
            return f"SKIPPED ({self.skipped})"
        text = (
            f"n_images={self.n_images} n_probes={self.n_probes} "
            f"dropped={self.dropped or 'none'} unprobed={self.unprobed or 'none'} "
            + (f"aliases={self.aliases} " if self.aliases else "")
            + f"max_shared_loading={self.max_loading:.2f} "
            f"magnitudes={self.labels.get('magnitude_source', 'unknown')}/"
            f"{self.labels.get('magnitude_mode', 'unknown')}"
        )
        if self.n_cells:
            text += f" fdr_survivors={self.n_survivors}/{self.n_cells}"
        if self.shrinkage is not None:
            text += f" {self.shrinkage['method']}={self.shrinkage['intensity']:.3f}"
        if self.R_residual is not None:
            text += f" residual_rank mean_offdiag={self.mean_offdiag:.3f}"
        if self.diagnostics is not None:
            text += (
                f" eff_rank={self.diagnostics.effective_rank:.1f}/{len(self.diagnostics.used)}"
            )
        return text


def analyze(
    d_grad: np.ndarray,
    probe_ids: np.ndarray,
    names: Sequence[str],
    *,
    n_min: int = 100,
    normalize_per_image: bool = True,
    bootstrap: bool = True,
    bootstrap_reps: int = 1000,
    bootstrap_seed: int = 0,
    fdr_alpha: float = 0.05,
    fdr_method: str = "bh",
    merge_aliases: bool = True,
    alias_rtol: float = ALIAS_RTOL,
    shrinkage: Optional[str] = None,
    residual_rank: bool = False,
    d_loss: Optional[np.ndarray] = None,
    labels: Optional[dict] = None,
) -> Analysis:
    """Sensaug's ``_emit`` chain on one (A ops x N images) G.

    ``probe_ids[j]`` is the batch column j came from (``Sweep.matrix()``'s
    ``batch_ids``), the cluster the bootstrap resamples. Below ``n_min`` images
    (default 100, SE ~ 0.10) nothing is computed and ``skipped`` says why: Sensaug
    discarded such a window rather than pool it with the next model state.

    ``fdr_method`` is ``fdr_correct``'s ("bh", Sensaug's, or "by"). ``shrinkage`` is
    "ledoit_wolf" or "schafer_strimmer" (``stats.shrinkage``), applied to the G the
    primary R comes from; Schafer-Strimmer shrinks toward ``closure_null(K)`` when
    normalizing, else toward 0.

    ``residual_rank`` adds R_residual (``residual.residual_rank_corr``), which becomes
    ``primary``. It has no bootstrap and no shrinkage, so it refuses both. ``d_loss`` is
    the (A, N) loss increase aligned to G (``Sweep.loss_matrix()``); its row means are
    ``loss_increase``.
    """
    if fdr_method not in FDR_METHODS:
        raise ValueError(f"unknown FDR method {fdr_method!r}, expected one of {sorted(FDR_METHODS)}")
    if shrinkage is not None and shrinkage not in SHRINKAGE_METHODS:
        raise ValueError(f"unknown shrinkage {shrinkage!r}, expected one of {sorted(SHRINKAGE_METHODS)}")
    if residual_rank and (bootstrap or shrinkage is not None):
        raise ValueError(
            "residual_rank is a decision R of its own: its cells are not the ones the "
            "bootstrap tests or shrinkage pulls. Pass bootstrap=False and no shrinkage."
        )
    d_grad = np.asarray(d_grad, dtype=np.float64)
    probe_ids = np.asarray(probe_ids)
    names = [str(n) for n in names]
    if d_grad.ndim != 2 or d_grad.shape[0] != len(names):
        raise ValueError(f"G is {d_grad.shape} but {len(names)} op names were given")
    if probe_ids.shape != (d_grad.shape[1],):
        raise ValueError(f"probe_ids is {probe_ids.shape}, expected ({d_grad.shape[1]},)")
    if d_loss is not None:
        d_loss = np.asarray(d_loss, dtype=np.float64)
        if d_loss.shape != d_grad.shape:
            raise ValueError(f"d_loss is {d_loss.shape}, expected G's shape {d_grad.shape}")

    finite = np.isfinite(d_grad)
    unprobed_rows = ~finite.any(axis=1)
    partial = ~finite.all(axis=1) & ~unprobed_rows
    if partial.any():
        raise ValueError(
            f"rows {[names[i] for i in np.flatnonzero(partial)]} are partly NaN. The probe "
            f"measures an op on every image or on none, so this G is corrupt."
        )

    n_ops, n_images = d_grad.shape
    # Before anything reads G: a duplicated row would count its op twice in the
    # shared factor, in closure_null's K and in the FDR family.
    alias_of = find_aliases(d_grad, alias_rtol) if merge_aliases else {}
    result = Analysis(
        names=names,
        n_images=int(n_images),
        n_probes=int(np.unique(probe_ids).size),
        normalize_per_image=normalize_per_image,
        unprobed=[names[i] for i in np.flatnonzero(unprobed_rows)],
        aliases={names[j]: names[i] for j, i in sorted(alias_of.items())},
        labels=dict(labels or {}),
    )
    if d_loss is not None:
        # NaN propagates: an op is either measured on every image or on none.
        result.loss_increase = (
            d_loss.mean(axis=1) if n_images else np.full(n_ops, np.nan)
        )
    if n_images < n_min:
        result.skipped = (
            f"the sweep holds only {n_images} images (n_min={n_min}); discarded rather "
            f"than pooled with another model state"
        )
        return result

    probed = np.array([i for i in np.flatnonzero(~unprobed_rows) if i not in alias_of], dtype=int)
    g = d_grad[probed]

    r_raw, dropped = correlate(g)
    shared = shared_image_factor(g)
    loadings = shared_factor_loadings(g, shared)

    primary, r_norm = g, None
    if normalize_per_image:
        primary = g / (shared + _DIV_EPS)
        r_norm, dropped = correlate(primary)

    result.R_raw = _expand(r_raw, probed, n_ops)
    result.R_scalenorm = None if r_norm is None else _expand(r_norm, probed, n_ops)
    result.dropped = [names[probed[i]] for i in dropped]
    result.shared_factor_loadings = np.full(n_ops, np.nan)
    result.shared_factor_loadings[probed] = loadings

    if bootstrap:
        keep = np.setdiff1d(np.arange(probed.size), dropped)
        if keep.size >= 2:
            d_keep = primary[keep]
            r, _ = correlate(d_keep)
            # The per-image normalization closes the composition, so an independent
            # pair sits at closure_null(K), not at 0.
            null = closure_null(keep.size) if normalize_per_image else 0.0
            ci_lo, ci_hi, p = cluster_bootstrap_ci(
                d_keep, probe_ids, n_reps=bootstrap_reps, seed=bootstrap_seed, alpha=fdr_alpha, null=null
            )
            q, survives = fdr_correct(p, alpha=fdr_alpha, method=fdr_method)
            result.bootstrap = Bootstrap(
                names=[names[probed[i]] for i in keep],
                r=r,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                p=p,
                q=q,
                survives=survives,
                null_r=null,
                n_reps=bootstrap_reps,
                seed=bootstrap_seed,
                alpha=fdr_alpha,
                method=fdr_method,
            )

    if shrinkage is not None:
        n_kept = probed.size - len(dropped)
        target = closure_null(n_kept) if normalize_per_image and shrinkage == "schafer_strimmer" else 0.0
        shrunk = shrink(primary, shrinkage, target)
        result.R_shrunk = _expand(shrunk.r, probed, n_ops)
        result.shrinkage = {"method": shrinkage, "intensity": shrunk.intensity, "target": shrunk.target}

    if residual_rank:
        r_res, dropped_res, slopes = residual_rank_corr(g)
        result.R_residual = _expand(r_res, probed, n_ops)
        result.residual_slopes = np.full(n_ops, np.nan)
        result.residual_slopes[probed] = slopes
        result.dropped = [names[probed[i]] for i in dropped_res]

    result.diagnostics = diagnose(result.primary, names, n_images=n_images)
    return result


def analyze_sweep(sweep, **kwargs) -> Analysis:
    """``analyze`` on a ``probe.Sweep`` (duck-typed: stats does not import the probe),
    with the sweep's provenance as the labels every log record carries, and its loss
    increase as ``d_loss`` when it has one."""
    d_grad, probe_ids = sweep.matrix()
    if "d_loss" not in kwargs and hasattr(sweep, "loss_matrix"):
        kwargs["d_loss"] = sweep.loss_matrix()
    labels = {
        "checkpoint": sweep.checkpoint,
        "probe_set": sweep.probe_set,
        "probe_seed": sweep.probe_seed,
        "magnitude_source": sweep.magnitude_source,
        "magnitude_mode": sweep.magnitude_mode,
        "magnitudes": sweep.magnitude_summary(),
        **sweep.meta,
        **kwargs.pop("labels", {}),
    }
    return analyze(d_grad, probe_ids, sweep.names, labels=labels, **kwargs)


def _expand(matrix: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
    """A (P, P) matrix over ``rows`` scattered into (n, n), NaN elsewhere."""
    full = np.full((n, n), np.nan)
    full[np.ix_(rows, rows)] = matrix
    return full
