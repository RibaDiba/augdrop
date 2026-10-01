"""ANALYZE: G -> R, its per-cell uncertainty, and a per-op redundancy score.

Ported from Sensaug: the pure-numpy functions of ``sensaug/hooks/grad_sens_analysis.py``
and ``compute_red`` from ``sensaug/redundancy.py``. No torch, and no op names except
``within_op_pairs``, which reads Sensaug's ``lighter_``/``darker_`` and ``_pos``/``_neg``
naming.

    corr.py        correlate(G) -> (R, dropped); shared_image_factor and its
                   loadings; closure_null, the baseline a normalized R is tested at
    dedup.py       find_aliases / merge_aliases: rows of G that are one op under two
                   names, merged before anything else
    shrinkage.py   ledoit_wolf / schafer_strimmer: R pulled toward a target by a
                   data-driven intensity
    fdr.py         cluster_bootstrap_ci (CIs + p-values, resampling whole batches);
                   fdr_correct (Benjamini-Hochberg or Benjamini-Yekutieli over the
                   off-diagonal cells); fisher_z_ci, the analytic reference
    redundancy.py  compute_red: red(a), the standardized row sum of R, reduced as
                   squared / abs / signed
    diagnostics.py diagnose(R): effective rank, participation ratio, condition
                   number, VIF, and the Marchenko-Pastur reference
    analyze.py     analyze / analyze_sweep: Sensaug's ``_emit`` chain on one G ->
                   an ``Analysis`` (R_raw, R_scalenorm, loadings, bootstrap, survivors)
    logs.py        corr_matrix_log.json / corr_bootstrap_log.txt /
                   corr_redundancy_log.txt, in Sensaug's record shapes

``analyze`` chains the functions the way Sensaug's ``_emit`` did::

    G, batch_ids = sweep.matrix()
    R_raw, _ = correlate(G)
    shared = shared_image_factor(G)
    loadings = shared_factor_loadings(G, shared)      # alarm if max |.| > 0.9
    G_norm = G / (shared + 1e-12)
    R_scalenorm, dropped = correlate(G_norm)
    keep = np.setdiff1d(np.arange(len(names)), dropped)
    ci_lo, ci_hi, p = cluster_bootstrap_ci(G_norm[keep], batch_ids, null=closure_null(keep.size))
    q, survives = fdr_correct(p)                      # (K, K): scatter back to (A, A)

with these additions: all-NaN (unprobed) rows and exact aliases are set aside first,
``shrinkage=`` optionally shrinks the primary R, and every R gets its diagnostics (see
analyze.py). Turning the result into q is ``selection``.
"""

from .analyze import Analysis, Bootstrap, analyze, analyze_sweep
from .corr import closure_null, correlate, shared_factor_loadings, shared_image_factor
from .dedup import ALIAS_RTOL, Merged, find_aliases, merge_aliases
from .diagnostics import (
    Diagnostics,
    diagnose,
    effective_rank,
    marchenko_pastur_edges,
    participation_ratio,
    vif,
)
from .fdr import FDR_METHODS, cluster_bootstrap_ci, fdr_correct, fisher_z_ci, fisher_z_se
from .logs import (
    BOOTSTRAP_LOG,
    CORR_MATRIX_LOG,
    REDUNDANCY_LOG,
    append_json_array,
    append_jsonl,
    bootstrap_record,
    corr_matrix_record,
    jsonable,
    redundancy_record,
    reset_logs,
    write_analysis,
)
from .redundancy import MODES, RedundancyScore, compute_red, summarise, within_op_pairs
from .shrinkage import METHODS as SHRINKAGE_METHODS
from .shrinkage import Shrunk, ledoit_wolf, schafer_strimmer, shrink

__all__ = [
    "ALIAS_RTOL",
    "Analysis",
    "BOOTSTRAP_LOG",
    "Bootstrap",
    "CORR_MATRIX_LOG",
    "Diagnostics",
    "FDR_METHODS",
    "MODES",
    "Merged",
    "REDUNDANCY_LOG",
    "RedundancyScore",
    "SHRINKAGE_METHODS",
    "Shrunk",
    "analyze",
    "analyze_sweep",
    "append_json_array",
    "append_jsonl",
    "bootstrap_record",
    "closure_null",
    "cluster_bootstrap_ci",
    "compute_red",
    "corr_matrix_record",
    "correlate",
    "diagnose",
    "effective_rank",
    "fdr_correct",
    "find_aliases",
    "fisher_z_ci",
    "fisher_z_se",
    "jsonable",
    "ledoit_wolf",
    "marchenko_pastur_edges",
    "merge_aliases",
    "participation_ratio",
    "redundancy_record",
    "reset_logs",
    "schafer_strimmer",
    "shared_factor_loadings",
    "shared_image_factor",
    "shrink",
    "summarise",
    "vif",
    "within_op_pairs",
    "write_analysis",
]
