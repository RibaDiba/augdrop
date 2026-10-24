"""The files an emission writes, in Sensaug's shapes so its offline tools
(``scripts/calibrate_lambda.py``, ``corr_viz``) read dropaug output unchanged.

Ported from sensaug/hooks/grad_sens_analysis.py (``_jsonable``, ``_write_json_atomic``,
and the records ``_emit``, ``_emit_bootstrap`` and ``prune_augmentations`` built).

    corr_matrix_log.json      one JSON array, one record per emission: R_raw,
                              R_scalenorm, names, loadings, magnitudes, dropped, role
    corr_bootstrap_log.txt    JSONL, per-cell CIs, p, BH-FDR q and survivors
    corr_redundancy_log.txt   JSONL, per-op red(a) and what was done with it

``aug_gradient_log.txt``, the per-batch G, is the probe's (``frozen_sweep.sweep``).
Sensaug's ``stale`` and ``static_pruned`` fields are written as empty lists: dropaug
has neither mechanism, and the keys stay so readers keyed on them don't break.
``unprobed`` (ops with no magnitude, NaN rows), ``aliases``, ``R_shrunk`` /
``shrinkage`` (null unless asked for), ``R_residual`` / ``residual_slopes`` (null
unless asked for), ``loss_increase``, ``mean_offdiag``, ``diagnostics`` and the
bootstrap's ``fdr_method`` are dropaug's additions.
"""

import json
import math
import os
from typing import Optional

import numpy as np

from .analyze import Analysis
from .redundancy import RedundancyScore

__all__ = [
    "BOOTSTRAP_LOG",
    "CORR_MATRIX_LOG",
    "REDUNDANCY_LOG",
    "append_json_array",
    "append_jsonl",
    "bootstrap_record",
    "corr_matrix_record",
    "jsonable",
    "redundancy_record",
    "reset_logs",
    "write_analysis",
    "write_json_atomic",
]

CORR_MATRIX_LOG = "corr_matrix_log.json"
BOOTSTRAP_LOG = "corr_bootstrap_log.txt"
REDUNDANCY_LOG = "corr_redundancy_log.txt"


def jsonable(value):
    """numpy -> python, and non-finite -> None.

    json.dumps writes a bare ``NaN`` token for float('nan'), which is not valid JSON:
    python reads it back, jq and pandas do not. R deliberately contains NaN (dropped
    ops), so every record goes through this on the way out.
    """
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_json_atomic(path, obj) -> None:
    """Rewrite ``path`` as one JSON document via a temp file + os.replace, so a crash
    mid-write leaves the previous file intact. allow_nan=False makes a value that
    skipped ``jsonable`` raise instead of writing a bare ``NaN``."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def append_json_array(path, record: dict) -> None:
    """Append one record to a file holding one JSON array (read-modify-write, so the
    file stays a single document across a run and across a resume)."""
    records = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            records = json.load(f)
    records.append(jsonable(record))
    write_json_atomic(path, records)


def append_jsonl(path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(jsonable(record), allow_nan=False) + "\n")


def reset_logs(work_dir) -> None:
    """Empty the three emission logs for a fresh (not resumed) run, so a new run in
    an old work_dir does not append to the previous run's history."""
    for name in (CORR_MATRIX_LOG, BOOTSTRAP_LOG, REDUNDANCY_LOG):
        path = os.path.join(work_dir, name)
        if os.path.exists(path):
            os.remove(path)


# --- records --------------------------------------------------------------------------


def corr_matrix_record(analysis: Analysis) -> dict:
    """One corr_matrix_log.json record: Sensaug's fields, plus ``unprobed``,
    ``aliases``, the shrunk and residual R, the loss increase, the diagnostics and the
    sweep's labels (``probe_set`` digest, ``probe_seed``)."""
    labels = analysis.labels
    return {
        "checkpoint": labels.get("checkpoint"),
        "iter": labels.get("iter"),
        "role": labels.get("role", "active"),
        "n_images": analysis.n_images,
        "n_probes": analysis.n_probes,
        "names": analysis.names,
        "R_raw": analysis.R_raw,
        "R_scalenorm": analysis.R_scalenorm,
        "dropped": analysis.dropped,
        "stale": [],
        "static_pruned": [],
        "unprobed": analysis.unprobed,
        "aliases": analysis.aliases,
        "R_shrunk": analysis.R_shrunk,
        "shrinkage": analysis.shrinkage,
        "R_residual": analysis.R_residual,
        "residual_slopes": analysis.residual_slopes,
        "loss_increase": analysis.loss_increase,
        "mean_offdiag": analysis.mean_offdiag,
        "diagnostics": None if analysis.diagnostics is None else analysis.diagnostics.as_dict(),
        "shared_factor_loadings": analysis.shared_factor_loadings,
        "magnitude_source": labels.get("magnitude_source"),
        "magnitude_mode": labels.get("magnitude_mode"),
        "magnitudes": labels.get("magnitudes"),
        "probe_set": labels.get("probe_set"),
        "probe_seed": labels.get("probe_seed"),
    }


def bootstrap_record(analysis: Analysis) -> Optional[dict]:
    """One corr_bootstrap_log.txt record, or None when no bootstrap ran."""
    boot = analysis.bootstrap
    if boot is None:
        return None
    cells = []
    for a in range(len(boot.names)):
        for b in range(a + 1, len(boot.names)):
            cells.append(
                {
                    "i": boot.names[a],
                    "j": boot.names[b],
                    "r": boot.r[a, b],
                    "ci_lo": boot.ci_lo[a, b],
                    "ci_hi": boot.ci_hi[a, b],
                    "ci_width": boot.ci_hi[a, b] - boot.ci_lo[a, b],
                    "p": boot.p[a, b],
                    "q": boot.q[a, b],
                    "survives_fdr": boot.survives[a, b],
                }
            )
    return {
        "checkpoint": analysis.labels.get("checkpoint"),
        "iter": analysis.labels.get("iter"),
        "n_reps": boot.n_reps,
        "seed": boot.seed,
        # Sensaug's value. A "probe" there is a batch here; the cluster is the same.
        "cluster_level": "probe",
        "matrix": "R_scalenorm" if analysis.normalize_per_image else "R_raw",
        "null_r": boot.null_r,
        "fdr_method": boot.method,
        "cells": cells,
    }


def redundancy_record(score: RedundancyScore, analysis: Analysis, mask_within_op: bool, **fields) -> dict:
    """One corr_redundancy_log.txt record: Sensaug's fields (without ``r`` and
    ``survives``, which the other two logs hold under the same ``iter``), plus
    whatever the caller adds, e.g. the selector's outcome."""
    return {
        "iter": analysis.labels.get("iter"),
        "checkpoint": analysis.labels.get("checkpoint"),
        "mode": score.mode,
        "fdr_gated": analysis.bootstrap is not None,
        "red": score.as_dict(),
        "raw": {n: float(v) for n, v in zip(score.names, score.raw)},
        "dropped": score.dropped,
        "names": analysis.names,
        "mask_within_op": bool(mask_within_op),
        **fields,
    }


def write_analysis(work_dir, analysis: Analysis) -> None:
    """Append ``analysis`` to corr_matrix_log.json and corr_bootstrap_log.txt. A
    skipped analysis writes nothing, as in Sensaug."""
    if analysis.skipped is not None:
        return
    append_json_array(os.path.join(work_dir, CORR_MATRIX_LOG), corr_matrix_record(analysis))
    record = bootstrap_record(analysis)
    if record is not None:
        append_jsonl(os.path.join(work_dir, BOOTSTRAP_LOG), record)
