"""Write G, R, q and eval artifacts (see ``schema``), each with its header.

Every writer takes the vocab its object is over, as a ``{name: OpSpec}`` or as op
names looked up in the registry, and refuses one whose names or order differ from the
object's: the hash has to describe the axes it labels. Files are written to a temp
file and moved into place, so a crash never leaves half an artifact.

These are dropaug's own artifacts. Sensaug's log files (``corr_matrix_log.json`` and
the rest) are still ``stats.logs``, for the offline tools that read them.
"""

import json
import os
from datetime import datetime, timezone
from typing import Iterable, Mapping, Optional

import numpy as np

from stats.logs import jsonable, write_json_atomic

from . import schema
from .hashing import file_hash, probe_set_hash, vocab_hash

__all__ = ["write_G", "write_R", "write_eval", "write_q"]


def _names(vocab) -> list:
    return [str(n) for n in (vocab.keys() if isinstance(vocab, Mapping) else vocab)]


def _check_axes(what: str, names, vocab) -> list:
    expected = _names(vocab)
    if list(names) != expected:
        raise ValueError(
            f"the {what}'s ops {list(names)[:4]}... are not the vocab's {expected[:4]}... in "
            f"the same order; the vocab hash would label the wrong axes"
        )
    return expected


def _created() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_G(path, sweep, vocab) -> dict:
    """One ``probe.Sweep`` as .npz. Returns the header."""
    names = _check_axes("sweep", sweep.names, vocab)
    d_grad, batch_ids = sweep.matrix()
    magnitudes = np.stack(
        [np.concatenate(sweep.magnitudes[n]) if sweep.magnitudes[n] else np.empty(0) for n in names]
    )
    head = schema.header(
        "G",
        names=names,
        vocab_hash=vocab_hash(vocab),
        probe_set_hash=sweep.probe_set,
        image_ids=sweep.ids(),
        checkpoint=sweep.checkpoint,
        probe_seed=sweep.probe_seed,
        magnitude_source=sweep.magnitude_source,
        magnitude_mode=sweep.magnitude_mode,
        methods=sweep.methods,
        meta=sweep.meta,
        failed_batches=sweep.failed_batches,
        created=_created(),
    )
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            G=d_grad,
            magnitudes=magnitudes,
            batch_ids=batch_ids,
            header=np.array(json.dumps(jsonable(head), allow_nan=False)),
        )
    os.replace(tmp, path)
    return head


def _bootstrap(boot) -> Optional[dict]:
    if boot is None:
        return None
    return {
        "names": boot.names,
        "r": boot.r,
        "ci_lo": boot.ci_lo,
        "ci_hi": boot.ci_hi,
        "p": boot.p,
        "q": boot.q,
        "survives": boot.survives,
        "null_r": boot.null_r,
        "n_reps": boot.n_reps,
        "seed": boot.seed,
        "alpha": boot.alpha,
        "method": boot.method,
    }


def write_R(path, analysis, vocab) -> dict:
    """One ``stats.Analysis`` as .json. The probe set is the one its labels name
    (``analyze_sweep`` puts it there). Returns what was written."""
    names = _check_axes("analysis", analysis.names, vocab)
    probe_set = analysis.labels.get("probe_set")
    if probe_set is None:
        raise ValueError(
            "the analysis names no probe set (labels['probe_set']); build it with "
            "stats.analyze_sweep, or pass labels={'probe_set': probe_set.digest}"
        )
    record = schema.header(
        "R",
        names=names,
        vocab_hash=vocab_hash(vocab),
        probe_set_hash=probe_set_hash(probe_set),
        n_images=analysis.n_images,
        n_probes=analysis.n_probes,
        normalize_per_image=analysis.normalize_per_image,
        skipped=analysis.skipped,
        R_raw=analysis.R_raw,
        R_scalenorm=analysis.R_scalenorm,
        R_shrunk=analysis.R_shrunk,
        shrinkage=analysis.shrinkage,
        dropped=analysis.dropped,
        unprobed=analysis.unprobed,
        static_pruned=analysis.static_pruned,
        aliases=analysis.aliases,
        shared_factor_loadings=analysis.shared_factor_loadings,
        bootstrap=_bootstrap(analysis.bootstrap),
        diagnostics=None if analysis.diagnostics is None else analysis.diagnostics.as_dict(),
        withheld=analysis.withheld,
        labels=analysis.labels,
        created=_created(),
    )
    record = jsonable(record)
    write_json_atomic(path, record)
    return record


def write_q(
    path,
    selection,
    vocab,
    *,
    selector: str,
    lam: Optional[float] = None,
    derived_from=None,
    labels: Optional[Mapping] = None,
) -> dict:
    """One ``selection.Selection`` as .json. ``derived_from`` is the path of the R
    artifact it was selected on; its sha256 is stored, so ``read_q`` can refuse a q
    whose R has since been rewritten. Returns what was written."""
    names = _check_axes("selection", selection.names, vocab)
    record = schema.header(
        "q",
        names=names,
        vocab_hash=vocab_hash(vocab),
        q=np.asarray(selection.q),
        applied=bool(selection.applied),
        reason=selection.reason,
        kept=None if selection.kept is None else list(selection.kept),
        exempt=list(selection.exempt),
        pruned=list(selection.pruned),
        selector=selector,
        lam=lam,
        derived_from=None
        if derived_from is None
        else {"path": os.fspath(derived_from), "sha256": file_hash(derived_from)},
        labels=dict(labels or {}),
        created=_created(),
    )
    record = jsonable(record)
    write_json_atomic(path, record)
    return record


def write_eval(path, records: Iterable[Mapping], vocab=None, *, labels: Optional[Mapping] = None) -> dict:
    """One checkpoint's ``dropaug.eval.run`` records as .json.

    ``vocab`` is what the op conditions were built from; by default, the ops the
    records name, in first-seen order. Every op condition must be in it. One file is
    one checkpoint, as for ``run``'s results file.
    """
    records = [dict(r) for r in records]
    checkpoints = {json.dumps(r.get("checkpoint"), sort_keys=True) for r in records}
    if len(checkpoints) > 1:
        raise ValueError(f"records from {len(checkpoints)} checkpoints; one eval artifact is one checkpoint")
    op_names = list(dict.fromkeys(r["condition"] for r in records if r.get("kind") == "op"))
    if vocab is None:
        vocab = op_names
    names = _names(vocab)
    stray = sorted(set(op_names) - set(names))
    if stray:
        raise ValueError(f"records evaluate ops {stray} that are not in the vocab")
    record = schema.header(
        "eval",
        names=names,
        vocab_hash=vocab_hash(vocab),
        checkpoint=records[0].get("checkpoint") if records else None,
        records=records,
        labels=dict(labels or {}),
        created=_created(),
    )
    record = jsonable(record)
    write_json_atomic(path, record)
    return record
