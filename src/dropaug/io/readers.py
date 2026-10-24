"""Read G, R, q and eval artifacts back, refusing any that don't match.

Every reader validates the header (``schema.validate``: format, schema_version, kind,
required keys) and then checks hashes:

    vocab            the ops you expect the file to be over (``{name: OpSpec}`` or
                     names). Refused unless its ``vocab_hash`` matches, so a G over
                     other ops, or the same ops in another order, never loads as yours.
    check_registry   (default True) also refuse a file whose ops the registry now
                     defines differently (another range, family or grad mode) than
                     when it was written. Its numbers were measured in the old units.
    probe_set        G and R only: a ``ProbeSet``, its digest, or its ids. Refused
                     unless the file was measured on those images.

Each returns the object it was written from: a ``probe.Sweep``, a ``stats.Analysis``,
a ``selection.Selection``, or the eval records. ``read_header`` returns just the
validated header of any of them.

``sweeps_from_gradient_log`` reads the probe's own ``aug_gradient_log.txt`` (Sensaug's
format, which has no header and no hash) back into Sweeps, one per model state.
"""

import json
from collections import OrderedDict
from typing import Optional

import numpy as np

from . import schema
from .hashing import file_hash, probe_set_hash, vocab_hash

__all__ = [
    "MismatchError",
    "read_G",
    "read_R",
    "read_eval",
    "read_header",
    "read_q",
    "sweeps_from_gradient_log",
]


class MismatchError(ValueError):
    """An artifact that is valid, but not over the ops, images or R it was expected to be."""


def _load(path):
    """(header, arrays) for any artifact: arrays is the npz for G, else None."""
    if str(path).endswith(".npz"):
        arrays = np.load(path, allow_pickle=False)
        if "header" not in arrays.files:
            raise schema.SchemaError(f"{path} has no header array; not a dropaug G")
        return json.loads(str(arrays["header"])), arrays
    with open(path) as f:
        return json.load(f), None


def read_header(path, kind: Optional[str] = None) -> dict:
    return schema.validate(_load(path)[0], kind)


def _check(head: dict, path, vocab, check_registry: bool, probe_set=None) -> None:
    stored = head["vocab_hash"]
    if vocab is not None and vocab_hash(vocab) != stored:
        raise MismatchError(
            f"{path} is over ops {head['names'][:4]}... (vocab {stored[:12]}), not the vocab "
            f"given ({vocab_hash(vocab)[:12]}). Different ops, order or definitions."
        )
    if check_registry:
        try:
            now = vocab_hash(head["names"])
        except KeyError as error:
            raise MismatchError(f"{path} names an op the registry no longer has: {error}") from None
        if now != stored:
            raise MismatchError(
                f"the registry now defines {path}'s ops differently than when it was written "
                f"(vocab {stored[:12]} then, {now[:12]} now): its magnitudes are in the old "
                f"units. check_registry=False reads it anyway."
            )
    if probe_set is not None and probe_set_hash(probe_set) != head.get("probe_set_hash"):
        raise MismatchError(
            f"{path} was measured on probe set {str(head.get('probe_set_hash'))[:12]}, not "
            f"{probe_set_hash(probe_set)[:12]}"
        )


def _array(value) -> Optional[np.ndarray]:
    """JSON back to a float array: null -> NaN (``stats.jsonable`` wrote NaN as null)."""
    return None if value is None else np.array(value, dtype=np.float64)


def read_G(path, *, vocab=None, probe_set=None, check_registry: bool = True):
    """A G artifact as a ``probe.Sweep``, its batches as they were probed."""
    from probe import Sweep

    head, arrays = _load(path)
    head = schema.validate(head, "G")
    _check(head, path, vocab, check_registry, probe_set)
    d_grad, magnitudes, batch_ids = arrays["G"], arrays["magnitudes"], arrays["batch_ids"]
    names, ids = head["names"], head["image_ids"]
    if d_grad.shape != (len(names), len(ids)) or magnitudes.shape != d_grad.shape or batch_ids.shape != (len(ids),):
        raise schema.SchemaError(
            f"{path}: G {d_grad.shape}, magnitudes {magnitudes.shape} and batch_ids "
            f"{batch_ids.shape} don't fit {len(names)} ops x {len(ids)} images"
        )
    batches = [np.flatnonzero(batch_ids == k) for k in np.unique(batch_ids)]
    return Sweep(
        names=list(names),
        checkpoint=head["checkpoint"],
        probe_set=head["probe_set_hash"],
        magnitude_source=head["magnitude_source"],
        magnitude_mode=head["magnitude_mode"],
        probe_seed=head["probe_seed"],
        methods=dict(head["methods"]),
        meta=dict(head["meta"]),
        grads={n: [d_grad[a, cols] for cols in batches] for a, n in enumerate(names)},
        magnitudes={n: [magnitudes[a, cols] for cols in batches] for a, n in enumerate(names)},
        image_ids=[tuple(ids[j] for j in cols) for cols in batches],
        failed_batches=list(head["failed_batches"]),
    )


def read_R(path, *, vocab=None, probe_set=None, check_registry: bool = True):
    """An R artifact as a ``stats.Analysis``; its diagnostics are recomputed."""
    from stats import Analysis, Bootstrap, diagnose

    head = read_header(path, "R")
    _check(head, path, vocab, check_registry, probe_set)
    boot = head.get("bootstrap")
    if boot is not None:
        boot = Bootstrap(
            names=list(boot["names"]),
            r=_array(boot["r"]),
            ci_lo=_array(boot["ci_lo"]),
            ci_hi=_array(boot["ci_hi"]),
            p=_array(boot["p"]),
            q=_array(boot["q"]),
            survives=np.array(boot["survives"], dtype=bool),
            null_r=boot["null_r"],
            n_reps=boot["n_reps"],
            seed=boot["seed"],
            alpha=boot["alpha"],
            method=boot.get("method", "bh"),
        )
    analysis = Analysis(
        names=list(head["names"]),
        n_images=head["n_images"],
        n_probes=head["n_probes"],
        normalize_per_image=head["normalize_per_image"],
        skipped=head["skipped"],
        R_raw=_array(head["R_raw"]),
        R_scalenorm=_array(head["R_scalenorm"]),
        dropped=list(head["dropped"]),
        unprobed=list(head["unprobed"]),
        static_pruned=list(head.get("static_pruned", ())),  # optional: files before the prune lack it
        aliases=dict(head["aliases"]),
        shared_factor_loadings=_array(head.get("shared_factor_loadings")),
        bootstrap=boot,
        R_shrunk=_array(head.get("R_shrunk")),
        shrinkage=head.get("shrinkage"),
        labels=dict(head["labels"]),
    )
    if analysis.primary is not None:
        analysis.diagnostics = diagnose(analysis.primary, analysis.names, n_images=analysis.n_images)
    return analysis


def read_q(path, *, vocab=None, derived_from=None, check_registry: bool = True):
    """A q artifact as a ``selection.Selection``. With ``derived_from`` (an R
    artifact's path), refused unless the q was selected on that file as it is now."""
    from selection import Selection

    head = read_header(path, "q")
    _check(head, path, vocab, check_registry)
    if derived_from is not None:
        source = head.get("derived_from")
        if source is None or source["sha256"] != file_hash(derived_from):
            raise MismatchError(
                f"{path} was not selected on {derived_from} as it is now "
                f"(recorded: {source['path'] if source else 'no source'})"
            )
    return Selection(
        names=tuple(head["names"]),
        q=np.array(head["q"], dtype=np.float64),
        applied=head["applied"],
        reason=head["reason"],
        kept=None if head.get("kept") is None else tuple(head["kept"]),
        exempt=tuple(head.get("exempt", ())),
        pruned=tuple(head.get("pruned", ())),
    )


def read_eval(path, *, vocab=None, checkpoint=None, check_registry: bool = True) -> list:
    """An eval artifact's records. With ``checkpoint``, refused unless they are that
    checkpoint's."""
    head = read_header(path, "eval")
    _check(head, path, vocab, check_registry)
    if checkpoint is not None and head["checkpoint"] != checkpoint:
        raise MismatchError(f"{path} scores checkpoint {head['checkpoint']!r}, not {checkpoint!r}")
    return list(head["records"])


def sweeps_from_gradient_log(path) -> list:
    """``aug_gradient_log.txt`` back into ``probe.Sweep``s, one per (checkpoint, meta,
    probe set, magnitude setting), in the order they were written.

    The log holds every batch the probe kept, so G is exactly recomputed, and so are the
    losses behind ``Sweep.loss_matrix()`` when the log has them. ``failed_batches`` is not in the log and comes back empty. Raises on a log whose
    batches of one sweep disagree on the ops.
    """
    from probe import Sweep

    sweeps: "OrderedDict[str, Sweep]" = OrderedDict()
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = json.dumps(
                [
                    record["checkpoint"],
                    record["meta"],
                    record["probe_set"],
                    record["magnitude_source"],
                    record["magnitude_mode"],
                    record["probe_seed"],
                ],
                sort_keys=True,
            )
            names = list(record["grads"])
            sweep = sweeps.get(key)
            if sweep is None:
                sweep = sweeps[key] = Sweep(
                    names=names,
                    checkpoint=record["checkpoint"],
                    probe_set=record["probe_set"],
                    magnitude_source=record["magnitude_source"],
                    magnitude_mode=record["magnitude_mode"],
                    probe_seed=record["probe_seed"],
                    methods=dict(record["methods"]),
                    meta=dict(record["meta"]),
                )
            elif names != sweep.names:
                raise schema.SchemaError(f"{path}:{line_no}: this batch's ops differ from its sweep's")
            # A log written before the probe recorded losses has none: NaN.
            losses = record.get("losses") or {}
            missing = [None] * len(record["image_ids"])
            for name in names:
                sweep.grads[name].append(_array(record["grads"][name]))
                sweep.magnitudes[name].append(_array(record["magnitudes"][name]))
                sweep.losses[name].append(_array(losses.get(name, missing)))
            sweep.clean_losses.append(_array(record.get("clean_loss", missing)))
            sweep.image_ids.append(tuple(record["image_ids"]))
    return list(sweeps.values())
