"""What a dropaug artifact is on disk: the header every file carries, and the keys each
kind must hold.

    G     one probe sweep (.npz): the (A, N) gradients, their magnitudes and batch ids,
          with the header stored as JSON under ``header``
    R     one ``stats.Analysis`` (.json): R_raw / R_scalenorm / R_shrunk, the bootstrap,
          the loadings and the sweep's labels
    q     one ``selection.Selection`` (.json): the weights a host was, or would have
          been, given
    eval  one checkpoint's ``dropaug.eval`` records (.json)

Every header holds ``format``, ``schema_version``, ``kind``, ``names`` (the axis order
of every array in the file) and ``vocab_hash`` (``hashing.vocab_hash`` of those ops).
G and R also hold ``probe_set_hash``.

A reader refuses any other ``schema_version`` rather than guess at a migration. There
are no migrations yet, and reading a field under the wrong meaning is the failure the
version exists to catch. So bump SCHEMA_VERSION whenever a field's MEANING changes, not
only when a key is added or removed.
"""

from typing import Mapping, Optional

__all__ = ["FORMAT", "KINDS", "REQUIRED", "SCHEMA_VERSION", "SchemaError", "header", "validate"]

FORMAT = "dropaug"
SCHEMA_VERSION = 1

#: Keys every header holds, whatever its kind.
COMMON = ("format", "schema_version", "kind", "names", "vocab_hash")

#: Keys each kind adds.
REQUIRED = {
    "G": (
        "probe_set_hash",
        "image_ids",
        "checkpoint",
        "probe_seed",
        "magnitude_source",
        "magnitude_mode",
        "methods",
        "meta",
        "failed_batches",
    ),
    "R": (
        "probe_set_hash",
        "n_images",
        "n_probes",
        "normalize_per_image",
        "skipped",
        "R_raw",
        "R_scalenorm",
        "dropped",
        "unprobed",
        "aliases",
        "labels",
    ),
    "q": ("q", "applied", "reason", "selector"),
    "eval": ("checkpoint", "records"),
}

KINDS = tuple(REQUIRED)


class SchemaError(ValueError):
    """A file that is not a readable dropaug artifact of the kind asked for."""


def header(kind: str, **fields) -> dict:
    """A header of ``kind`` with ``fields``, validated."""
    record = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "kind": kind, **fields}
    return validate(record, kind)


def validate(record: Mapping, kind: Optional[str] = None) -> dict:
    """``record`` as a dict, or SchemaError saying what is wrong with it."""
    if not isinstance(record, Mapping):
        raise SchemaError(f"a dropaug header is a JSON object, got {type(record).__name__}")
    if record.get("format") != FORMAT:
        raise SchemaError(f"not a dropaug artifact (format={record.get('format')!r})")
    version = record.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version {version!r}, but this dropaug reads {SCHEMA_VERSION}. There is no "
            f"migration; re-derive the file, or read it with the dropaug that wrote it."
        )
    found = record.get("kind")
    if found not in REQUIRED:
        raise SchemaError(f"unknown artifact kind {found!r}; expected one of {KINDS}")
    if kind is not None and found != kind:
        raise SchemaError(f"expected a {kind} artifact, got {found}")
    missing = [key for key in COMMON + REQUIRED[found] if key not in record]
    if missing:
        raise SchemaError(f"{found} artifact is missing {missing}")
    names = record["names"]
    if not isinstance(names, list) or len(set(names)) != len(names):
        raise SchemaError(f"names must be a list of unique op names, got {names!r}")
    return dict(record)
