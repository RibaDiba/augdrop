"""SAVE and LOAD: dropaug's own artifacts, each with a versioned header and the hashes
that say what it is over.

    schema.py   SCHEMA_VERSION, the header, the keys each kind must hold, validate
    hashing.py  vocab_hash (the ops, as defined), probe_set_hash, file_hash
    writers.py  write_G (.npz) / write_R / write_q / write_eval (.json)
    readers.py  read_G / read_R / read_q / read_eval, which refuse a mismatched schema
                or hash; sweeps_from_gradient_log for the probe's own log

    sweep = probe.sweep(...)
    write_G("G.npz", sweep, ops)
    analysis = stats.analyze_sweep(read_G("G.npz", vocab=ops, probe_set=probe_set))
    write_R("R.json", analysis, ops)

Imported as ``dropaug.io``. A top-level ``io`` would shadow the standard library's,
the reason ``select`` became ``selection``. Sensaug-format logs stay in ``stats.logs``.
"""

from .hashing import file_hash, op_fingerprint, probe_set_hash, vocab_hash
from .readers import (
    MismatchError,
    read_eval,
    read_G,
    read_header,
    read_q,
    read_R,
    sweeps_from_gradient_log,
)
from .schema import FORMAT, KINDS, SCHEMA_VERSION, SchemaError
from .writers import write_eval, write_G, write_q, write_R

__all__ = [
    "FORMAT",
    "KINDS",
    "MismatchError",
    "SCHEMA_VERSION",
    "SchemaError",
    "file_hash",
    "op_fingerprint",
    "probe_set_hash",
    "read_G",
    "read_R",
    "read_eval",
    "read_header",
    "read_q",
    "sweeps_from_gradient_log",
    "vocab_hash",
    "write_G",
    "write_R",
    "write_eval",
    "write_q",
]
