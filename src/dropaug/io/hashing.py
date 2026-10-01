"""The hashes an artifact is checked by.

    vocab_hash      the ops an artifact's axes are, in order, AS DEFINED: name, family,
                    kind, source, mag_range, identity_at, grad_mode, warps_label. Two
                    files agree on it only if every row means the same op in the same
                    units. The implementation (``fn``) is not hashed, so an op whose code
                    changed but whose declared metadata did not keeps its hash.
    probe_set_hash  ``probe.ProbeSet.digest``: the sha256 of the probe images' ids, in
                    probe order
    file_hash       sha256 of a file's bytes, for "this q was derived from that R"
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Sequence

__all__ = ["file_hash", "op_fingerprint", "probe_set_hash", "vocab_hash"]


def op_fingerprint(spec) -> list:
    """The part of an ``OpSpec`` that fixes what its row of G measures."""
    return [
        spec.name,
        spec.family,
        spec.kind,
        spec.source,
        None if spec.mag_range is None else [float(v) for v in spec.mag_range],
        None if spec.identity_at is None else float(spec.identity_at),
        spec.grad_mode,
        bool(spec.warps_label),
    ]


def _specs(vocab):
    """``vocab`` as a list of OpSpecs: a ``{name: OpSpec}`` mapping, or op names looked
    up in the current registry."""
    if isinstance(vocab, Mapping):
        specs = list(vocab.values())
        mislabelled = [name for name, spec in vocab.items() if spec.name != name]
        if mislabelled:
            raise ValueError(f"vocab keys {mislabelled} don't match their OpSpec's name")
        return specs
    from vocab import get_op

    return [get_op(str(name)) for name in vocab]


def vocab_hash(vocab) -> str:
    """sha256 over the ops' fingerprints, in order. ``vocab`` is a ``{name: OpSpec}``
    (``vocab.build_vocab``'s) or a sequence of op names, which are looked up in the
    registry as it is now."""
    payload = json.dumps([op_fingerprint(spec) for spec in _specs(vocab)], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def probe_set_hash(probe_set: Any) -> str:
    """The digest of a ``ProbeSet``. A str is taken to be a digest already, and a
    sequence of ids is hashed the way ``ProbeSet`` hashes its ids (in the order
    given, which for a ProbeSet is probe order)."""
    if isinstance(probe_set, str):
        return probe_set
    digest = getattr(probe_set, "digest", None)
    if digest is not None:
        return digest
    if isinstance(probe_set, Sequence):
        from probe.probe_set import _digest

        return _digest([str(i) for i in probe_set])
    raise TypeError(f"can't hash a probe set given as {type(probe_set).__name__}")


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
