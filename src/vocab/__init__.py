"""What an augmentation is: the op vocabulary the gradient probe measures.

    OpSpec                    one op: metadata + `apply(images, magnitude, label)`
    OPS, get_op               every op in every family, by name
    magnitude_bins(name, n)   the bins a torchvision method draws the op's magnitude from
    build_vocab(config)       an ordered {name: OpSpec} selection (the axes of R)
    FAMILIES                  family name -> its ops
    audit(vocab)              run every vocab check, get an AuditReport
"""

from .audit import AuditReport, CheckResult, audit
from .families import FAMILIES
from .op import GRAD_MODES, KINDS, OpSpec
from .registry import DEFAULT_FAMILIES, OPS, build_vocab, get_op, magnitude_bins

__all__ = [
    "AuditReport",
    "CheckResult",
    "DEFAULT_FAMILIES",
    "FAMILIES",
    "GRAD_MODES",
    "KINDS",
    "OPS",
    "OpSpec",
    "audit",
    "build_vocab",
    "get_op",
    "magnitude_bins",
]
