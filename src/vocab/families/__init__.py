"""The op families. Each module declares `FAMILY` (its name) and `OPS` (a tuple
of OpSpecs, in the order they become axes of R)."""

from typing import Dict, Tuple

from ..op import OpSpec
from . import autoaugment, geometric, legacy_hsv, pil_autoaug, randaug_default, trivialaug_wide

FAMILIES: Dict[str, Tuple[OpSpec, ...]] = {
    module.FAMILY: module.OPS
    for module in (legacy_hsv, geometric, pil_autoaug, randaug_default, trivialaug_wide, autoaugment)
}

__all__ = ["FAMILIES"]
