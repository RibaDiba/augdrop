"""The op families. Each module declares `FAMILY` (its name) and `OPS` (a tuple
of OpSpecs, in the order they become axes of R). The torchvision-backed ones, and the
differentiable `trivialaug_diff`, `randaug_diff`, `augmix_diff` and `smartaug_diff`, also
declare `BINS`, the magnitude bins their method draws from (`randaug_default.BinTable`).
AutoAugment's Invert has no magnitude, and AutoAugment's other ops are
`randaug_default`'s, whose table torchvision's AutoAugment shares. `madaug` holds the
two MADAug ops no other family has (Cutout, Flip); its other 14 are `randaug_default`'s
and `autoaugment`'s. `faster_aa` is Faster AutoAugment's 17 ops (its 9 signed ones split
into _pos/_neg), whose magnitudes come from a searched policy, not bins."""

from typing import Callable, Dict, Tuple

from ..op import OpSpec
from . import (
    augmix_diff,
    autoaugment,
    faster_aa,
    geometric,
    legacy_hsv,
    madaug,
    pil_autoaug,
    randaug_default,
    randaug_diff,
    smartaug_diff,
    trivialaug_diff,
    trivialaug_wide,
)

FAMILIES: Dict[str, Tuple[OpSpec, ...]] = {
    module.FAMILY: module.OPS
    for module in (
        legacy_hsv,
        geometric,
        pil_autoaug,
        randaug_default,
        trivialaug_wide,
        autoaugment,
        trivialaug_diff,
        randaug_diff,
        augmix_diff,
        smartaug_diff,
        madaug,
        faster_aa,
    )
}

#: op name -> its BinTable, for the ops whose family has one.
BINS: Dict[str, Callable] = {
    name: table
    for module in (randaug_default, trivialaug_wide, trivialaug_diff, randaug_diff, augmix_diff, smartaug_diff)
    for name, table in module.BINS.items()
}

__all__ = ["BINS", "FAMILIES"]
