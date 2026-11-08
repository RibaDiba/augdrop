"""The op families. Each module declares `FAMILY` (its name) and `OPS` (a tuple
of OpSpecs, in the order they become axes of R). The torchvision-backed ones, and the
differentiable `trivialaug_diff`, `randaug_diff`, `augmix_diff` and `smartaug_diff`, also
declare `BINS`, the magnitude bins their method draws from (`randaug_default.BinTable`).
AutoAugment's Invert has no magnitude, and AutoAugment's other ops are
`randaug_default`'s, whose table torchvision's AutoAugment shares. `madaug` holds the
two MADAug ops no other family has (Cutout, Flip); its other 14 are `randaug_default`'s
and `autoaugment`'s. `faster_aa` is Faster AutoAugment's 17 ops (its 9 signed ones split
into _pos/_neg), whose magnitudes come from a searched policy, not bins. `deepaa` is
Deep AutoAugment's ImageNet search space and `advaa` Adversarial AutoAugment's 16 ops
(AutoAugment's Table 6), both PIL's ops with their signed ones split; their magnitudes
come from their hosts' policies, not bins. `greedy_aa` is Greedy AutoAugment's 20 ops (the TF
AutoAugment CIFAR code's, at its 32 px ranges as fractions of the image), on `deepaa`'s
bodies, its signed ops split; its host maps a level to each op's magnitude. `adaaug` is
AdaAug's 16 ops (Fast AutoAugment's, at its ranges), on `deepaa`'s bodies and `madaug`'s
Cutout and Flip, its signed ops split; its host maps the policy's level to each op."""

from typing import Callable, Dict, Tuple

from ..op import OpSpec
from . import (
    adaaug,
    advaa,
    augmix_diff,
    autoaugment,
    deepaa,
    faster_aa,
    geometric,
    greedy_aa,
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
        deepaa,
        advaa,
        greedy_aa,
        adaaug,
    )
}

#: op name -> its BinTable, for the ops whose family has one.
BINS: Dict[str, Callable] = {
    name: table
    for module in (randaug_default, trivialaug_wide, trivialaug_diff, randaug_diff, augmix_diff, smartaug_diff)
    for name, table in module.BINS.items()
}

__all__ = ["BINS", "FAMILIES"]
