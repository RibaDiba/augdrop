"""Augmentation-method adapters. train.py picks one by name:

    host = build_host(args.method, probe_interval=2000)
    host = build_host("uniform", pruned=["blur", "noise"])   # a static prune (prune.py)
"""

from typing import Iterable

from .autoaugment import AutoAugmentHost
from .base import AugHost
from .prune import PRUNE_FILE, Prune, draw_random_prune, resolve_prune
from .randaugment import RandAugmentHost
from .sensaug import SensAugHost
from .trivialaugment import TrivialAugmentDiffHost, TrivialAugmentHost
from .uniform import UniformHost

__all__ = [
    "AugHost",
    "AutoAugmentHost",
    "RandAugmentHost",
    "SensAugHost",
    "TrivialAugmentDiffHost",
    "TrivialAugmentHost",
    "UniformHost",
    "HOSTS",
    "PRUNE_FILE",
    "Prune",
    "build_host",
    "draw_random_prune",
    "resolve_prune",
]

HOSTS = {
    cls.name: cls
    for cls in (
        SensAugHost,
        RandAugmentHost,
        TrivialAugmentHost,
        TrivialAugmentDiffHost,
        AutoAugmentHost,
        UniformHost,
    )
}


def build_host(name: str, *, pruned: Iterable[str] = (), hold_none_prob: bool = True, **kwargs) -> AugHost:
    """Construct the host registered under ``name`` with its keyword arguments, then
    apply the static prune ``pruned`` (``AugHost.prune``)."""
    try:
        cls = HOSTS[name]
    except KeyError:
        raise ValueError(f"unknown host {name!r}; expected one of {sorted(HOSTS)}") from None
    host = cls(**kwargs)
    host.prune(pruned, hold_none_prob=hold_none_prob)
    return host
