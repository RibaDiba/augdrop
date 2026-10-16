"""Augmentation-method adapters. train.py picks one by name:

    host = build_host(args.method, probe_interval=2000)
"""

from .autoaugment import AutoAugmentHost
from .base import AugHost
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
    "build_host",
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


def build_host(name: str, **kwargs) -> AugHost:
    """Construct the host registered under ``name`` with its keyword arguments."""
    try:
        cls = HOSTS[name]
    except KeyError:
        raise ValueError(f"unknown host {name!r}; expected one of {sorted(HOSTS)}") from None
    return cls(**kwargs)
