"""Backends: one folder per training framework.

    base.py      TaskBackend (a probe.ModelAdapter that can also build and load its
                 model and name its clean probe data) and ProbeData
    mmseg/       MMSegmentation (SegFormer and every other segmentor): the backend,
                 the two training hooks, and the GPU train-time augmentation
    cls/         image classifiers (timm, torchvision, any nn.Module): the backend,
                 clean ImageFolder / CIFAR readers, and the two hooks as callbacks
                 for a hand-written loop
    augment.py   training-time draws from a host, shared by both backends
    magnitudes.py  the probe's magnitudes for a host (each method's own intensity),
                 shared by both backends and scripts/probe.py

A backend package imports its framework lazily, so ``import tasks`` needs none of
them. ``build_backend(name, **kwargs)`` imports the one asked for.
"""

import importlib

from .base import ProbeData, TaskBackend

__all__ = ["BACKENDS", "ProbeData", "TaskBackend", "build_backend"]

#: name -> (module, class), imported on first use.
BACKENDS = {
    "mmseg": ("tasks.mmseg.backend", "MMSegBackend"),
    "cls": ("tasks.cls.backend", "ClsBackend"),
}


def build_backend(name: str, **kwargs) -> TaskBackend:
    """Construct the backend registered under ``name`` with its keyword arguments."""
    try:
        module, cls = BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; expected one of {sorted(BACKENDS)}") from None
    return getattr(importlib.import_module(module), cls)(**kwargs)
