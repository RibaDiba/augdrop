"""TaskBackend: one training framework's side of dropaug.

A backend wraps one model and the dataset it is probed on, and speaks the contracts
the framework-free packages define. It IS a ``probe.ModelAdapter``, so
``probe.sweep(backend, ...)`` takes it directly, and with ``predict`` it is also the
adapter ``dropaug.eval`` scores (see ``dropaug/eval/robustness.py``).

    build_model()                -> nn.Module, from the backend's config, in eval()
    load_ckpt(path)              -> the checkpoint's meta; weights go onto .model
    probe_data()                 -> ProbeData: the CLEAN dataset and its stable ids
    to_unit(batch)               -> (images01, label), as the probe defines it
    loss(images01, label, batch) -> (B,) per-image loss
    batch_ids(batch), collate    optional, as the probe defines them
    predict(images01, batch)     optional, for dropaug.eval

``probe_data`` returns the dataset rather than a loader: ``ProbeSet.loader`` builds
the loader itself, never shuffled and never dropping a batch, because G's columns are
labelled by that order. ``loss`` is the name the probe calls.

Training-loop glue (hooks, callbacks, the train-time augmentation) lives next to each
backend and stays glue: it gates and calls ``probe`` / ``stats`` / ``selection`` /
``hosts``, and computes nothing of its own.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Sequence

import torch

__all__ = ["ProbeData", "TaskBackend"]


@dataclass(frozen=True)
class ProbeData:
    """What a ProbeSet is built from and swept over.

    dataset  map-style, yielding CLEAN items (no random transform in its pipeline)
    ids      ``ids[i]`` is the stable id of ``dataset[i]``: a path relative to the
             dataset root, so the same image has the same id on every cluster
    source   a free label for the dataset and split
    """

    dataset: Any
    ids: Sequence[str]
    source: str = ""


class TaskBackend(ABC):
    """Base backend. See the module docstring for the contract."""

    #: The key ``tasks.build_backend`` finds this backend by.
    name: ClassVar[str] = ""

    model: Optional[torch.nn.Module] = None

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """Build the model from the backend's config, put it on the device in eval()
        mode, set ``self.model`` and return it. Weights are left to ``load_ckpt``."""

    @abstractmethod
    def load_ckpt(self, path) -> dict:
        """Load a checkpoint's weights onto ``self.model`` and return its meta."""

    @abstractmethod
    def probe_data(self) -> ProbeData:
        """The clean dataset the probe set is drawn from, with its ids."""

    @abstractmethod
    def to_unit(self, batch) -> tuple:
        """A loader batch as ``(images01, label)``: (B, 3, H, W) float RGB in [0, 1] on
        the model's device, and the label the ops carry along (or None)."""

    @abstractmethod
    def loss(self, images01: torch.Tensor, label: Optional[torch.Tensor], batch) -> torch.Tensor:
        """(B,) per-image loss of the (possibly perturbed) images against ``label``.
        Normalizes internally; image i's loss depends on image i alone."""
