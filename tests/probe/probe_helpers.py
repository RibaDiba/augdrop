"""Shared test helpers: a toy model adapter and toy ops standing in for a real model
and vocab/. A plain module rather than conftest.py so tests can import from it.

The ops mirror vocab's ``OpSpec`` contract (``apply(images, magnitude, label)``,
``mag_range``, ``identity_at``, ``grad_mode``, ``warps_label``) without importing
vocab, whose kornia/torchvision dependencies the probe doesn't need.

BatchNorm is the point of the toy model. It is the module whose train/eval mode
decides whether the images in a batch are coupled, and the per-image measurement
lives or dies on that.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

IGNORE = 255
N_CLASSES = 3


@dataclass(frozen=True)
class Op:
    name: str
    fn: Callable
    kind: str = "photometric"
    mag_range: Optional[Tuple[float, float]] = (0.0, 1.0)
    identity_at: Optional[float] = 0.0
    grad_mode: str = "autograd"

    @property
    def warps_label(self):
        return self.kind == "geometric"

    def apply(self, images, magnitude=None, label=None):
        if self.kind == "geometric":
            return self.fn(images, magnitude, label)
        return self.fn(images, magnitude), label


class ToyAdapter:
    """conv + BatchNorm + 1x1 classifier, per-image cross-entropy over valid pixels."""

    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, N_CLASSES, 1),
        )
        # Non-trivial running statistics, so eval() BN is not the identity.
        self.model[1].running_mean.uniform_(-0.2, 0.2)
        self.model[1].running_var.uniform_(0.5, 1.5)
        self.model.eval()
        self.seen_labels = []

    def to_unit(self, batch):
        return batch["image"], batch["label"]

    def loss(self, images01, label, batch):
        self.seen_labels.append(label)
        ce = F.cross_entropy(self.model(images01), label, ignore_index=IGNORE, reduction="none")
        valid = (label != IGNORE).to(ce.dtype)
        return (ce * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)

    def batch_ids(self, batch):
        return list(batch["id"])


class ScalarLossAdapter(ToyAdapter):
    """A batch-mean loss, like mmseg's ``model.loss``."""

    def loss(self, images01, label, batch):
        return super().loss(images01, label, batch).mean()


# --- ops ------------------------------------------------------------------------


def _m(magnitude, images):
    return torch.as_tensor(magnitude, dtype=images.dtype, device=images.device).reshape(-1, 1, 1)


def lighter_r(images, magnitude):
    m = _m(magnitude, images)
    red = images[:, 0] + m * (1 - images[:, 0])
    return torch.cat([red[:, None], images[:, 1:]], dim=1)


def darker_g(images, magnitude):
    m = _m(magnitude, images)
    green = images[:, 1] * (1 - 0.8 * m)
    return torch.cat([images[:, :1], green[:, None], images[:, 2:]], dim=1)


def noise(images, magnitude):
    return (images + _m(magnitude, images)[:, None] * 0.2 * torch.randn_like(images)).clamp(0, 1)


def pooled(images, magnitude):
    """The Sensaug bug: pools the per-image magnitude, so every image sees them all."""
    return images * (1 - 0.5 * torch.as_tensor(magnitude).mean())


def enhance(images, factor):
    """Own units: a PIL-style enhance factor in [0.1, 1.9], identity at 1."""
    return (images * _m(factor, images)[:, None]).clamp(0, 1)


def _shift(x, pixels, mode):
    """Shift right by ``pixels`` (per image), via affine_grid."""
    b, _, h, w = x.shape
    theta = torch.zeros(b, 2, 3, dtype=x.dtype, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2 * pixels.reshape(-1) / w
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


def translate_x(images, magnitude, label=None):
    pixels = torch.as_tensor(magnitude, dtype=images.dtype, device=images.device) * 0.25 * images.shape[-1]
    images_out = _shift(images, pixels, "bilinear")
    if label is None:
        return images_out, None
    # Shift label + 1 with zero padding, so out-of-frame reads 0 and becomes IGNORE.
    shifted = _shift((label[:, None].to(images.dtype) + 1), pixels.detach(), "nearest")[:, 0]
    return images_out, torch.where(shifted == 0, IGNORE, shifted.round().long() - 1)


def unwarped_translate_x(images, magnitude, label=None):
    """The Sensaug bug: moves the image and leaves the label where it was."""
    return translate_x(images, magnitude)[0], label


def no_grad_path(fn):
    """What a torchvision-backed op looks like to autograd: no path to the magnitude."""
    return lambda images, magnitude: fn(images, torch.as_tensor(magnitude).detach())


def make_ops():
    return {
        "lighter_R": Op("lighter_R", lighter_r),
        "darker_G": Op("darker_G", darker_g),
        "noise": Op("noise", noise),
        "translate_x": Op("translate_x", translate_x, kind="geometric"),
    }


# --- data -------------------------------------------------------------------------


def make_dataset(n=8, size=12, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "image": torch.rand(3, size, size, generator=generator),
            "label": torch.randint(0, N_CLASSES, (size, size), generator=generator),
            "id": f"img_{i:03d}",
        }
        for i in range(n)
    ]


def make_batch(n=4, size=12, seed=1):
    return torch.utils.data.default_collate(make_dataset(n, size, seed))
