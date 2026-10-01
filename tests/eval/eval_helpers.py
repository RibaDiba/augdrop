"""Toy models for tests/eval: weightless "oracles" whose right answer is known, so a
score is checkable by hand.

Segmentation: a pixel's class is the nearest of CENTERS to its red value, and the toy
images paint each class's stripe at its center, so clean scores 100 mIoU. Anything that
moves the red channel costs score, and anything that moves pixels costs score unless
the label moves with them.

Classification: the class is the dominant channel.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader

CENTERS = (1 / 6, 1 / 2, 5 / 6)
N_CLASSES = 3


class RedOracle(torch.nn.Module):
    def forward(self, images01):
        centers = torch.tensor(CENTERS, dtype=images01.dtype, device=images01.device).view(1, -1, 1, 1)
        return -(images01[:, :1] - centers).abs()


class ChannelOracle(torch.nn.Module):
    def forward(self, images01):
        return images01.mean(dim=(2, 3))


class SegAdapter:
    def __init__(self):
        self.model = RedOracle()
        self.calls = 0

    def to_unit(self, batch):
        return batch["image"], batch["label"]

    def predict(self, images01, batch):
        self.calls += 1
        return self.model(images01)


class ClsAdapter:
    def __init__(self):
        self.model = ChannelOracle()

    def to_unit(self, batch):
        return batch["image"], None

    def predict(self, images01, batch):
        return self.model(images01)

    def target(self, label, batch):
        return batch["target"]


def seg_items(n=6, h=24, w=36, seed=0):
    """Three vertical stripes, red at the class centers; green and blue random."""
    g = torch.Generator().manual_seed(seed)
    label = torch.zeros(h, w, dtype=torch.long)
    label[:, w // 3 : 2 * w // 3] = 1
    label[:, 2 * w // 3 :] = 2
    items = []
    for i in range(n):
        image = torch.rand(3, h, w, generator=g)
        image[0] = torch.tensor(CENTERS)[label]
        # 8-bit representable, like a decoded image.
        image = (image * 255).round() / 255
        items.append({"image": image, "label": label.clone(), "id": f"img{i}"})
    return items


def cls_items(n=9, size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n):
        target = i % N_CLASSES
        image = 0.2 + 0.05 * torch.rand(3, size, size, generator=g)
        image[target] += 0.5
        items.append({"image": (image * 255).round() / 255, "target": target, "id": f"img{i}"})
    return items


def loader(items, batch_size=4):
    return DataLoader(items, batch_size=batch_size, shuffle=False)


@dataclass(frozen=True)
class FakeGeometric:
    """A geometric op that shifts the image but hands its label back unmoved: the bug
    ``apply_op`` exists to refuse."""

    name: str = "fake_shift"
    kind: str = "geometric"
    mag_range: Optional[Tuple[float, float]] = (0.0, 1.0)
    identity_at: Optional[float] = 0.0
    grad_mode: str = "autograd"
    warps_label: bool = True
    family: str = "fake"
    source: str = "test"
    fn: Callable = None

    def apply(self, images, magnitude=None, label=None):
        return torch.roll(images, shifts=3, dims=-1), label
