"""TaskBackend for image classifiers: timm and torchvision models, or any nn.Module
that maps normalized (B, 3, H, W) images to (B, K) logits.

The batch is ``dropaug.eval``'s classification item, collated by torch's default
collate::

    {"image": (B, 3, H, W) float RGB in [0, 1], "target": (B,) class ids, "id": [B ids]}

``ImageFolderItems`` and ``CifarItems`` here yield it for clean data, and
``dropaug.eval``'s ``ImageNetC`` / ``CifarC`` for corrupted data, with the same ids
(``<wnid>/<file>``, and the test-set index for CIFAR). So one backend is the probe's
adapter, the clean eval's and the corruption eval's.

Normalization belongs to the backend (``mean`` / ``std``), not the dataset: the probe
perturbs [0, 1] images, and ``to_unit`` refuses a batch outside [0, 1] so a dataset
that normalizes is caught at the first batch. The clean transform is deterministic
(resize the short side, center crop), which the probe's contract requires.

There is no label map. ``to_unit`` returns label None; the geometric ops move the image
alone, which is right, since a class id is invariant to rotation and translation. The
loss is plain per-image cross-entropy (``label_smoothing`` optional). Whatever the
training recipe added on top (Mixup, CutMix, distillation) is left out on purpose,
because the batch-level ones couple the images.
"""

import os
from collections.abc import Mapping
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from probe import ProbeError

from ..base import ProbeData, TaskBackend

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "ClsBackend", "CifarItems", "ImageFolderItems", "build_dataset"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


# --- clean datasets -------------------------------------------------------------------


def _center(image: torch.Tensor, resize: Optional[int], crop: Optional[int]) -> torch.Tensor:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    if resize is not None:
        image = TF.resize(image, resize, interpolation=InterpolationMode.BILINEAR, antialias=True)
    if crop is not None:
        image = TF.center_crop(image, crop)
    return image.clamp_(0.0, 1.0)


class ImageFolderItems(torch.utils.data.Dataset):
    """``<root>/<class>/<image>``, the layout torchvision's ImageFolder and ImageNet
    val (sorted into wnid folders) use, as clean items.

    Class ids index ``classes``, which defaults to the sorted class folders, as
    ImageFolder orders them. Each image's short side is resized to ``resize`` and the
    center ``crop`` x ``crop`` kept: 256/224 is the ImageNet evaluation transform, and
    what ImageNet-C was rendered from. Item ``"id"`` is ``<class>/<file>``.
    """

    def __init__(self, root: str, *, resize: Optional[int] = 256, crop: Optional[int] = 224, classes=None):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"no image folder {root}")
        if classes is None:
            classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.classes = list(classes)
        self.samples = []
        for index, name in enumerate(self.classes):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"class folder {folder} is missing")
            for file in sorted(os.listdir(folder)):
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    self.samples.append((os.path.join(folder, file), index, f"{name}/{file}"))
        if not self.samples:
            raise FileNotFoundError(f"no images under {root}")
        self.ids = [image_id for _, _, image_id in self.samples]
        self.resize, self.crop = resize, crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from torchvision.io import ImageReadMode, decode_image

        path, target, image_id = self.samples[i]
        image = decode_image(path, mode=ImageReadMode.RGB).float().div_(255.0)
        return {"image": _center(image, self.resize, self.crop), "target": target, "id": image_id}


class CifarItems(torch.utils.data.Dataset):
    """CIFAR-10 / CIFAR-100 as torchvision stores them, as clean items. Never downloads
    (compute nodes have no internet); stage the data at ``root``. Item ``"id"`` is the
    index in the split, which is CIFAR-10-C's id for the test split."""

    def __init__(self, root: str, *, name: str = "cifar10", train: bool = False):
        from torchvision import datasets

        classes = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
        if name not in classes:
            raise ValueError(f"unknown CIFAR {name!r}; expected one of {sorted(classes)}")
        self.data = classes[name](root, train=train, download=False)
        self.classes = list(self.data.classes)
        self.ids = [str(i) for i in range(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = torch.from_numpy(self.data.data[i]).permute(2, 0, 1).float().div_(255.0)
        return {"image": image, "target": int(self.data.targets[i]), "id": str(i)}


def build_dataset(spec: Mapping) -> torch.utils.data.Dataset:
    """A clean dataset from a config: ``dict(type="imagefolder", root=..., resize=256,
    crop=224)`` or ``dict(type="cifar10" | "cifar100", root=..., train=False)``."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "imagefolder":
        return ImageFolderItems(**spec)
    if kind in ("cifar10", "cifar100"):
        return CifarItems(name=kind, **spec)
    raise ValueError(f"unknown dataset type {kind!r}; expected imagefolder, cifar10 or cifar100")


# --- the backend ----------------------------------------------------------------------


class ClsBackend(TaskBackend):
    """A classifier as a probe (and eval) adapter.

    model    an nn.Module, or ``dict(library="timm" | "torchvision", name=..., **kwargs)``
             (e.g. ``num_classes=100``), built by ``build_model``
    data     the clean probe data: a ``ProbeData``, a dataset yielding the items above,
             or a ``build_dataset`` config
    mean, std
             the model's input normalization, in [0, 1] units. Default: a timm
             model's ``pretrained_cfg``, else ImageNet's. Give your own for a model
             trained with other statistics (CIFAR recipes usually are).
    label_smoothing
             for the probe's cross-entropy; 0 is the plain likelihood
    device   where ``build_model`` puts the model; default cuda when available
    """

    name = "cls"

    def __init__(
        self,
        model,
        *,
        data=None,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        label_smoothing: float = 0.0,
        device=None,
    ):
        self.spec = None if isinstance(model, torch.nn.Module) else dict(model)
        self.model = model if isinstance(model, torch.nn.Module) else None
        self.data = data
        self._mean, self._std = mean, std
        self.label_smoothing = float(label_smoothing)
        self.device = device

    # --- building ---------------------------------------------------------------

    def build_model(self) -> torch.nn.Module:
        """The classifier, on ``device``, in eval(). A timm model is built without
        pretrained weights unless ``pretrained=True`` is in its spec (compute nodes
        can't download them); weights come from ``load_ckpt``."""
        if self.model is None:
            spec = dict(self.spec)
            library, name = spec.pop("library", "timm"), spec.pop("name")
            if library == "timm":
                import timm

                spec.setdefault("pretrained", False)
                self.model = timm.create_model(name, **spec)
            elif library == "torchvision":
                from torchvision import models

                self.model = models.get_model(name, **spec)
            else:
                raise ValueError(f"unknown model library {library!r}; expected timm or torchvision")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device).eval()
        return self.model

    def load_ckpt(self, path, key: Optional[str] = None) -> dict:
        """Load a checkpoint's weights onto the model and return its other entries
        (epoch, arch, metric, ...). The state dict is ``key`` when given, else the first
        of ``state_dict`` / ``model`` / ``state_dict_ema`` present, else the file itself
        (a bare state dict). A DDP ``module.`` prefix is stripped. Loading is strict."""
        if self.model is None:
            raise RuntimeError("no model to load into: call build_model() first")
        # torch >= 2.6 defaults to weights_only=True, which refuses the non-tensor
        # entries training scripts save next to the weights. These are first-party files.
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        state, meta = checkpoint, {}
        if isinstance(checkpoint, Mapping) and not all(torch.is_tensor(v) for v in checkpoint.values()):
            keys = [key] if key is not None else [k for k in ("state_dict", "model", "state_dict_ema") if k in checkpoint]
            if not keys or keys[0] not in checkpoint:
                raise KeyError(f"{path} holds {sorted(checkpoint)}; pass key= to name the state dict")
            state = checkpoint[keys[0]]
            meta = {k: v for k, v in checkpoint.items() if not isinstance(v, Mapping)}
        state = {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        return meta

    def probe_data(self) -> ProbeData:
        """The clean dataset and its item ids."""
        if self.data is None:
            raise ValueError("no probe data: give ClsBackend data=")
        if isinstance(self.data, ProbeData):
            return self.data
        dataset = build_dataset(self.data) if isinstance(self.data, Mapping) else self.data
        ids = getattr(dataset, "ids", None)
        if ids is None:
            # Reads every item once, image included; a dataset with an `ids` list skips it.
            ids = [dataset[i]["id"] for i in range(len(dataset))]
        return ProbeData(dataset, list(ids), source=type(dataset).__name__)

    def batch_ids(self, batch) -> list:
        return [str(i) for i in batch["id"]]

    # --- the probe's adapter -----------------------------------------------------

    def to_unit(self, batch):
        images = batch["image"].to(self._device(), torch.float32)
        low, high = float(images.min()), float(images.max())
        if low < 0.0 or high > 1.0:
            raise ProbeError(
                f"batch images span [{low:.3f}, {high:.3f}], not [0, 1]. The dataset must "
                "not normalize; the backend does (mean/std)."
            )
        return images, None

    def loss(self, images01, label, batch) -> torch.Tensor:
        logits = self.model(self._normalize(images01))
        return F.cross_entropy(
            logits, self.target(label, batch), reduction="none", label_smoothing=self.label_smoothing
        )

    # --- dropaug.eval's adapter --------------------------------------------------

    def predict(self, images01, batch) -> torch.Tensor:
        return self.model(self._normalize(images01))

    def target(self, label, batch) -> torch.Tensor:
        return torch.as_tensor(batch["target"], device=self._device()).long().reshape(-1)

    # --- helpers -----------------------------------------------------------------

    def _device(self):
        return next(self.model.parameters()).device

    def _mean_std(self):
        mean, std = self._mean, self._std
        cfg = getattr(self.model, "pretrained_cfg", None) or {}
        mean = mean if mean is not None else cfg.get("mean", IMAGENET_MEAN)
        std = std if std is not None else cfg.get("std", IMAGENET_STD)
        device = self._device()
        return (
            torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1),
            torch.as_tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1),
        )

    def _normalize(self, images01: torch.Tensor) -> torch.Tensor:
        mean, std = self._mean_std()
        return (images01 - mean) / std
