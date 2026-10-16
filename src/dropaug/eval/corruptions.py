"""Common corruptions: ImageNet-C, CIFAR-C and Cityscapes-C, pre-rendered or on the fly.

There are two ways to get a corrupted image, and their numbers are not interchangeable.

PRE-RENDERED. Hendrycks & Dietterich (ICLR 2019) released ImageNet-C, Tiny-ImageNet-C,
CIFAR-10-C and CIFAR-100-C as files, rendered once at the dataset's own resolution (and,
for ImageNet-C, JPEG-encoded). ``ImageNetC``, ``CifarC`` and ``CityscapesC`` read them.
The AlexNet-normalized mCE (``mce``) is defined on the released ImageNet-C files only.

ON THE FLY. ``corrupt`` applies the same 19 corruptions through the ``imagecorruptions``
package (Michaelis et al., 2019), the re-implementation that works at any image size and
without ImageMagick. ``corruption_perturbation`` applies it to the unit batch the model
sees, after the adapter's own resize. That is where Sensaug's test_robust.py applied its
corruptions (``ImageNetCTransform``, inserted after Resize), and it is the usual way to
get Cityscapes-C, which has no official release. Sensaug ran Hendrycks' original code
(sensaug/dataset/imagenet_c.py: motion blur and snow through Wand/ImageMagick, glass
blur hardcoded to 224x224), so its motion_blur and snow numbers do not reproduce here.

``combination`` is Sensaug's ``CombinationPerturbation``: six fixed settings of its base
ops (RGB and HSV shifts, Gaussian blur, Gaussian noise) applied together with cv2 on
8-bit images. test_robust.py reported it as its own category.

Items. Every dataset here yields a dict with ``"image"``, a (3, H, W) float RGB tensor in
[0, 1], then either ``"target"`` (classification, an int) or ``"label"`` (segmentation,
an (H, W) int64 map where 255 means ignore), and ``"id"``. ``"id"`` is the same across
corruptions and severities, so rows can be matched. The default collate stacks these
items. A ``transform(item) -> item``, when given, runs last.

Optional dependencies are imported on first use: ``imagecorruptions`` for on-the-fly
corruptions and ``cv2`` for ``combination``. imagecorruptions 1.1.2 (the latest
release) predates NumPy 2 and scikit-image 0.19. ``fog`` needs numpy<2 (it uses
``np.float_``), and ``glass_blur`` and ``gaussian_blur`` need scikit-image<0.19 (they
pass ``multichannel=``). The other 16 run on current versions.
"""

import contextlib
import importlib
import os
import zlib
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image

Tensor = torch.Tensor

SEVERITIES = (1, 2, 3, 4, 5)

#: The 15 benchmark corruptions, in Hendrycks & Dietterich's order. ALEXNET_ERR and
#: every published mCE table use this order.
IMAGENET_C = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
#: The 4 held-out corruptions released alongside ("extra" / "validation"), not in mCE.
IMAGENET_C_EXTRA = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
ALL_CORRUPTIONS = IMAGENET_C + IMAGENET_C_EXTRA

CATEGORIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"),
    "weather": ("snow", "frost", "fog", "brightness", "spatter"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate"),
}

#: The 7 corruptions test_robust.py evaluated as "unseen" (IMAGENETC_NAME_FN_DICT).
SENSAUG_UNSEEN = ("motion_blur", "zoom_blur", "pixelate", "jpeg_compression", "snow", "frost", "fog")

#: AlexNet's top-1 error rate per corruption, averaged over the 5 severities: mCE's
#: normalizer. From hendrycks/robustness, as google-research/augmix imagenet.py copies it.
ALEXNET_ERR = {
    "gaussian_noise": 0.886428,
    "shot_noise": 0.894468,
    "impulse_noise": 0.922640,
    "defocus_blur": 0.819880,
    "glass_blur": 0.826268,
    "motion_blur": 0.785948,
    "zoom_blur": 0.798360,
    "snow": 0.866816,
    "frost": 0.826572,
    "fog": 0.819324,
    "brightness": 0.564592,
    "contrast": 0.853204,
    "elastic_transform": 0.646056,
    "pixelate": 0.717840,
    "jpeg_compression": 0.606500,
}

COMBINATION = "combination"
# sensaug/dataset/parameters.py. Columns: B, G, R, H, S, V shift alphas (the sign is
# the direction), blur kernel size, noise sigma (0-255 units), and one unused column.
COMBINATION_PARAMETERS = (
    (-0.1180, 0.4343, 0.1445, 0.3040, -0.2600, 0.1816, 3.0, 10.0, 17.0),
    (0.0420, -0.5085, 0.3695, -0.0570, -0.1978, -0.4526, 27.0, 7.0, 68.0),
    (0.1774, -0.1150, 0.1299, -0.0022, -0.2119, -0.0747, 1.0, 6.0, 86.0),
    (-0.2599, -0.0166, -0.2702, -0.4273, 0.0238, -0.2321, 5.0, 8.0, 8.0),
    (-0.2047, 0.0333, 0.3342, -0.4400, 0.2513, 0.0013, 35.0, 6.0, 1.0),
    (-0.6613, -0.0191, 0.3842, 0.3568, 0.5522, 0.0998, 21.0, 3.0, 37.0),
)
#: test_robust.py's combination "levels": setting k is COMBINATION_PARAMETERS[k - 1].
#: Six unordered settings, not severities.
COMBINATION_LEVELS = (1, 2, 3, 4, 5, 6)

_CATEGORY = {name: cat for cat, names in CATEGORIES.items() for name in names}

# (image uint8 HWC, name, severity, seed) -> image uint8 HWC
CorruptFn = Callable[[np.ndarray, str, int, int], np.ndarray]


def category(name: str) -> Optional[str]:
    """noise / blur / weather / digital, ``"combination"``, or None for any other name."""
    if name == COMBINATION:
        return COMBINATION
    return _CATEGORY.get(name)


def default_severities(name: str) -> tuple:
    return COMBINATION_LEVELS if name == COMBINATION else SEVERITIES


def corruption_source(name: str) -> str:
    if name == COMBINATION:
        return "sensaug CombinationPerturbation (cv2, 8-bit)"
    return "imagecorruptions (on the fly, 8-bit)"


def check_backend(name: str) -> None:
    """Raise now if ``corrupt(..., name, ...)`` could not run: an unknown name, or its
    optional dependency missing. A suite checks this before its first eval rather than
    an hour in."""
    if name == COMBINATION:
        _require("cv2", "opencv-python")
    elif name in ALL_CORRUPTIONS:
        _require("imagecorruptions")
    else:
        raise KeyError(f"unknown corruption {name!r}; expected one of {ALL_CORRUPTIONS + (COMBINATION,)}")


def corrupt(image: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
    """One (H, W, 3) uint8 RGB image, corrupted, with the same shape and dtype.

    ``seed`` fixes every random draw. imagecorruptions draws from numpy's global RNG,
    which is seeded for the call and restored afterwards, so no caller's stream moves.
    """
    if name == COMBINATION:
        return combination(image, severity, seed)
    if name not in ALL_CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; expected one of {ALL_CORRUPTIONS + (COMBINATION,)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
    backend = _require("imagecorruptions")
    with _numpy_seeded(seed):
        out = backend.corrupt(image, corruption_name=name, severity=int(severity))
    return np.asarray(out, dtype=np.uint8)


def combination(image: np.ndarray, level: int, seed: int) -> np.ndarray:
    """Sensaug's ``CombinationPerturbation`` at setting ``level`` (1-6, its ``choice`` + 1),
    on an (H, W, 3) uint8 RGB image.

    Same arithmetic in the same order: a per-channel RGB shift, then an HSV shift in
    OpenCV's 8-bit HSV (H in [0, 180)), then a cv2 Gaussian blur, then Gaussian noise.
    One deliberate difference: Sensaug wrote each shifted channel back into the uint8
    image, so a value past 255 overflowed the cast (in practice it wrapped, and a bright
    pixel under a 1.12x gain came out dark). Here it is clipped to [0, 255]. Values
    inside [0, 255] are truncated as before.
    """
    cv2 = _require("cv2", "opencv-python")
    if level not in COMBINATION_LEVELS:
        raise ValueError(f"combination level must be one of {COMBINATION_LEVELS}, got {level!r}")
    b, g, r, h, s, v, ksize, sigma, _ = COMBINATION_PARAMETERS[level - 1]

    rgb = image.astype(np.float64)
    for channel, alpha in ((0, r), (1, g), (2, b)):
        rgb[..., channel] = _shift(rgb[..., channel], alpha, 255.0)
    hsv = cv2.cvtColor(_to_uint8(rgb), cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv[..., 0] = _shift(hsv[..., 0], h, 180.0)
    hsv[..., 1] = _shift(hsv[..., 1], s, 255.0)
    # V darkens toward 10, not 0, in the original.
    hsv[..., 2] = hsv[..., 2] - v * hsv[..., 2] + (10.0 if v < 0 else 255.0) * v
    out = cv2.cvtColor(_to_uint8(hsv), cv2.COLOR_HSV2RGB)
    out = cv2.GaussianBlur(out, (int(ksize), int(ksize)), 0)
    noisy = out + np.random.default_rng(seed).normal(0.0, sigma, size=out.shape)
    return _to_uint8(noisy)


class ImageCorruption:
    """One corruption at one severity, as ``(image, position) -> image`` on one (H, W, 3)
    uint8 RGB image, where ``position`` is the image's place in the (unshuffled) pass.

    The image is seeded by (seed, name, severity, position), so its draw doesn't depend
    on the batch size or on which process applies it. ``corruption_perturbation`` applies
    it to a unit batch in the eval's own process. A backend's ``corrupted_dataset``
    applies it to each item as the loader reads it, in the DataLoader's workers, as
    Sensaug's test_robust.py did (its ImageNetCTransform sat after Resize). Both see the
    same 8-bit image at the same position, so they give the same pixels.
    """

    def __init__(self, name: str, severity: int, *, seed: int = 0, fn: Optional[CorruptFn] = None):
        if fn is None:
            if name != COMBINATION and name not in ALL_CORRUPTIONS:
                raise KeyError(f"unknown corruption {name!r}")
            if severity not in default_severities(name):
                raise ValueError(f"{name} severity must be one of {default_severities(name)}, got {severity!r}")
        self.name, self.severity, self.seed = name, severity, seed
        self.fn = fn

    def __call__(self, image: np.ndarray, position: int) -> np.ndarray:
        image_seed = zlib.crc32(f"{self.seed}:{self.name}:{self.severity}:{position}".encode())
        corrupted = np.asarray((self.fn or corrupt)(image, self.name, self.severity, image_seed))
        if corrupted.shape != image.shape:
            raise ValueError(
                f"corruption {self.name!r} returned shape {corrupted.shape} for an image of "
                f"shape {image.shape}; it must keep the image's size"
            )
        # Writable too: some backends hand back a read-only view, which torch warns on.
        return np.require(corrupted, np.uint8, ["C", "W"])


def corruption_perturbation(name: str, severity: int, *, seed: int = 0, fn: Optional[CorruptFn] = None):
    """A condition's ``perturb(images01, label, index)``: every image of the unit batch
    through ``ImageCorruption(name, severity, seed=seed, fn=fn)`` (``fn`` defaults to
    ``corrupt``), on CPU at 8 bits, one image at a time in this process.

    Image j of a batch is at position ``index + j``, where ``index`` is how many images
    the pass has seen. That makes each image's draw independent of the batch size,
    provided the loader does not shuffle. The label passes through: none of these
    corruptions moves pixels far enough to move a label (the benchmarks keep it).
    """
    corrupt_image = ImageCorruption(name, severity, seed=seed, fn=fn)

    def perturb(images01: Tensor, label: Optional[Tensor], index: int):
        batch = (images01.detach() * 255.0).round().clamp(0, 255).to(torch.uint8)
        batch = batch.permute(0, 2, 3, 1).cpu().numpy()
        out = [torch.from_numpy(corrupt_image(image, index + j)) for j, image in enumerate(batch)]
        corrupted01 = torch.stack(out).permute(0, 3, 1, 2).to(images01.device, images01.dtype) / 255.0
        return corrupted01, label

    return perturb


def mce(accuracies: Mapping[str, Sequence[float]], baseline: Mapping[str, float] = ALEXNET_ERR) -> dict:
    """ImageNet-C's mean Corruption Error (Hendrycks & Dietterich, 2019), in percent.

    ``accuracies[c]`` is top-1 accuracy in percent at each of the 5 severities, which is
    what ``ClsMetric`` reports. CE_c is the mean error over severities divided by
    ``baseline[c]`` (AugMix's ``compute_mce``). This equals the paper's ratio of sums,
    because the baseline is AlexNet's mean over the same severities. mCE is the mean CE
    over every baseline corruption. All of them must be present, since an mCE over a
    subset is not comparable to a published one.
    """
    missing = [c for c in baseline if c not in accuracies]
    if missing:
        raise ValueError(f"mCE needs every baseline corruption; missing {missing}")
    ce = {}
    for name, base in baseline.items():
        acc = np.asarray(accuracies[name], dtype=float)
        if acc.shape != (len(SEVERITIES),):
            raise ValueError(f"{name}: expected one accuracy per severity {SEVERITIES}, got {acc.tolist()}")
        ce[name] = 100.0 * float(np.mean(1.0 - acc / 100.0)) / base
    return {"mCE": float(np.mean(list(ce.values()))), "CE": ce}


# --- pre-rendered datasets ------------------------------------------------------------


class CifarC(torch.utils.data.Dataset):
    """CIFAR-10-C / CIFAR-100-C, as released:

        <root>/<corruption>.npy     (5N, 32, 32, 3) uint8, severities 1-5 stacked in order
        <root>/labels.npy           (5N,)

    N is 10000 for the released sets. Item ``"id"`` is the index into the clean test
    set, so it is the same at every corruption and severity.
    """

    def __init__(self, root: str, corruption: str, severity: int, transform=None):
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {severity!r}")
        images = np.load(os.path.join(root, f"{corruption}.npy"), mmap_mode="r")
        labels = np.load(os.path.join(root, "labels.npy"))
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"{corruption}.npy: expected (N, H, W, 3), got {images.shape}")
        if len(images) != len(labels) or len(images) % len(SEVERITIES):
            raise ValueError(
                f"{corruption}.npy holds {len(images)} images and labels.npy {len(labels)}; "
                f"expected equal counts divisible by {len(SEVERITIES)} severities"
            )
        n = len(images) // len(SEVERITIES)
        start = (severity - 1) * n
        self.images = images[start : start + n]
        self.labels = labels[start : start + n]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = torch.from_numpy(np.array(self.images[i], dtype=np.uint8)).permute(2, 0, 1)
        item = {"image": image.float().div_(255.0), "target": int(self.labels[i]), "id": str(i)}
        return self.transform(item) if self.transform is not None else item


IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


class ImageNetC(torch.utils.data.Dataset):
    """ImageNet-C (or Tiny-ImageNet-C), as released:

        <root>/<corruption>/<severity>/<wnid>/<image>.JPEG

    Class ids are indices into ``classes``, which defaults to the sorted wnid folders.
    That is ImageNet's standard order, the one torchvision's ImageFolder uses. Pass
    ``classes`` to evaluate a model trained on a subset, in that model's order. Item
    ``"id"`` is ``<wnid>/<file>``.
    """

    def __init__(self, root: str, corruption: str, severity: int, transform=None, classes=None):
        base = os.path.join(root, corruption, str(severity))
        if not os.path.isdir(base):
            raise FileNotFoundError(f"no ImageNet-C folder {base}")
        if classes is None:
            classes = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        self.classes = list(classes)
        self.samples = []
        for index, wnid in enumerate(self.classes):
            folder = os.path.join(base, wnid)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"class folder {folder} is missing")
            for file in sorted(os.listdir(folder)):
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    self.samples.append((os.path.join(folder, file), index, f"{wnid}/{file}"))
        if not self.samples:
            raise FileNotFoundError(f"no images under {base}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, target, image_id = self.samples[i]
        item = {"image": _read_image(path), "target": target, "id": image_id}
        return self.transform(item) if self.transform is not None else item


class Cityscapes(torch.utils.data.Dataset):
    """Clean Cityscapes, read by the same code as ``CityscapesC`` so the clean and
    corrupted numbers differ only by the corruption:

        images  <image_root>/<city>/<stem>_leftImg8bit.png
                (default <root>/leftImg8bit/<split>)
        labels  <root>/gtFine/<split>/<city>/<stem>_gtFine_labelTrainIds.png

    Labels are the 19 train ids with 255 as ignore, as written by cityscapesscripts'
    createTrainIdLabelImgs or mmseg's cityscapes converter. Item ``"id"`` is
    ``<city>/<stem>``.
    """

    IMAGE_SUFFIX = "_leftImg8bit.png"
    LABEL_SUFFIX = "_gtFine_labelTrainIds.png"

    def __init__(self, root: str, split: str = "val", transform=None, image_root: Optional[str] = None):
        image_root = image_root or os.path.join(root, "leftImg8bit", split)
        label_root = os.path.join(root, "gtFine", split)
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"no image folder {image_root}")
        self.samples = []
        for city in sorted(os.listdir(image_root)):
            folder = os.path.join(image_root, city)
            if not os.path.isdir(folder):
                continue
            for file in sorted(os.listdir(folder)):
                if not file.endswith(self.IMAGE_SUFFIX):
                    continue
                stem = file[: -len(self.IMAGE_SUFFIX)]
                label = os.path.join(label_root, city, stem + self.LABEL_SUFFIX)
                if not os.path.isfile(label):
                    raise FileNotFoundError(f"{file} has no label at {label}")
                self.samples.append((os.path.join(folder, file), label, f"{city}/{stem}"))
        if not self.samples:
            raise FileNotFoundError(f"no *{self.IMAGE_SUFFIX} images under {image_root}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        image_path, label_path, image_id = self.samples[i]
        item = {"image": _read_image(image_path), "label": _read_label(label_path), "id": image_id}
        return self.transform(item) if self.transform is not None else item


class CityscapesC(Cityscapes):
    """Pre-rendered Cityscapes-C. There is no official release, so this reads the layout
    you get by corrupting leftImg8bit/val once, with labels from the clean tree:

        images  <root>/<corruption>/<severity>/<city>/<stem>_leftImg8bit.png
        labels  <cityscapes_root>/gtFine/<split>/<city>/<stem>_gtFine_labelTrainIds.png

    Rendering once pays off at Cityscapes' 2048x1024, where glass_blur (a pure-Python
    pixel-shuffling loop) and zoom_blur are slow. On-the-fly corruption at the model's
    input resolution
    (``corruption_perturbation``) is the other option, and the two are different
    measurements.
    """

    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int,
        cityscapes_root: str,
        split: str = "val",
        transform=None,
    ):
        super().__init__(
            cityscapes_root,
            split=split,
            transform=transform,
            image_root=os.path.join(root, corruption, str(severity)),
        )


# --- helpers --------------------------------------------------------------------------


def _read_image(path: str) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)


def _read_label(path: str) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"label {path} is not a single-channel map (shape {array.shape})")
    return torch.from_numpy(array.astype(np.int64))


def _shift(x: np.ndarray, alpha: float, top: float) -> np.ndarray:
    """Sensaug's shift toward ``top`` (alpha >= 0) or away from it (alpha < 0)."""
    direction = 0.0 if alpha < 0 else 1.0
    return x - alpha * x + top * direction * alpha


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


@contextlib.contextmanager
def _numpy_seeded(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _require(module: str, pip_name: Optional[str] = None):
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            f"this corruption needs the optional dependency {module!r} "
            f"(pip install {pip_name or module})"
        ) from error
