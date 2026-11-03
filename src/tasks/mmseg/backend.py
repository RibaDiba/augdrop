"""TaskBackend for MMSegmentation: SegFormer, and any segmentor whose ``loss`` and
``inference`` follow mmseg's ``EncoderDecoder``.

Ported from Sensaug's ``CollectGradientHook._probe_batch`` / ``_grad_for_op``
(sensaug/hooks/grad_hook.py), the parts that know mmseg: running the data
preprocessor, de-normalizing to [0, 1] RGB, stacking ``gt_sem_seg``, swapping a warped
label in, and summing ``model.loss``. The rest of those functions is the probe now.
Also ported: ``scripts/compute_grad_corr.py``'s checkpoint loading, for post-hoc use.

The clean batch. The probe's batch comes from the val (or test) dataloader config,
whose pipeline must contain no random transform. It goes through the model's own
data preprocessor with ``training=True``: mmseg pads and stacks ``gt_sem_seg`` only
on that branch. That branch is also where ``GpuAugSegDataPreProcessor`` applies the
training augmentation, so ``to_unit`` suspends it. A random per-image augmentation
under every probe image would be a factor shared by all ops, inflating all of R.

The label. mmseg val pipelines resize the image but load the label afterwards, at
its native size, so image and label can differ in resolution. The DIFF32 geometric
ops warp the label in its own pixel frame (``vocab.diff_ops.warp_image_and_label``),
and mmseg's loss resizes the logits to the label, so that is handled. Two labels of
different native sizes can't be stacked, which is why a sweep on such a dataset
needs ``batch_size=1`` (Sensaug's default).

The loss is per image: ``model.loss`` on each image of the batch in turn, summed over
its ``loss*`` entries as mmengine's ``parse_losses`` does. At batch size 1 that is
exactly Sensaug's number. At larger batch sizes it is each image's own mean over its
own valid pixels, where Sensaug's batch mean weighted images by their valid-pixel
count (a per-image factor the scale normalization divides out anyway).
"""

import contextlib
import copy
import os
from typing import Optional

import torch
import torch.nn.functional as F

from probe import ProbeError

from ..base import ProbeData, TaskBackend

__all__ = ["MMSegBackend", "functional_logits", "pooled_features", "unwrap"]

#: Pipeline transforms that are not random by name but still are. Anything with
#: "Random" in its type is refused too.
_RANDOM_TRANSFORMS = frozenset(
    {"PhotoMetricDistortion", "BioMedicalGaussianNoise", "BioMedicalGaussianBlur", "TestTimeAug"}
)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The segmentor itself, through DDP if present."""
    return model.module if hasattr(model, "module") else model


def pooled_features(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """(B, C): the segmentor's deepest ``extract_feat`` output (backbone, and neck if it
    has one), average-pooled, for normalized ``inputs``. MADAug's policy reads a
    classifier's penultimate feature; this is a segmentor's nearest."""
    feats = unwrap(model).extract_feat(inputs)
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    return feats.mean(dim=(-2, -1)) if feats.dim() == 4 else feats.flatten(1)


def functional_logits(model: torch.nn.Module, weights: dict, inputs: torch.Tensor, size) -> torch.Tensor:
    """The decode head's logits for normalized ``inputs`` with the segmentor's parameters
    replaced by ``weights`` (``torch.func.functional_call``, ``mode="tensor"``), resized to
    ``size`` the way mmseg's loss resizes them: (B, K, *size)."""
    model = unwrap(model)
    logits = torch.func.functional_call(model, weights, (inputs,), {"mode": "tensor"})
    return F.interpolate(logits, size=tuple(size), mode="bilinear", align_corners=getattr(model, "align_corners", False))


class MMSegBackend(TaskBackend):
    """An mmseg segmentor as a probe (and eval) adapter.

    cfg     an mmengine ``Config`` or a path to one: the experiment's config, which
            supplies the model and the ``{split}_dataloader`` the probe reads
    model   an existing segmentor to wrap (a training run's); omit and call
            ``build_model`` for post-hoc use
    split   which dataloader config is the probe's data: "val" (Sensaug's) or "test"
    device  where ``build_model`` puts the model; default cuda when available
    """

    name = "mmseg"

    def __init__(self, cfg, *, model: Optional[torch.nn.Module] = None, split: str = "val", device=None):
        if isinstance(cfg, (str, os.PathLike)):
            from mmengine.config import Config

            cfg = Config.fromfile(str(cfg))
        self.cfg = cfg
        self.model = None if model is None else unwrap(model)
        self.split = split
        self.device = device
        self._data_root = ""
        # (the loader batch, its preprocessed data_samples), so loss() reuses what
        # to_unit() built instead of re-running the preprocessor for every op.
        self._prepared = None

    @classmethod
    def from_runner(cls, runner, **kwargs) -> "MMSegBackend":
        """Wrap a live mmengine Runner's model, reading its config."""
        return cls(runner.cfg, model=runner.model, **kwargs)

    # --- building ---------------------------------------------------------------

    def build_model(self) -> torch.nn.Module:
        """The segmentor from ``cfg.model``, with SyncBN reverted (no process group
        post hoc), on ``device``, in eval(). ``init_weights`` is not called: the
        weights come from ``load_ckpt``, and a Pretrained init_cfg would try to
        download a backbone (cluster compute nodes have no internet)."""
        from mmengine.model import revert_sync_batchnorm
        from mmengine.registry import init_default_scope
        from mmseg.registry import MODELS

        init_default_scope(self.cfg.get("default_scope", "mmseg"))
        model = revert_sync_batchnorm(MODELS.build(copy.deepcopy(self.cfg.model)))
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device).eval()
        return self.model

    def load_ckpt(self, path, strict: bool = False) -> dict:
        """Load an mmengine checkpoint's weights onto the model; returns its meta
        (``iter``, the dumped ``cfg``, ...). ``strict`` raises ``RuntimeError`` on a
        missing or unexpected key (mmengine's own message lists them); otherwise they
        are only logged, as mmengine does."""
        from mmengine.runner import load_checkpoint

        if self.model is None:
            raise RuntimeError("no model to load into: call build_model() first")
        with _trusted_torch_load():
            checkpoint = load_checkpoint(self.model, str(path), map_location="cpu", strict=strict)
        self.model.eval()
        return dict(checkpoint.get("meta", {})) if isinstance(checkpoint, dict) else {}

    def probe_data(self) -> ProbeData:
        """The ``{split}_dataloader``'s dataset, built as configured, with each image's
        path relative to the dataset's ``data_root`` as its id."""
        from mmengine.registry import init_default_scope
        from mmseg.registry import DATASETS

        loader_cfg = self.cfg.get(f"{self.split}_dataloader")
        if not loader_cfg:
            raise ValueError(f"the config has no {self.split}_dataloader to probe on")
        dataset_cfg = copy.deepcopy(loader_cfg["dataset"])
        _require_clean(dataset_cfg.get("pipeline", []), self.split)

        init_default_scope(self.cfg.get("default_scope", "mmseg"))
        dataset = DATASETS.build(dataset_cfg)
        self._data_root = getattr(dataset, "data_root", None) or ""
        ids = [self._image_id(dataset.get_data_info(i)["img_path"]) for i in range(len(dataset))]
        return ProbeData(dataset, ids, source=f"{type(dataset).__name__}:{self.split}")

    @property
    def collate(self):
        """mmengine's default collate: a list of inputs, a list of data samples."""
        from mmengine.dataset import pseudo_collate

        return pseudo_collate

    def batch_ids(self, batch) -> list:
        return [self._image_id(sample.metainfo["img_path"]) for sample in batch["data_samples"]]

    def _image_id(self, path: str) -> str:
        path = os.path.relpath(path, self._data_root) if self._data_root else path
        return path.replace(os.sep, "/")

    # --- the probe's adapter -----------------------------------------------------

    def to_unit(self, batch):
        dp = self.model.data_preprocessor
        if not getattr(dp, "channel_conversion", True):
            raise ProbeError(
                "the data preprocessor has bgr_to_rgb=False, so the images stay in "
                "mmcv's BGR order and every per-channel op would hit the wrong channel"
            )
        suspended = getattr(dp, "suspended", None)
        with suspended() if suspended is not None else contextlib.nullcontext():
            data = dp(batch, training=True)
        samples = data["data_samples"]
        self._prepared = (batch, samples)
        mean, std = _mean_std(dp)
        images01 = ((data["inputs"] * std + mean) / 255.0).clamp(0.0, 1.0)
        return images01, _stack_labels(samples)

    def loss(self, images01, label, batch) -> torch.Tensor:
        samples = self._samples(batch)
        inputs = self._normalize(images01)
        with _swapped_gt(samples, label):
            losses = [
                _total_loss(self.model.loss(inputs[i : i + 1], [sample]))
                for i, sample in enumerate(samples)
            ]
        return torch.stack(losses)

    # --- dropaug.eval's adapter --------------------------------------------------

    def predict(self, images01, batch) -> torch.Tensor:
        """(B, K, H, W) logits at the label's resolution: the segmentor's own
        inference (whole or slide, per its test_cfg), resized to the label the way
        mmseg's loss resizes them, so eval and probe see the same geometry."""
        samples = self._samples(batch)
        metas = [sample.metainfo for sample in samples]
        logits = self.model.inference(self._normalize(images01), metas)
        label = _stack_labels(samples)
        if label is None:
            return logits
        return F.interpolate(
            logits,
            size=tuple(label.shape[-2:]),
            mode="bilinear",
            align_corners=getattr(self.model, "align_corners", False),
        )

    def corrupted_dataset(self, dataset, corrupt_image):
        """``dataset`` with ``corrupt_image(image, position)`` applied to each item's
        image as it is read: after the pipeline's Resize and before the preprocessor,
        where Sensaug's test_robust.py applied its ImageNetCTransform. Under a
        DataLoader that runs in its workers, alongside the GPU. ``position`` is the
        item's index, so an unshuffled loader gives each image the same seed, and the
        same pixels, as ``corruption_perturbation`` on the unit batch. The one exception
        is an image the preprocessor pads (smaller than its ``size``): this corrupts it
        before the padding, as Sensaug did, where the unit batch already holds the pad."""
        if self.model is None:
            raise RuntimeError("no model: call build_model() first")
        if not getattr(self.model.data_preprocessor, "channel_conversion", True):
            raise ProbeError(
                "the data preprocessor has bgr_to_rgb=False, so the loaded images would "
                "reach the corruption in mmcv's BGR order"
            )
        return _CorruptedItems(dataset, corrupt_image)

    # --- MADAug's policy (tasks/mmseg/madaug.py) -------------------------------------

    def features(self, images01: torch.Tensor) -> torch.Tensor:
        """The policy's input for [0, 1] images: ``pooled_features`` of the model."""
        return pooled_features(self.model, self._normalize(images01))

    def functional_loss(self, weights: dict, batch) -> torch.Tensor:
        """``model.loss`` on the clean ``batch`` with the segmentor's parameters replaced
        by ``weights`` (``torch.func.functional_call``), summed over its ``loss*`` entries:
        the validation loss MADAug's policy update takes at the virtual weights."""
        images01, _ = self.to_unit(batch)
        samples = self._samples(batch)
        losses = torch.func.functional_call(
            self.model, weights, (self._normalize(images01), samples), {"mode": "loss"}
        )
        return _total_loss(losses)

    # --- helpers -----------------------------------------------------------------

    def _samples(self, batch):
        """The preprocessed data samples of ``batch``: to_unit's, if it was the last
        batch it saw, else rebuilt."""
        if self._prepared is None or self._prepared[0] is not batch:
            self.to_unit(batch)
        return self._prepared[1]

    def _normalize(self, images01: torch.Tensor) -> torch.Tensor:
        """[0, 1] RGB back into the model's input space: Sensaug's re-normalization."""
        mean, std = _mean_std(self.model.data_preprocessor)
        return (images01 * 255.0 - mean) / std


class _CorruptedItems(torch.utils.data.Dataset):
    """An mmseg dataset whose packed images go through ``corrupt_image(image, index)``
    as (H, W, 3) uint8 RGB. PackSegInputs leaves the loaded image as a (3, H, W) uint8
    tensor in mmcv's BGR order, which the preprocessor flips to RGB later."""

    def __init__(self, dataset, corrupt_image):
        self.dataset = dataset
        self.corrupt_image = corrupt_image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        inputs = item["inputs"]
        if not torch.is_tensor(inputs) or inputs.dtype != torch.uint8 or inputs.dim() != 3 or inputs.shape[0] != 3:
            raise ProbeError(
                f"expected the packed image as a (3, H, W) uint8 tensor, got "
                f"{getattr(inputs, 'dtype', type(inputs).__name__)} {tuple(getattr(inputs, 'shape', ()))}"
            )
        rgb = inputs.flip(0).permute(1, 2, 0).contiguous().numpy()
        corrupted = torch.from_numpy(self.corrupt_image(rgb, index))
        item["inputs"] = corrupted.permute(2, 0, 1).flip(0).contiguous()
        return item


def _mean_std(dp):
    if getattr(dp, "_enable_normalize", True):
        return dp.mean.view(1, -1, 1, 1), dp.std.view(1, -1, 1, 1)
    return 0.0, 1.0


def _stack_labels(samples) -> Optional[torch.Tensor]:
    """(B, 1, H, W) ground truth, or None when the batch carries none.

    Raises rather than falling back to no label: a geometric op would then warp the
    image without its label, and the loss would measure the misalignment.
    """
    gts = [getattr(sample, "gt_sem_seg", None) for sample in samples]
    if all(gt is None for gt in gts):
        return None
    if any(gt is None for gt in gts):
        raise ProbeError("some samples in this batch carry a gt_sem_seg and some don't")
    shapes = sorted({tuple(gt.data.shape) for gt in gts})
    if len(shapes) != 1:
        raise ProbeError(
            f"labels of mixed sizes {shapes} can't be stacked. mmseg val pipelines load "
            f"the label at its native size; probe this dataset at batch_size=1."
        )
    return torch.stack([gt.data for gt in gts])


@contextlib.contextmanager
def _swapped_gt(samples, label):
    """Install ``label`` (as the op returned it) as the samples' ground truth for the
    duration. Restored on the way out: one set of samples serves every op, and a
    geometric op that left its warped label behind would corrupt every op after it."""
    if label is None:
        yield
        return
    originals = [sample.gt_sem_seg.data for sample in samples]
    try:
        for sample, one in zip(samples, label):
            sample.gt_sem_seg.data = one if one.dim() == 3 else one.unsqueeze(0)
        yield
    finally:
        for sample, original in zip(samples, originals):
            sample.gt_sem_seg.data = original


def _total_loss(losses: dict) -> torch.Tensor:
    """mmengine's parse_losses: every entry whose key contains "loss" (not
    ``acc_seg``), a tensor or a list of them, reduced to a scalar and summed."""
    terms = []
    for key, value in losses.items():
        if "loss" not in key:
            continue
        terms.append(value.mean() if torch.is_tensor(value) else sum(v.mean() for v in value))
    if not terms:
        raise ProbeError(f"model.loss returned no loss entries, only {sorted(losses)}")
    return sum(terms)


def _require_clean(pipeline, split: str) -> None:
    """Refuse a probe pipeline that draws at random: the probe's contract is a clean,
    deterministic batch. (The probe's own check re-runs to_unit on one batch, which
    catches randomness in the preprocessor but not in the dataset pipeline.)"""
    random = [
        t.get("type")
        for t in pipeline
        if "Random" in str(t.get("type", "")) or t.get("type") in _RANDOM_TRANSFORMS
    ]
    if random:
        raise ValueError(
            f"the {split} pipeline has random transforms {random}; the probe needs clean "
            f"images, one fixed version of each"
        )


@contextlib.contextmanager
def _trusted_torch_load():
    """torch >= 2.6 defaults ``torch.load`` to ``weights_only=True``, which can't
    unpickle mmengine checkpoints (``message_hub`` holds HistoryBuffers of numpy
    arrays), and mmengine's loader exposes no override. These are first-party
    checkpoints, so restore the old default for this one load. Sensaug's
    compute_grad_corr.py patched it for the whole process; this scopes it."""
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original
