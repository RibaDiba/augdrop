"""Training-time augmentation for mmseg: the vocab's differentiable ops, applied to the
batch on GPU after collation.

Ported from Sensaug's ``GpuAugSegDataPreProcessor`` (sensaug/dataset/gpu_augment.py).
The ops are batched GPU tensor functions, and that is how the reference implementation
(Shu et al., ICRA 2021) applies them: to each batch, inside the training step. A
per-image pipeline transform runs on CPU in a dataloader worker with a numpy<->torch
round trip per sample, which is what made Sensaug's per-image path unaffordable.

The de-normalize -> op -> re-normalize arithmetic is the backend's (``MMSegBackend``
``to_unit`` / ``_normalize``): training must apply the SAME function to the same
tensor that the probe differentiates.

What changed from Sensaug. The policy is a dropaug host, built from the ``host``
config and exposed as ``.host`` for the hooks: they read its schedule and vocabulary,
and hand it q. The sampling is Sensaug's, read off the host (``tasks.augment``, shared
with the cls loop):

- a ``sensaug`` host with an SA pdf: ``(op, level)`` drawn from ``host.pdf()`` (q
  already applied), level jittered by N(0, 0.1) and clipped to [0, 1], as
  ``RandomTrainTransformNew`` did
- otherwise (a ``uniform`` host, or ``sensaug`` before its first pdf): op drawn from
  ``host.sampling_probs()`` with ``host.none_prob`` for "none", level uniform on
  [0, 1], as ``RandomAlphaTrainTransform`` did

- a ``trivialaugment_diff`` host: TrivialAugmentWide's policy (one op, one uniform
  bin) over vocab's ``trivialaug_diff``, at the bin's magnitude in the op's units
- a ``randaugment_diff`` host: RandAugment's published (N, M) over ``randaug_diff``,
  N draws per image applied in turn
- an ``augmix_diff`` host: AugMix as Sensaug ran it over ``augmix_diff``, chains mixed
  with the clean image, the mask left as it was (Sensaug's behaviour)
- a ``smartsampling_diff`` host: SmartSamplingAugment over ``smartaug_diff``, 2 distinct
  ops at one level with P annealed by the step (``host.draw_slots``; HostProgressHook
  sets the step)
- an ``autoaugment_diff`` host: AutoAugment's ImageNet policy over ``randaug_diff`` (and
  ``autoaugment.Invert``), a sub-policy per image and its two steps (``host.draw_slots``)
- a ``madaug`` host (``POLICY_HOSTS``): MADAug's ops at its policy's per-image draws, which
  depend on the model, so ``MADAugPolicyHook`` (``madaug_hooks.py``) computes them and
  installs them before each step (``set_policy_slots``); this applies them
  (``apply_slots``) and raises if a training batch arrives without them
- a ``faster_autoaugment`` host (``STAGE_HOSTS``): Faster AutoAugment's searched policy over
  ``faster_aa``, a sub-policy per image and its stages applied in turn
  (``host.draw_stages``); the policy file comes from scripts/faa_search.py

The torchvision methods (RandAugment, TrivialAugment, AutoAugment) are refused
(``check_trainable``): they train through those ``*_diff`` hosts.
``augment=False`` builds the host (so the hooks can probe) and leaves training clean:
Sensaug's control arm.

The eval slot is Sensaug's ``set_eval_spec``: ``set_eval(op, level)`` applies one op at
one level to every image on the ``training=False`` path, labels moving with the
geometric ops, until ``set_eval_none()``. Only the SA loop (``loops.py``) sets it. It is
independent of ``augment``, so the control arm's SA still measures.

Sensaug's ``pruned_ops`` is the host's static prune (``host=dict(..., pruned=[...])``,
``AugHost.prune``, from the ``prune:`` config block): a pruned op has probability 0 in
every draw, the warmup policy's included.
"""

import contextlib
from typing import Optional, Tuple

import torch
from mmcv.transforms import BaseTransform
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS, TRANSFORMS

from hosts import build_host
from vocab import get_op

from ..augment import NO_OP, POLICY_HOSTS, apply_draws, apply_slots, augment_batch, check_trainable, sample_draws

__all__ = ["GpuAugSegDataPreProcessor", "NO_OP", "apply_draws", "augment_batch", "sample_draws"]


@TRANSFORMS.register_module()
class BinarizeLabel(BaseTransform):
    """``gt_seg_map`` > ``threshold`` becomes 1, the rest 0 (after ``LoadAnnotations``).

    For masks stored as 0/255 (ISIC 2017): mmseg reads 255 as the ignore label, so they
    would train with no foreground class. The files stay as downloaded."""

    def __init__(self, threshold: int = 127):
        self.threshold = threshold

    def transform(self, results: dict) -> dict:
        seg = results["gt_seg_map"]
        results["gt_seg_map"] = (seg > self.threshold).astype(seg.dtype)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"


@MODELS.register_module()
class GpuAugSegDataPreProcessor(SegDataPreProcessor):
    """``SegDataPreProcessor`` that augments each training batch on GPU.

    host     the host's config: ``dict(type=<build_host name>, **its kwargs)``,
             e.g. ``dict(type="sensaug", max_iters=160000)``
    augment  False keeps the host (the hooks need it) but trains on clean images
    """

    def __init__(self, *args, host: dict, augment: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        host = dict(host)
        self.host = build_host(host.pop("type"), **host)
        self.ops = {name: get_op(name) for name in self.host.vocabulary()}
        self.augment = bool(augment)
        if self.augment:
            check_trainable(self.host, self.ops)
        self._suspended = False
        self._eval_spec: Optional[Tuple[str, float]] = None
        self._policy_slots = None

    @contextlib.contextmanager
    def suspended(self):
        """No augmentation on either path for the duration, for measurement code that
        needs the ``training=True`` branch (mmseg pads ``gt_sem_seg`` only there) but a
        clean batch: the probe's ``to_unit``. Sensaug's ``suspended_augmentation``."""
        previous, self._suspended = self._suspended, True
        try:
            yield
        finally:
            self._suspended = previous

    def set_eval(self, op: str, level: float) -> None:
        """Apply ``op`` at ``level`` to every image on the eval path (Sensaug's
        ``set_eval_spec``). ``NO_OP`` sets a clean slot."""
        if op != NO_OP and op not in self.ops:
            raise KeyError(f"unknown op {op!r}; the host's vocabulary is {sorted(self.ops)}")
        self._eval_spec = (op, float(level))

    def set_eval_none(self) -> None:
        """Clean images on the eval path again."""
        self._eval_spec = None

    def set_policy_slots(self, slots) -> None:
        """A ``POLICY_HOSTS`` host's draws for the next training batch: slots of per-image
        ``(op, magnitude)`` (``MADAugHost.draw``). Used once, by that batch."""
        self._policy_slots = slots

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training)
        if self._suspended:
            return data
        inputs = data["inputs"]
        if training:
            if not self.augment:
                return data
            draws = None  # the host's whole policy: augment_batch
            if self.host.name in POLICY_HOSTS:
                if self._policy_slots is None:
                    raise RuntimeError(
                        f"host {self.host.name!r} trains on its policy's draws, and none were "
                        f"installed for this batch; it trains with MADAugPolicyHook (config.py adds "
                        f"it for a host with a policy: block), or set augment=False to train clean"
                    )
                draws, self._policy_slots = self._policy_slots, None
        else:
            if self._eval_spec is None or self._eval_spec[0] == NO_OP:
                return data
            draws = [self._eval_spec] * int(inputs.shape[0])

        samples = data.get("data_samples") or []

        mean = self.mean.view(1, -1, 1, 1)
        std = self.std.view(1, -1, 1, 1)
        # no_grad: the loss is backpropagated to the weights and the graph stops at
        # the input. Only the probe differentiates an op, and it builds its own graph.
        with torch.no_grad():
            rgb01 = ((inputs * std + mean) / 255.0).clamp(0.0, 1.0)
            labels = _gather_labels(samples)
            if draws is None:
                rgb01, labels = augment_batch(self.host, rgb01, labels, self.ops)
            elif training:
                rgb01, labels = apply_slots(rgb01, labels, draws, self.ops)
            else:
                rgb01, labels = apply_draws(rgb01, labels, draws, self.ops)
            data["inputs"] = (rgb01 * 255.0 - mean) / std

        if labels is not None:
            for sample, label in zip(samples, labels):
                sample.gt_sem_seg.data = label
        return data


def _gather_labels(samples) -> Optional[torch.Tensor]:
    """(B, 1, H, W) labels, or None when the batch has none. Raises on labels that
    can't be stacked, where Sensaug fell back to warping the image alone: a geometric
    op would then train on, or be scored against, a label that no longer lines up."""
    gts = [getattr(sample, "gt_sem_seg", None) for sample in samples]
    if not gts or all(gt is None for gt in gts):
        return None
    shapes = {None if gt is None else tuple(gt.data.shape) for gt in gts}
    if len(shapes) != 1 or None in shapes:
        raise ValueError(
            f"labels of shapes {sorted(map(str, shapes))} can't be stacked; the train "
            f"pipeline must crop every image and label to one size, and the val "
            f"dataloader the SA loop evaluates on needs batch size 1"
        )
    return torch.stack([gt.data for gt in gts])
