"""Faster AutoAugment's policy search, for segmentation: dda's ``AdvTrainer``
(github.com/moskomule/dda, ``faster_autoaugment/search.py``) with no training framework.

Half of each batch is augmented by the policy, the other half is the real data. A critic
learns to tell them apart (WGAN-GP), and the policy learns to make augmented images the
critic can't tell from real ones, while a task loss keeps it from changing what an image
shows (paper §4.2, Algorithm 2). One step, as ``AdvTrainer.wgan_loss`` takes it:

1. the critic (with the task head under it): ``task_factor * task(real) - critic(real)
   + critic(policy(a)) + gp_factor * gradient penalty``, the policy's output detached;
2. the policy, the critic frozen: ``task_factor * task(policy(a)) - critic(policy(a))``.

Both use Adam at lr 1e-3, betas (0, 0.999). Paper defaults: 20 epochs, 10 sub-policies of
2 stages, the op softmax and the relaxed Bernoulli at temperature 0.05 (dda's code uses
0.1 for the Bernoulli), task factor ("epsilon") 0.1, gradient penalty 10. The paper
searches on a 4000-image subset of the training set (Table 3).

**For segmentation** (the paper and dda are classification-only), as AutoAlbument
(albumentations-team/autoalbument, ``FAASemanticSegmentationModel``) adapts it:

- the critic and the task model are one segmentor (``SegmentorCritic``): its decode head
  gives the task's logits, and a two-layer perceptron on its deepest feature, average
  pooled, gives the critic's score (dda puts the same perceptron on its classifier's
  penultimate feature). One segmentor searches each dataset, and every backbone then
  trains on the policy it finds (the paper searched with WRN-40-2 / ResNet-50 and trained
  larger models)
- the task loss is a soft-target cross-entropy (``soft_cross_entropy``) against the
  policy's blended soft mask (``policy.py``), each pixel weighted by its non-ignore mass,
  where AutoAlbument uses a binary cross-entropy on one-hot masks

Kept as dda has it: the gradient penalty takes the norm of the critic's input gradient
over the channel dimension, per pixel (``grad.norm(2, dim=1)``), where WGAN-GP takes it
over the whole image. The real images are the second half of the batch and the augmented
ones the first.

The search does not know mmseg: ``scripts/faa_search.py`` builds the segmentor and the
data. ``save_policy`` writes what the host reads (``host.load_policy``).
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .host import POLICY_FORMAT, POLICY_VERSION, load_policy
from .policy import Policy

__all__ = [
    "SearchConfig",
    "SegmentorCritic",
    "gradient_penalty",
    "save_policy",
    "search",
    "search_step",
    "soft_cross_entropy",
    "soft_mask",
]


@dataclass
class SearchConfig:
    """The search's knobs, paper / dda defaults (module docstring)."""

    epochs: int = 20
    task_factor: float = 0.1
    gp_factor: float = 10.0
    lr: float = 1e-3
    policy_lr: float = 1e-3
    betas: Tuple[float, float] = (0.0, 0.999)
    log_interval: int = 20


def soft_mask(label: Tensor, num_classes: int, ignore_index: int = 255) -> Tensor:
    """(B, K + 1, H, W) float: one-hot classes, the last channel the ignore index (and
    any id outside [0, K)), for a (B, H, W) or (B, 1, H, W) integer label."""
    if label.dim() == 4:
        label = label[:, 0]
    label = label.long()
    ids = torch.where((label >= 0) & (label < num_classes), label, torch.full_like(label, num_classes))
    if ignore_index is not None:
        ids = torch.where(label == ignore_index, torch.full_like(label, num_classes), ids)
    return F.one_hot(ids, num_classes + 1).permute(0, 3, 1, 2).float()


def soft_cross_entropy(logits: Tensor, target: Tensor, align_corners: bool = False) -> Tensor:
    """(B,): each image's cross-entropy of ``logits`` (B, K, h, w), resized to the target
    as mmseg's loss resizes them, against the soft ``target`` (B, K + 1, H, W), averaged
    over its non-ignore mass."""
    size = target.shape[-2:]
    if logits.shape[-2:] != size:
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=align_corners)
    classes = target[:, :-1]
    ce = -(classes * F.log_softmax(logits, dim=1)).sum(dim=1)
    valid = classes.sum(dim=1)
    return ce.flatten(1).sum(1) / valid.flatten(1).sum(1).clamp(min=1.0)


class SegmentorCritic(nn.Module):
    """A segmentor (anything with mmseg's ``extract_feat`` and ``decode_head``) as the
    search's task model and critic: ``forward(inputs) -> (logits, critic score (B,))``,
    one backbone pass for both. ``feature_dim`` is the channel count of the segmentor's
    deepest ``extract_feat`` output; the critic head is dda's ``Discriminator``
    perceptron on it, average pooled."""

    def __init__(self, segmentor: nn.Module, feature_dim: int):
        super().__init__()
        self.segmentor = segmentor
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 1)
        )

    @property
    def align_corners(self) -> bool:
        return bool(getattr(self.segmentor, "align_corners", False))

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        feats = self.segmentor.extract_feat(inputs)
        logits = self.segmentor.decode_head.forward(feats)
        deepest = feats[-1] if isinstance(feats, (list, tuple)) else feats
        pooled = deepest.mean(dim=(-2, -1)) if deepest.dim() == 4 else deepest.flatten(1)
        return logits, self.critic(pooled).view(-1)


def gradient_penalty(model: nn.Module, real: Tensor, fake: Tensor) -> Tensor:
    """dda's ``AdvTrainer.gradient_penalty`` (the norm over the channel dimension, per
    pixel, as dda takes it)."""
    alpha = real.new_empty(real.size(0), 1, 1, 1).uniform_(0, 1)
    interpolated = (alpha * real + (1 - alpha) * fake).detach().requires_grad_(True)
    _, output = model(interpolated)
    (grad,) = torch.autograd.grad(
        outputs=output, inputs=interpolated, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True
    )
    return (grad.norm(2, dim=1) - 1).pow(2).mean()


def search_step(
    model: nn.Module,
    policy: Policy,
    images01: Tensor,
    label: Tensor,
    *,
    normalize: Callable[[Tensor], Tensor],
    num_classes: int,
    ignore_index: int,
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    config: SearchConfig,
) -> Dict[str, float]:
    """One ``AdvTrainer.wgan_loss`` step on a batch (an even number of [0, 1] images and
    their labels). Returns its losses."""
    if images01.shape[0] < 2 or images01.shape[0] % 2:
        raise ValueError(f"the search splits each batch in two halves; got {images01.shape[0]} images")
    main_opt, policy_opt = optimizers
    b = images01.shape[0] // 2
    a_images, a_label = images01[:b], label[:b]
    n_images, n_label = images01[b:], label[b:]
    a_mask = soft_mask(a_label, num_classes, ignore_index)
    n_mask = soft_mask(n_label, num_classes, ignore_index)
    align = getattr(model, "align_corners", False)

    # The critic, and the task head under it.
    model.requires_grad_(True)
    main_opt.zero_grad(set_to_none=True)
    real = normalize(n_images)
    logits, real_score = model(real)
    task_loss = config.task_factor * soft_cross_entropy(logits, n_mask, align).mean()
    (task_loss - real_score.mean()).backward()
    with torch.no_grad():
        augmented, _ = policy(a_images)
    fake = normalize(augmented)
    _, fake_score = model(fake)
    gp = config.gp_factor * gradient_penalty(model, real.detach(), fake)
    (fake_score.mean() + gp).backward()
    main_opt.step()

    # The policy, the critic frozen.
    model.requires_grad_(False)
    policy_opt.zero_grad(set_to_none=True)
    augmented, augmented_mask = policy(a_images, a_mask)
    logits, policy_score = model(normalize(augmented))
    policy_task = config.task_factor * soft_cross_entropy(logits, augmented_mask, align).mean()
    (policy_task - policy_score.mean()).backward()
    policy_opt.step()
    model.requires_grad_(True)

    with torch.no_grad():
        return {
            "task": task_loss.item(),
            "policy_task": policy_task.item(),
            "d_loss": (-real_score.mean() + fake_score.mean() + gp).item(),
            "gp": gp.item(),
            "a_loss": (-policy_score.mean()).item(),
        }


def save_policy(path: str, policy: Policy, **meta) -> str:
    """Write ``policy.export()`` as the host's policy file (``host.load_policy`` reads it
    back), atomically, with ``meta`` (strings and numbers) alongside. Returns ``path``."""
    exported = policy.export()
    fields = {
        "format": np.array(POLICY_FORMAT),
        "version": np.array(POLICY_VERSION),
        "ops": np.array(exported["ops"]),
        "weights": exported["weights"],
        "probabilities": exported["probabilities"],
        "magnitudes": exported["magnitudes"],
        "meta_temperature": np.array(policy.temperature),
        "meta_bernoulli_temperature": np.array(policy.bernoulli_temperature),
    }
    for key, value in meta.items():
        fields[f"meta_{key}"] = np.array(value if isinstance(value, (int, float, bool)) else str(value))
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=directory)
    os.close(fd)
    try:
        np.savez(tmp, **fields)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    load_policy(path)  # what was written must read back
    return path


def search(
    model: nn.Module,
    policy: Policy,
    epoch_batches: Callable[[int], Iterable[Tuple[Tensor, Tensor]]],
    *,
    normalize: Callable[[Tensor], Tensor],
    num_classes: int,
    ignore_index: int = 255,
    config: SearchConfig = SearchConfig(),
    out_dir: Optional[str] = None,
    meta: Optional[dict] = None,
    log: Callable[[str], None] = print,
) -> Policy:
    """Run the search for ``config.epochs`` epochs. ``epoch_batches(epoch)`` yields that
    epoch's ``(images01, label)`` batches, on the model's device. With ``out_dir``, the
    policy is saved after every epoch (``epoch_<N>.npz``) and at the end (``final.npz``),
    and every ``log_interval`` steps' mean losses go to ``search_log.jsonl``."""
    model.train()
    policy.train()
    main_opt = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=config.policy_lr, betas=config.betas)
    meta = dict(meta or {})
    log_path = None if out_dir is None else os.path.join(out_dir, "search_log.jsonl")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "search_config.json"), "w") as f:
            json.dump({"search": asdict(config), "meta": meta}, f, indent=2, default=str)

    step = 0
    for epoch in range(1, config.epochs + 1):
        window: Dict[str, list] = {}
        for images01, label in epoch_batches(epoch):
            losses = search_step(
                model,
                policy,
                images01,
                label,
                normalize=normalize,
                num_classes=num_classes,
                ignore_index=ignore_index,
                optimizers=(main_opt, policy_opt),
                config=config,
            )
            step += 1
            for key, value in losses.items():
                window.setdefault(key, []).append(value)
            if not all(np.isfinite(v) for v in losses.values()):
                raise FloatingPointError(f"non-finite search loss at epoch {epoch}, step {step}: {losses}")
            if step % config.log_interval == 0:
                record = {"epoch": epoch, "step": step, **{k: float(np.mean(v)) for k, v in window.items()}}
                log(json.dumps(record))
                if log_path is not None:
                    with open(log_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                window = {}
        if out_dir is not None:
            save_policy(os.path.join(out_dir, f"epoch_{epoch}.npz"), policy, epoch=epoch, step=step, **meta)
    if out_dir is not None:
        save_policy(os.path.join(out_dir, "final.npz"), policy, epoch=config.epochs, step=step, **meta)
    return policy
