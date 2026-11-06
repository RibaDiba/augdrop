"""DeepAA's policy search, layer by layer, by regularized gradient matching (Zheng et al.,
ICLR 2022, §3.3; ``DeepAA_search.py``), on a frozen model.

Notation for one search image x, at the layer being searched:

- ``l(.)`` the model's per-image loss; ``g_v`` the gradient of the mean loss over x's own
  validation batch; ``g_T`` the augmented training gradient (below);
- ``tangent = g_v - g_T (g_v . g_T) / |g_T|^2``, the part of g_v orthogonal to g_T;
- ``importance_t = (grad l(t(x)) . tangent) / (|g_T| |g_v|)``, which is
  ``d cos(g_v, g_T) / d p_t`` at layer 1, the JVP DeepAA takes
  (``step2_cal_JVP_jvpStep``).

Layer 1 (``train_policy_stage1``): every kept transform t applied to x (``get_dense_aug``);
``g_T = sum_t p_t grad l(t(x))``; the importance averaged over the step's images; the
policy loss ``-0.25 sum_t importance_t p_t``.

Layer k >= 2 (``train_policy_stage2``): ``g_T`` the mean gradient of ``EXP_gT =
exp_gt_factor |kept|`` copies of x each drawn through layers 1..k; ``EXP_G`` copies of x
drawn through layers 1..k-1, each then through every kept transform of layer k, their
importances averaged over the EXP_G; across the step's images the mean minus the standard
deviation (Eq. 10, c = 1; ``reduce_mean - reduce_std``, the population std); the policy
loss ``-n sum_t importance_t p_t``, n the step's images. DeepAA's code takes no std
penalty at layer 1 and neither does this (the paper's Eq. 10 reads as every layer's).

Each layer's logits take one Adam step a search step (lr 0.025, betas (0.9, 0.999), eps
1e-7: Keras' default, which DeepAA's ``tf.optimizers.Adam`` has). The model is frozen: no
weight, no BatchNorm statistic and no ``.grad`` moves (DeepAA restores a pretrained model,
sets BatchNorm untrainable and calls the model with ``training=False``). The gradients are
weighted backward passes in chunks (DeepAA's ``output_gradients`` over
``_PARALLEL_BATCH`` images); the JVP is forward-mode (``torch.func.jvp``, DeepAA's
``ForwardAccumulator``), or with ``jvp="finite_diff"`` a central difference along the
tangent, ``eps = fd_eps / |tangent|`` (for a model whose ops lack forward-mode support).

The images of one pass are regenerated, never stored: each chunk is drawn under its own
seed, so layer 1's gradient and JVP passes see the same images, random ops included
(DeepAA augments once and reuses them).

Framework-free: the caller gives ``Task`` the per-image loss on [0, 1] images, the model's
parameters and the op application (``tasks.augment.apply_draws`` over the vocab).
"""

import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .policy import LayerPolicy, export
from .space import draws as transform_draws

__all__ = ["MODES", "Search", "Settings", "Task", "image_importance", "pass_counts"]

Tensor = torch.Tensor
Params = Dict[str, Tensor]
#: (params, images01 (B, 3, H, W), labels (B, 1, H, W)) -> per-image losses (B,).
LossFn = Callable[[Params, Tensor, Tensor], Tensor]
#: (images01, labels, draws) -> (images01, labels): each image's (op, magnitude) applied.
Augment = Callable[[Tensor, Tensor, List[Tuple[str, float]]], Tuple[Tensor, Tensor]]

MODES = ("forward", "finite_diff")


@dataclass
class Settings:
    """The search's knobs; defaults are DeepAA's released settings (the README's
    commands and ``search_policy``'s call)."""

    layers: int = 5  # K: the published ImageNet policies' (the README searches 6)
    steps_per_layer: int = 1024  # --search_bno (the paper says 512)
    images_per_step: int = 16  # --train_same_labels: search images a step, over every rank
    val_per_image: int = 64  # search_policy(val_bs=64): each image's validation batch
    exp_g: int = 16  # --EXP_G: layer-k JVP copies per image
    exp_gt_factor: int = 4  # --EXP_gT_factor: layer-k gradient copies, times |kept|
    policy_lr: float = 0.025  # --policy_lr
    chunk: int = 16  # images a pass (_PARALLEL_BATCH_small, resnet50's)
    jvp: str = "forward"
    fd_eps: float = 0.01

    def __post_init__(self):
        for key in ("layers", "steps_per_layer", "images_per_step", "val_per_image", "exp_g", "exp_gt_factor", "chunk"):
            if int(getattr(self, key)) < 1:
                raise ValueError(f"search.{key} must be >= 1, got {getattr(self, key)}")
        if self.jvp not in MODES:
            raise ValueError(f"search.jvp must be one of {MODES}, got {self.jvp!r}")
        if not self.policy_lr > 0 or not self.fd_eps > 0:
            raise ValueError("search.policy_lr and search.fd_eps must be > 0")


def pass_counts(num_kept: int, settings: Settings) -> Dict[str, int]:
    """Image passes the whole search makes, over every rank: ``backward`` (forward +
    backward, the gradients) and ``jvp`` (the importances), at ``num_kept`` transforms."""
    s, n, v = settings.steps_per_layer, settings.images_per_step, settings.val_per_image
    first = s * n * (v + num_kept)
    later = (settings.layers - 1) * s * n * (v + settings.exp_gt_factor * num_kept)
    jvp = s * n * num_kept + (settings.layers - 1) * s * n * settings.exp_g * num_kept
    return {"backward": int(first + later), "jvp": int(jvp)}


def _norm(grads: Sequence[Tensor]) -> Tensor:
    return torch.sqrt(sum((g.double() ** 2).sum() for g in grads))


def _dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    return sum((x.double() * y.double()).sum() for x, y in zip(a, b))


class Task:
    """The frozen model as the search sees it (module docstring)."""

    def __init__(
        self,
        loss_fn: LossFn,
        params: Params,
        augment: Augment,
        *,
        chunk: int = 16,
        jvp: str = "forward",
        fd_eps: float = 0.01,
    ):
        if jvp not in MODES:
            raise ValueError(f"jvp must be one of {MODES}, got {jvp!r}")
        self.loss_fn = loss_fn
        self.names = list(params)
        self.primal = {k: v.detach() for k, v in params.items()}
        self.augment = augment
        self.chunk = int(chunk)
        self.jvp = jvp
        self.fd_eps = float(fd_eps)

    def grad(self, batches: Iterable[Tuple[Tensor, Tensor, Tensor]]) -> List[Tensor]:
        """``sum_i w_i grad l(x_i)`` over ``(images, labels, weights)`` batches."""
        leaves = {k: v.detach().requires_grad_(True) for k, v in self.primal.items()}
        total = None
        for images, labels, weights in batches:
            loss = (self.loss_fn(leaves, images, labels) * weights.to(images.dtype)).sum()
            grads = torch.autograd.grad(loss, [leaves[k] for k in self.names], allow_unused=True)
            grads = [torch.zeros_like(leaves[k]) if g is None else g.detach() for k, g in zip(self.names, grads)]
            total = grads if total is None else [a + b for a, b in zip(total, grads)]
        if total is None:
            raise ValueError("no images to take a gradient over")
        return total

    def directional(self, tangent: Sequence[Tensor], images: Tensor, labels: Tensor) -> Tensor:
        """(B,): each image's ``grad l . tangent``."""
        if self.jvp == "forward":
            tangents = dict(zip(self.names, (t.to(self.primal[k].dtype) for k, t in zip(self.names, tangent))))
            _, out = torch.func.jvp(lambda p: self.loss_fn(p, images, labels), (self.primal,), (tangents,))
            return out.detach()
        eps = self.fd_eps / float(_norm(tangent).clamp_min(1e-30))
        with torch.no_grad():
            plus = {k: self.primal[k] + eps * t.to(self.primal[k].dtype) for k, t in zip(self.names, tangent)}
            minus = {k: self.primal[k] - eps * t.to(self.primal[k].dtype) for k, t in zip(self.names, tangent)}
            return (self.loss_fn(plus, images, labels) - self.loss_fn(minus, images, labels)) / (2.0 * eps)

    def apply(self, images: Tensor, labels: Tensor, stages: Sequence[List[Tuple[str, float]]], seed: int):
        """``stages`` of per-image draws applied in turn under ``seed`` (the random ops'
        torch RNG), so a regenerated batch is the same batch."""
        devices = [images.device] if images.device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            for draws in stages:
                images, labels = self.augment(images, labels, draws)
        return images, labels


def _repeat(x: Tensor, n: int) -> Tensor:
    return x.expand(n, *x.shape[1:]).contiguous()


def _val_grad(task: Task, vals: Sequence[Tuple[Tensor, Tensor]]) -> List[Tensor]:
    w = torch.full((1,), 1.0 / len(vals))
    return task.grad((x, y, w.to(x.device)) for x, y in vals)


def _dense(task: Task, prefix: Tuple[Tensor, Tensor], draws: List[Tuple[str, float]], seed: int):
    """Chunks ``(images, labels, span)`` of the one prefix image through each of ``draws``
    (one per kept transform), each chunk under its own seed."""
    x, y = prefix
    for c, start in enumerate(range(0, len(draws), task.chunk)):
        part = draws[start : start + task.chunk]
        images, labels = task.apply(_repeat(x, len(part)), _repeat(y, len(part)), [part], seed + c)
        yield images, labels, slice(start, start + len(part))


def _sampled(
    task: Task, x: Tensor, y: Tensor, layers: Sequence[LayerPolicy], n: int, rng: np.random.Generator,
    generator: torch.Generator, seed: int,
):
    """Chunks ``(images, labels)`` of ``n`` copies of x, each drawn through ``layers`` in
    turn (each layer's transform from its policy)."""
    ids = [layer.sample(n, generator) for layer in layers]
    for c, start in enumerate(range(0, n, task.chunk)):
        stop = min(n, start + task.chunk)
        stages = [transform_draws(layer_ids[start:stop], rng) for layer_ids in ids]
        yield task.apply(_repeat(x, stop - start), _repeat(y, stop - start), stages, seed + c)


def image_importance(
    task: Task,
    layers: Sequence[LayerPolicy],
    k: int,
    x: Tensor,
    y: Tensor,
    vals: Sequence[Tuple[Tensor, Tensor]],
    settings: Settings,
    rng: np.random.Generator,
    generator: torch.Generator,
    seed: int,
) -> Tuple[Tensor, float]:
    """One search image's importances (|kept|,) for layer ``k`` (0-based), and its
    cos(g_v, g_T) (module docstring). ``x`` (1, 3, H, W) in [0, 1], ``y`` (1, 1, H, W);
    ``vals`` its validation batch, ``(image, label)`` pairs of one image each."""
    layer = layers[k]
    kept = layer.kept.cpu().numpy()
    g_val = _val_grad(task, vals)
    if k == 0:
        draws = transform_draws(kept, rng)
        probs = layer.probs().detach().to(x.device)
        g_train = task.grad((im, lb, probs[span]) for im, lb, span in _dense(task, (x, y), draws, seed))
    else:
        n = settings.exp_gt_factor * len(kept)
        w = torch.full((1,), 1.0 / n, device=x.device)
        g_train = task.grad(
            (im, lb, w.expand(im.shape[0])) for im, lb in _sampled(task, x, y, layers[: k + 1], n, rng, generator, seed)
        )
    norm_v, norm_t = _norm(g_val), _norm(g_train)
    vt = _dot(g_val, g_train)
    scale = vt / norm_t**2
    tangent = [(gv.double() - gt.double() * scale).to(gv.dtype) for gv, gt in zip(g_val, g_train)]
    denom = norm_t * norm_v

    if k == 0:
        out = torch.cat([task.directional(tangent, im, lb) for im, lb, _ in _dense(task, (x, y), draws, seed)])
        importance = out.double() / denom
    else:
        prefixes = list(_sampled(task, x, y, layers[:k], settings.exp_g, rng, generator, seed + 1_000_003))
        prefix_x = torch.cat([im for im, _ in prefixes])
        prefix_y = torch.cat([lb for _, lb in prefixes])
        rows = []
        for g in range(prefix_x.shape[0]):
            draws = transform_draws(kept, rng)
            row = torch.cat(
                [
                    task.directional(tangent, im, lb)
                    for im, lb, _ in _dense(task, (prefix_x[g : g + 1], prefix_y[g : g + 1]), draws, seed + 7919 * (g + 1))
                ]
            )
            rows.append(row.double())
        importance = torch.stack(rows).mean(dim=0) / denom
    return importance.float().cpu(), float((vt / denom).item())


class Search:
    """The K layers' policies and their Adam, stepped one search step at a time (module
    docstring). ``state_dict`` / ``load_state_dict`` resume it mid-layer."""

    def __init__(self, kept: Sequence[int], settings: Settings):
        self.settings = settings
        self.layers = nn.ModuleList([LayerPolicy(kept) for _ in range(int(settings.layers))])
        self.optimizers = [
            torch.optim.Adam(layer.parameters(), lr=settings.policy_lr, betas=(0.9, 0.999), eps=1e-7)
            for layer in self.layers
        ]
        self.layer = 0
        self.step = 0

    @property
    def kept(self) -> np.ndarray:
        return self.layers[0].kept.cpu().numpy()

    @property
    def done(self) -> bool:
        return self.layer >= len(self.layers)

    def param_count(self) -> int:
        """The policy's numbers: K x |kept|."""
        return sum(layer.logits.numel() for layer in self.layers)

    def update(self, importance: Tensor) -> Dict[str, float]:
        """One Adam step of the current layer on the step's importances (n, |kept|), every
        rank's images; moves to the next layer after ``steps_per_layer``."""
        k = self.layer
        importance = importance.double()
        n = importance.shape[0]
        if k == 0:
            combined, mult = importance.mean(dim=0), 0.25
        else:
            combined, mult = importance.mean(dim=0) - importance.std(dim=0, unbiased=False), float(n)
        layer, optimizer = self.layers[k], self.optimizers[k]
        optimizer.zero_grad(set_to_none=True)
        loss = -(combined.to(layer.logits.dtype) * layer.probs()).sum() * mult
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        stats = {
            "layer": k,
            "step": self.step,
            "policy_loss": float(loss.detach()),
            "entropy": layer.entropy(),
            "identity": float(layer.probs()[0].detach()),
            "importance_mean": float(combined.mean()),
        }
        self.step += 1
        if self.step >= self.settings.steps_per_layer:
            self.layer, self.step = self.layer + 1, 0
        return stats

    def export(self) -> dict:
        return export(self.layers)

    def state_dict(self) -> dict:
        return {
            "settings": asdict(self.settings),
            "kept": self.kept.tolist(),
            "layer": self.layer,
            "step": self.step,
            "layers": [layer.state_dict() for layer in self.layers],
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state: dict) -> None:
        if list(state["kept"]) != self.kept.tolist():
            raise ValueError("the saved search ran over another space (its kept transforms differ)")
        if int(len(state["layers"])) != len(self.layers):
            raise ValueError(f"the saved search has {len(state['layers'])} layers, this one {len(self.layers)}")
        for layer, saved in zip(self.layers, state["layers"]):
            layer.load_state_dict(saved)
        for opt, saved in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(saved)
        self.layer, self.step = int(state["layer"]), int(state["step"])


def run_step(
    task: Task,
    search: Search,
    images: Tensor,
    labels: Tensor,
    vals: Sequence[Sequence[Tuple[Tensor, Tensor]]],
    rng: np.random.Generator,
    generator: torch.Generator,
    seed: int,
) -> Tuple[Tensor, List[float], float]:
    """This rank's part of one search step: each of its images' importances (n, |kept|)
    and cos, at the current layer, and the seconds it took. The caller gathers every
    rank's rows and calls ``search.update``."""
    start = time.time()
    rows, cos = [], []
    for i in range(images.shape[0]):
        importance, c = image_importance(
            task, search.layers, search.layer, images[i : i + 1], labels[i : i + 1], vals[i],
            search.settings, rng, generator, seed + 104_729 * i,
        )
        rows.append(importance)
        cos.append(c)
    return torch.stack(rows), cos, time.time() - start
