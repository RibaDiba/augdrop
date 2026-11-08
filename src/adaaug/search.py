"""AdaAug's search, for segmentation: ``ada_aug/search.py`` with no training framework.

The search model gf (a segmentor here) and the policy h train together. For each training
batch, as the code's ``train`` loop takes it:

1. **exploit**: h (on gf's feature, gf in eval) draws one op per image at its λ, at
   temperature 1 and no δ (``adaaug_config`` hard-codes k = 1, δ = 0, T = 1:
   ``hosts.adaaug.SEARCH_EXPLOIT``); gf trains on the batch so augmented (train mode, the
   cross-entropy, gradient clipped to norm 5, its optimizer and a cosine schedule)
2. **explore**, at steps 0, ``search_freq``, 2 ``search_freq``, ... of each epoch (the
   code's ``step % search_freq == 0``, the step counted within the epoch): a fresh random
   batch of ``explore_batch`` held-out images (``next(iter(search_queue))``, a new
   shuffled iterator each time), h's loss on them (``explore.explore_loss``), and one Adam
   step of h. gf gets no gradient from it (the code zeroes gf's before its next step)

Defaults are ``scripts/search.sh``'s reduced CIFAR-10 run: 200 epochs, h on Adam at lr
1e-3 with weight decay 1e-3 (``--proj_weight_decay``'s default), betas (0.9, 0.999), an
explore every 3 steps; gf on SGD at lr 0.1 with cosine decay to 0.001 there, here the
segmentor's own recipe optimizer (built by ``scripts/adaaug_search.py``) decayed by
cosine to ``lr_min_ratio`` (0.01, the code's 0.001 / 0.1) of its lr. The explore batch is
``get_search_divider``'s: 32 images for WRN-40-2, 16 for any other model (the default
here). The code's Cutout(16) after the ops is not added (the recipe's pipeline is the
base augmentation).

What a run writes to ``out_dir``:

- ``state.pt`` after every epoch: gf, h, both optimizers, the schedule, the epoch. A
  search that finds one resumes after its epoch (each epoch's randomness is seeded by
  ``(seed, epoch)``, so a resumed search makes the draws the uninterrupted one would)
- ``epoch_<N>/`` every ``save_every`` epochs and ``final/`` at the end: ``gf.pth`` (the
  segmentor's state dict), ``h.pth`` (h's) and ``meta.json`` (the outputs searched, the
  ops left out, h's shape and size, the explore's copies per image, and the caller's
  ``meta``: what built gf), the meta last, so a ``meta.json`` marks a finished save.
  ``hosts.adaaug.load_meta`` reads it; ``tasks/mmseg/adaaug_hooks.py`` loads gf and h
- ``search_log.jsonl``: every ``log_interval`` steps the mean exploit and explore losses,
  ``exploit_seconds`` and ``explore_seconds`` (the explore's share is what a prune cuts),
  the explore's copies, h's parameters, and h's mean p and λ per output on the explore
  images (the code logs them per class, ``add_history``)
- ``search_config.json``
"""

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from hosts.adaaug import FINAL_DIR, META_FILE, AdaAugHost
from madaug.policy import Projection, eval_mode, predict
from madaug.relax import soft_cross_entropy
from tasks.augment import apply_slots, forbidden_ops

from .explore import explore_loss, pooled

__all__ = [
    "SearchConfig",
    "exploit_step",
    "explore_step",
    "pass_counts",
    "policy_params",
    "save_search",
    "search",
    "segmentor_logits",
]

STATE_FILE = "state.pt"


@dataclass
class SearchConfig:
    """The search's knobs, ``search.sh``'s reduced CIFAR-10 defaults (module docstring)."""

    epochs: int = 200
    search_freq: int = 3
    explore_batch: int = 16
    policy_lr: float = 1e-3
    policy_weight_decay: float = 1e-3
    grad_clip: float = 5.0
    lr_min_ratio: float = 0.01
    save_every: int = 20
    log_interval: int = 20
    seed: int = 0


def policy_params(feature_dim: int, n_outputs: int, n_layers: int = 0, n_hidden: int = 128) -> int:
    """h's parameters (``madaug.Projection``): ``(F + 1) 2 n`` for a linear h over n
    outputs on an F-wide feature."""
    return sum(p.numel() for p in Projection(feature_dim, n_layers, n_hidden, n_outputs).parameters())


def pass_counts(
    *, train_images: int, batch_size: int, epochs: int, search_freq: int, explore_batch: int, n_outputs: int
) -> Dict[str, int]:
    """What a search costs, in images through gf: ``exploit`` (forward and backward, a
    batch a step), ``explores`` (h's updates), ``explore_copies`` (forward and backward
    through f, ``n_outputs`` a held-out image), ``feature_passes`` (gf's forward under
    no_grad, for h's input: every exploit image and every explore image)."""
    steps = train_images // batch_size  # drop_last
    explores = math.ceil(steps / search_freq) * epochs
    return {
        "steps": steps * epochs,
        "exploit_images": steps * batch_size * epochs,
        "explores": explores,
        "explore_copies": explores * explore_batch * n_outputs,
        "feature_passes": steps * batch_size * epochs + explores * explore_batch,
    }


def segmentor_logits(segmentor, inputs: Tensor, size) -> Tensor:
    """The decode head's logits for normalized ``inputs``, resized to ``size`` as mmseg's
    loss resizes them."""
    logits = segmentor.decode_head.forward(segmentor.extract_feat(inputs))
    return nn.functional.interpolate(
        logits, size=tuple(size), mode="bilinear", align_corners=bool(getattr(segmentor, "align_corners", False))
    )


def exploit_step(
    segmentor,
    policy: Projection,
    images01: Tensor,
    label: Tensor,
    host: AdaAugHost,
    ops: Dict,
    normalize: Callable[[Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    *,
    grad_clip: float = 5.0,
    ignore_index: int = 255,
) -> float:
    """One step of gf on the batch as h augments it (module docstring, step 1). Returns
    the loss."""
    with torch.no_grad(), eval_mode(segmentor):
        levels, p = predict(policy, pooled(segmentor, normalize(images01)), host.temperature)
    slots = host.draw(p.double().cpu().numpy(), levels.double().cpu().numpy(), rng)
    augmented, augmented_label = apply_slots(images01, label, slots, ops, forbidden_ops(host))
    segmentor.train()
    optimizer.zero_grad(set_to_none=True)
    logits = segmentor_logits(segmentor, normalize(augmented), augmented_label.shape[-2:])
    weights = torch.ones(images01.shape[0], 1, dtype=logits.dtype, device=logits.device)
    loss = soft_cross_entropy(logits, [augmented_label], weights, ignore_index)
    loss.backward()
    nn.utils.clip_grad_norm_([q for q in segmentor.parameters() if q.requires_grad], grad_clip)
    optimizer.step()
    return float(loss.detach())


def explore_step(
    segmentor,
    policy: Projection,
    images01: Tensor,
    label: Tensor,
    host: AdaAugHost,
    ops: Dict,
    normalize: Callable[[Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    *,
    ignore_index: int = 255,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """One update of h on a held-out batch (module docstring, step 2): each image's
    ``explore_loss`` over the batch's size, backpropagated on its own, then one step.
    Returns the batch's loss and h's mean p and λ per output on it."""
    optimizer.zero_grad(set_to_none=True)
    batch = images01.shape[0]
    total, ps, lvs = 0.0, [], []
    requires = [q.requires_grad for q in segmentor.parameters()]
    segmentor.requires_grad_(False)
    try:
        with eval_mode(segmentor):
            with torch.no_grad():
                features = pooled(segmentor, normalize(images01))
            for i in range(batch):
                loss, seen = explore_loss(
                    segmentor,
                    policy,
                    features[i : i + 1],
                    images01[i : i + 1],
                    label[i : i + 1],
                    host,
                    ops,
                    normalize,
                    rng,
                    ignore_index,
                )
                (loss / batch).backward()
                total += float(loss.detach()) / batch
                ps.append(seen["p"])
                lvs.append(seen["levels"])
    finally:
        for q, r in zip(segmentor.parameters(), requires):
            q.requires_grad_(r)
    optimizer.step()
    return total, {"p": np.mean(ps, axis=0), "levels": np.mean(lvs, axis=0)}


def _cosine(total_steps: int, lr_min_ratio: float) -> Callable[[int], float]:
    """The cosine factor on the optimizer's lr, 1 at step 0 to ``lr_min_ratio`` at
    ``total_steps`` (``CosineAnnealingLR`` with ``eta_min = lr_min_ratio * lr``)."""
    total = max(1, int(total_steps))

    def factor(step: int) -> float:
        return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))

    return factor


def _atomic(path: str, write: Callable[[str], None]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=os.path.splitext(path)[1])
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_search(directory: str, segmentor, policy: Projection, meta: dict) -> str:
    """``gf.pth``, ``h.pth`` and ``meta.json`` in ``directory`` (module docstring), the
    meta last. Returns ``directory``."""
    _atomic(os.path.join(directory, "gf.pth"), lambda p: torch.save(segmentor.state_dict(), p))
    _atomic(os.path.join(directory, "h.pth"), lambda p: torch.save(policy.state_dict(), p))

    def write_meta(path):
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    _atomic(os.path.join(directory, META_FILE), write_meta)
    return directory


def search(
    segmentor,
    policy: Projection,
    host: AdaAugHost,
    ops: Dict,
    epoch_batches: Callable[[int], Iterable[Tuple[Tensor, Tensor]]],
    explore_batch: Callable[[int, int], Tuple[Tensor, Tensor]],
    *,
    normalize: Callable[[Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    config: SearchConfig = SearchConfig(),
    out_dir: Optional[str] = None,
    meta: Optional[dict] = None,
    ignore_index: int = 255,
    log: Callable[[str], None] = print,
) -> Projection:
    """Run the search (module docstring). ``epoch_batches(epoch)`` yields that epoch's
    training ``(images01, label)`` batches and ``explore_batch(epoch, k)`` the k-th
    explore's held-out batch, both on the segmentor's device. ``host`` is an
    ``AdaAugHost`` at the starting policy, pruned as the search is (its outputs are
    ``policy``'s). ``optimizer`` is gf's; h's Adam is made here."""
    if tuple(host.outputs) and policy.n_outputs != len(host.outputs):
        raise ValueError(f"the policy has {policy.n_outputs} outputs, the host {len(host.outputs)}")
    if host.searched:
        raise ValueError("the search starts from the starting policy (policy_file: initial)")
    policy_opt = torch.optim.Adam(
        policy.parameters(), lr=config.policy_lr, betas=(0.9, 0.999), weight_decay=config.policy_weight_decay
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _cosine(steps_per_epoch * config.epochs, config.lr_min_ratio)
    )
    params = sum(p.numel() for p in policy.parameters())
    meta = dict(meta or {})
    meta.update(
        {
            "outputs": list(host.outputs),
            "pruned_ops": list(host.pruned()),
            "n_layers": policy.n_layers,
            "policy_params": params,
            "explore_copies_per_image": len(host.outputs),
            "search": asdict(config),
        }
    )
    log_path = None if out_dir is None else os.path.join(out_dir, "search_log.jsonl")
    start = 1
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "search_config.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        state_path = os.path.join(out_dir, STATE_FILE)
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if state.get("outputs") != list(host.outputs) or state.get("search") != asdict(config):
                raise ValueError(f"{state_path} is a search of other outputs or settings; use another --output-dir")
            segmentor.load_state_dict(state["gf"])
            policy.load_state_dict(state["h"])
            optimizer.load_state_dict(state["gf_optimizer"])
            policy_opt.load_state_dict(state["h_optimizer"])
            schedule.load_state_dict(state["schedule"])
            start = int(state["epoch"]) + 1
            log(f"resuming after epoch {state['epoch']} ({state_path})")
    log(
        f"policy over {len(host.outputs)} outputs ({len(host.pruned())} vocab ops left out): {params} parameters, "
        f"{len(host.outputs)} explore copies per held-out image"
    )

    step = (start - 1) * steps_per_epoch
    for epoch in range(start, config.epochs + 1):
        torch.manual_seed(config.seed * 100003 + epoch)
        rng = np.random.default_rng([config.seed, epoch])
        window: Dict[str, list] = {}
        n_explore = 0
        for index, (images01, label) in enumerate(epoch_batches(epoch)):
            started = time.perf_counter()
            exploit = exploit_step(
                segmentor, policy, images01, label, host, ops, normalize, optimizer, rng,
                grad_clip=config.grad_clip, ignore_index=ignore_index,
            )
            schedule.step()
            exploit_seconds = time.perf_counter() - started
            record = {"exploit": exploit, "exploit_seconds": exploit_seconds}
            if index % config.search_freq == 0:
                started = time.perf_counter()
                images, labels = explore_batch(epoch, n_explore)
                n_explore += 1
                explore, seen = explore_step(
                    segmentor, policy, images, labels, host, ops, normalize, policy_opt, rng, ignore_index=ignore_index
                )
                record.update({"explore": explore, "explore_seconds": time.perf_counter() - started})
                for k, op in enumerate(host.outputs):
                    record[f"p.{op}"] = float(seen["p"][k])
                    record[f"level.{op}"] = float(seen["levels"][k])
            step += 1
            if not all(np.isfinite(v) for v in record.values()):
                raise FloatingPointError(f"non-finite search loss at epoch {epoch}, step {step}: {record}")
            for key, value in record.items():
                window.setdefault(key, []).append(value)
            if step % config.log_interval == 0:
                line = {
                    "epoch": epoch,
                    "step": step,
                    **{k: float(np.mean(v)) for k, v in window.items()},
                    "explores_in_window": len(window.get("explore", [])),
                    "explore_copies": len(window.get("explore", [])) * config.explore_batch * len(host.outputs),
                    "policy_params": params,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                log(json.dumps(line))
                if log_path is not None:
                    with open(log_path, "a") as f:
                        f.write(json.dumps(line) + "\n")
                window = {}
        if out_dir is not None:
            state = {
                "gf": segmentor.state_dict(),
                "h": policy.state_dict(),
                "gf_optimizer": optimizer.state_dict(),
                "h_optimizer": policy_opt.state_dict(),
                "schedule": schedule.state_dict(),
                "epoch": epoch,
                "outputs": list(host.outputs),
                "search": asdict(config),
            }
            _atomic(os.path.join(out_dir, STATE_FILE), lambda p: torch.save(state, p))
            if config.save_every > 0 and epoch % config.save_every == 0 and epoch < config.epochs:
                save_search(os.path.join(out_dir, f"epoch_{epoch}"), segmentor, policy, {**meta, "epoch": epoch, "step": step})
    if out_dir is not None:
        save_search(os.path.join(out_dir, FINAL_DIR), segmentor, policy, {**meta, "epoch": config.epochs, "step": step})
    return policy
