"""mmengine glue for Deep AutoAugment's search in the run. Nothing is computed here.

    DeepAASearchHook   at the pause (``at``, before the first step after it): the search
                       space is the transforms the run's prune leaves (the static prune
                       plus PruneHook's latched M, ``deepaa.kept_transforms``); DeepAA's
                       K-layer search (``deepaa.Search``, ``deepaa.run_step``) on the
                       frozen model; the policy saved (``<work_dir>/deepaa_search/
                       final.npz``) and installed on the host (``install_policy``).
                       Training then resumes on it: same model, same schedule.

The recipe (the user's, 2026-11-05: train, pause, prune, search, resume): the host starts
with ``policy_file: search`` and draws nothing, as DeepAA's ``pretrain()`` trains on its
default augmentation only; the probe at ``at`` gives NMS its R and PruneHook latches M in
``after_train_iter``; CheckpointHook saves ``iter_<at>.pth``; then this hook searches.

The search images are the train split through its pipeline and the model's preprocessor
(augmentation suspended), as the images a training step hands the host; each search image
gets its own validation batch of ``val_per_image`` clean images from the val split, one at
a time (MADAug's split; DeepAA pairs each with images of its class from a held-out part
of train, and a segmentation image has no class: random images, as the paper's text has
it). The per-image loss is the decode head's logits (``functional_logits``, resized to the
label as mmseg's loss resizes them) under cross-entropy averaged over the image's
non-ignore pixels; an auxiliary head is not in it. The model runs in eval and functionally
on the unwrapped module: no weight, BatchNorm statistic or ``.grad`` moves, and no DDP hook
fires.

DDP: each rank searches ``images_per_step / world`` of a step's images (DeepAA's
MirroredStrategy splits them the same way), the importances are all-gathered, and every
rank takes the same Adam step. The search's state (the layers' logits and Adam, the layer
and step) is saved every ``save_every`` steps to ``deepaa_search/state.pt``: a job stopped
mid-search resumes from ``iter_<at>.pth`` and the search continues from that state. Its
loaders are not fast-forwarded (that would load every image the steps before it used);
they restart seeded by the step they resume at, so a resumed search sees fresh, still
reproducible batches. The installed policy goes into every later checkpoint
(``deepaa_policy``), so a resume past the pause loads it and does not search again.

Checked on stand-ins for mmengine and mmseg only (no mmengine in ``sensAug``).
"""

import copy
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import all_gather, broadcast_object_list, get_rank, get_world_size, is_main_process
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS

import deepaa
import madaug
from hosts.deep_autoaugment import SEARCH
from tasks.augment import apply_draws

from .backend import MMSegBackend, functional_logits, unwrap
from .hooks import host_of
from .transforms import _gather_labels

__all__ = ["CHECKPOINT_KEY", "DeepAASearchHook", "SEARCH_DIR"]

#: Where the installed policy lives in a checkpoint.
CHECKPOINT_KEY = "deepaa_policy"

#: The search's directory in the work dir.
SEARCH_DIR = "deepaa_search"


def per_image_loss(model, normalize, ignore_index: int):
    """``(params, images01, labels) -> (B,)``: each image's cross-entropy over its
    non-ignore pixels, the segmentor's parameters replaced by ``params``."""

    def loss(params, images01, labels):
        logits = functional_logits(model, params, normalize(images01), labels.shape[-2:])
        target = labels[:, 0].long()
        ce = F.cross_entropy(logits.float(), target, ignore_index=ignore_index, reduction="none")
        valid = (target != ignore_index).to(ce.dtype)
        return (ce * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)

    return loss


class _Vals:
    """One search image's validation batch, ``n`` clean ``(image, label)`` pairs loaded
    one at a time as the gradient iterates over them (64 full-size images per search
    image would not fit at once)."""

    def __init__(self, loader, backend, n: int):
        self.loader, self.backend, self.n = loader, backend, int(n)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for _ in range(self.n):
            yield self.backend.to_unit(next(self.loader))


@HOOKS.register_module()
class DeepAASearchHook(Hook):
    """Deep AutoAugment's search at the pause, then training on its policy (module
    docstring). ``at`` is the completed iteration the search follows; the rest are
    ``deepaa.Settings`` (DeepAA's released settings by default) plus ``save_every`` and
    ``seed``."""

    priority = "ABOVE_NORMAL"

    def __init__(self, at: int, save_every: int = 32, seed: int = 0, **settings):
        if int(at) < 1:
            raise ValueError(f"at must be a completed iteration >= 1, got {at}")
        self.at = int(at)
        self.save_every = max(1, int(save_every))
        self.seed = int(seed)
        self.settings = deepaa.Settings(**settings)

    def before_run(self, runner) -> None:
        self.host = host_of(runner)
        if not hasattr(self.host, "install_policy"):
            raise TypeError(f"host {self.host.name!r} takes no searched policy; DeepAASearchHook drives deep_autoaugment")
        if self.host.policy_file != SEARCH:
            raise ValueError(f"DeepAASearchHook needs host.policy_file={SEARCH!r}, got {self.host.policy_file!r}")
        model = unwrap(runner.model)
        self.dp = model.data_preprocessor
        self.dir = os.path.join(runner.work_dir, SEARCH_DIR)
        world = get_world_size()
        if self.settings.images_per_step % world:
            raise ValueError(
                f"search.images_per_step={self.settings.images_per_step} must split evenly over {world} ranks"
            )
        counts = deepaa.pass_counts(len(deepaa.kept_transforms(self.host.pruned())), self.settings)
        print_log(
            f"[deepaa] the search follows iteration {self.at}: {self.settings.layers} layers x "
            f"{self.settings.steps_per_layer} steps, {self.settings.images_per_step} images a step "
            f"({self.settings.images_per_step // world} a rank); over the unpruned space about "
            f"{counts['backward']:,} image backward passes and {counts['jvp']:,} JVPs "
            f"(fewer after a prune: they scale with the kept transforms)",
            logger="current",
        )

    # --- checkpoints ------------------------------------------------------------------------

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        if not self.host.pending:
            checkpoint[CHECKPOINT_KEY] = {k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in self.host.policy().items()}

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        state = checkpoint.get(CHECKPOINT_KEY)
        if state is None:
            return
        policy = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in state.items()}
        self.host.install_policy(policy)
        print_log(f"[deepaa] the checkpoint's searched policy installed (iter {runner.iter})", logger="current")

    # --- the search ---------------------------------------------------------------------------

    def before_train_iter(self, runner, batch_idx, data_batch=None) -> None:
        if int(runner.iter) != self.at or not self.host.pending:
            return
        self._search(runner)

    def _loader(self, runner, split: str, batch_size: int, offset: int):
        """An endless shuffled loader over ``split``, seeded by the search's seed, the rank
        and ``offset`` (the step the search starts or resumes at)."""
        seed = (self.seed + 1_009 * int(offset)) % (2**31 - 1)
        cfg = copy.deepcopy(runner.cfg.get(f"{split}_dataloader"))
        cfg["batch_size"] = int(batch_size)
        cfg["sampler"] = dict(type="InfiniteSampler", shuffle=True, seed=seed)
        cfg["persistent_workers"] = False
        return iter(runner.build_dataloader(cfg, seed=seed + 7 * get_rank()))

    def _search(self, runner) -> None:
        start = time.time()
        latched = getattr(runner, "latched_prune", ()) or ()
        pruned = tuple(n for n in self.host.vocabulary() if n in set(self.host.pruned()) | set(latched))
        kept = deepaa.kept_transforms(pruned)
        search = deepaa.Search(kept, self.settings)
        state_path = os.path.join(self.dir, "state.pt")
        if os.path.exists(state_path):
            search.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
            print_log(f"[deepaa] resuming the search at layer {search.layer + 1}, step {search.step}", logger="current")
        os.makedirs(self.dir, exist_ok=True)
        print_log(
            f"[deepaa] pause at iteration {self.at}: searching {len(kept)} of {deepaa.NUM_TRANSFORMS} transforms "
            f"(pruned {list(pruned)}), {search.param_count()} policy logits",
            logger="current",
        )

        model = unwrap(runner.model)
        mean, std = self.dp.mean.view(1, -1, 1, 1), self.dp.std.view(1, -1, 1, 1)
        normalize = lambda images01: (images01 * 255.0 - mean) / std  # noqa: E731
        ignore = int(getattr(model.decode_head, "ignore_index", 255))
        params = {k: v for k, v in model.named_parameters() if v.requires_grad}
        task = deepaa.Task(
            per_image_loss(model, normalize, ignore),
            params,
            lambda x, y, draws: apply_draws(x, y, draws, self.dp.ops),
            chunk=self.settings.chunk,
            jvp=self.settings.jvp,
            fd_eps=self.settings.fd_eps,
        )
        backend = MMSegBackend.from_runner(runner, split="val")
        per_rank = self.settings.images_per_step // get_world_size()
        offset = search.layer * self.settings.steps_per_layer + search.step
        train_iter = self._loader(runner, "train", per_rank, offset)
        val_iter = self._loader(runner, "val", 1, offset)
        rank = get_rank()
        log_path = os.path.join(self.dir, "search_log.jsonl")

        with madaug.eval_mode(model):
            while not search.done:
                step_seed = (self.seed * 1_000_003 + 10_007 * search.layer + search.step) % (2**31 - 1)
                with self.dp.suspended():
                    data = self.dp(next(train_iter), training=True)
                images01 = ((data["inputs"] * std + mean) / 255.0).clamp(0.0, 1.0)
                labels = _gather_labels(data.get("data_samples") or [])
                vals = [_Vals(val_iter, backend, self.settings.val_per_image) for _ in range(per_rank)]
                rng = np.random.default_rng([step_seed, rank])
                generator = torch.Generator().manual_seed(step_seed * 31 + rank)
                rows, cos, seconds = deepaa.run_step(task, search, images01, labels, vals, rng, generator, step_seed + 997 * rank)
                importance = torch.cat([r.cpu() for r in all_gather(rows.to(images01.device))])
                cos = [c for part in all_gather(torch.tensor(cos, device=images01.device)) for c in part.cpu().tolist()]
                stats = search.update(importance)
                stats.update(cos_sim=float(np.mean(cos)), step_seconds=seconds, policy_params=search.param_count(), kept=len(kept))
                if is_main_process():
                    with open(log_path, "a") as f:
                        f.write(json.dumps(stats) + "\n")
                    if search.done or (search.step % self.save_every == 0):
                        torch.save(search.state_dict(), state_path + ".tmp")
                        os.replace(state_path + ".tmp", state_path)
                runner.message_hub.update_scalar("train/deepaa_search_cos_sim", stats["cos_sim"])

        policy = search.export()
        path = os.path.join(self.dir, "final.npz")
        if is_main_process():
            deepaa.save_npz(path, policy, pruned)
            with open(os.path.join(self.dir, "search_config.json"), "w") as f:
                json.dump(
                    {
                        "at": self.at,
                        "settings": search.state_dict()["settings"],
                        "pruned_ops": list(pruned),
                        "kept_transforms": kept.tolist(),
                        "policy_params": search.param_count(),
                        "unpruned_policy_params": self.settings.layers * deepaa.NUM_TRANSFORMS,
                        "seconds": time.time() - start,
                    },
                    f,
                    indent=1,
                )
        # Every rank installs rank 0's policy (each took the same steps; this rules out drift).
        shared = [policy]
        broadcast_object_list(shared, src=0)
        self.host.install_policy(shared[0])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        seconds = time.time() - start
        runner.message_hub.update_scalar("train/deepaa_search_seconds", seconds)
        runner.message_hub.update_scalar("train/deepaa_policy_params", search.param_count())
        print_log(
            f"[deepaa] searched in {seconds / 3600:.2f} h; policy {path} installed "
            f"(per-layer no-op {np.round(1 - self.host.layer_probs().sum(axis=1), 3).tolist()}); training resumes",
            logger="current",
        )
