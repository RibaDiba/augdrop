"""mmengine glue for one Greedy AutoAugment search trial: a child network trained on one
chain, scored on held-out training images. Nothing is computed here.

The paper's child (``Search/main.py``, fixed: the released loop trains one network across
every candidate and stops each "epoch" after one batch): a fresh network of the trained
network's architecture ("the child networks and training networks share the same
infrastructures"), trained ``epochs`` epochs on the training images with every image
augmented by the chain (probability 1), then scored on the held-out ones, clean. Here:

- **the network** is the run's task and recipe (``tasks.mmseg.config.build``), initialized
  as training initializes it (the pretrained backbone), the same seed on every trial (the
  code reseeds before every candidate, ``divide_train``'s ``random.seed(10000)``);
- **the data** are ``train_ids`` of the train split through its training pipeline, and the
  score set ``score_ids`` of the same split through the val pipeline (clean, deterministic),
  disjoint (``scripts/gaa_search.py`` draws both); the paper holds 2500 of CIFAR's 50000 out;
- **the augmentation** is the host in ``chain`` mode in the preprocessor: every image, the
  whole chain, on GPU, drawn per batch (the code applies it once, to a stored copy);
- **resolution** ``r`` scales every resize (an ``r`` Resize is added where the pipeline has
  none), every crop, the preprocessor's pad size and a slide window and stride;
- **the schedule**: ``iters = ceil(epochs * |train_ids| / batch)`` steps at ``batch`` images
  on one GPU (default the trunk's batch over its 4 GPUs); the recipe's optimizer, its
  ``param_scheduler`` compressed onto the child's iterations (each begin and end times
  iters / the schedule's last end, so a warmup keeps its share);
- **the score** is the val evaluator's mIoU (the paper's accuracy) on the score set, as the
  test loop's (``runner.test()``) once training ends; there is no val loop, so training
  never stops to validate. No checkpoints, no probe hooks.

Checked on stand-ins only (no mmengine or mmseg in ``sensAug``).
"""

import copy
import math
import os
from typing import List, Optional, Sequence

from omegaconf import OmegaConf

#: The hooks a child does without: the probe, the selector and the kept checkpoints.
DROPPED_HOOKS = ("CollectGradientHook", "ApplyWeightsHook", "PruneHook", "KeepCheckpointHook")

#: Pipeline steps whose ``scale`` / ``crop_size`` the resolution scales.
_RESIZES = ("Resize", "RandomResize", "RandomChoiceResize")
_CROPS = ("RandomCrop",)


def _scaled(value, r: float):
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return type(value)(_scaled(v, r) for v in value)
        return type(value)(max(1, int(round(float(v) * r))) for v in value)
    return max(1, int(round(float(value) * r)))


def scale_pipeline(pipeline: Sequence[dict], r: float, *, insert_after: str) -> List[dict]:
    """``pipeline`` at resolution ``r``: each resize's ``scale`` and each crop's
    ``crop_size`` times r; with no resize at all, ``Resize(scale_factor=r)`` right after
    the ``insert_after`` step (both the image and, if loaded, the label)."""
    steps = [dict(step) for step in pipeline]
    if r == 1.0:
        return steps
    resized = False
    for step in steps:
        if step.get("type") in _RESIZES:
            for key in ("scale", "scales"):
                if step.get(key) is not None:
                    step[key] = _scaled(step[key], r)
                    resized = True
            if step.get("scale_factor") is not None:
                step["scale_factor"] = float(step["scale_factor"]) * r
                resized = True
        if step.get("type") in _CROPS and step.get("crop_size") is not None:
            step["crop_size"] = _scaled(step["crop_size"], r)
    if not resized:
        at = next((i for i, s in enumerate(steps) if s.get("type") == insert_after), None)
        if at is None:
            raise ValueError(f"no {insert_after} step to put the resolution's Resize after: {steps}")
        steps.insert(at + 1, dict(type="Resize", scale_factor=float(r), keep_ratio=False))
    return steps


def compress_schedule(schedulers, iters: int) -> list:
    """Each scheduler's ``begin`` / ``end`` times ``iters`` / the schedule's last end."""
    schedulers = [dict(s) for s in (schedulers if isinstance(schedulers, (list, tuple)) else [schedulers])]
    ends = [int(s["end"]) for s in schedulers if s.get("end") is not None]
    if not ends:
        return schedulers
    factor = iters / float(max(ends))
    for s in schedulers:
        begin = int(round(int(s.get("begin", 0)) * factor))
        s["begin"] = begin
        if s.get("end") is not None:
            s["end"] = max(begin + 1, int(round(int(s["end"]) * factor)))
    return schedulers


def child_iters(epochs: float, train_size: int, batch_size: int) -> int:
    return max(1, int(math.ceil(float(epochs) * int(train_size) / int(batch_size))))


def child_batch_size(cfg, setting: Optional[int]) -> int:
    """``host.search.batch_size``, or the trunk's batch: ``task.batch_size`` per GPU over
    train.sbatch's 4 GPUs."""
    return int(setting) if setting else 4 * int(cfg.task.batch_size)


def child_dictconfig(cfg, chain: Sequence[Sequence]):
    """The run's composed config with the host as one search candidate: the chain on
    every image (``chain`` mode), augmentation on, no fork, no kept checkpoints."""
    out = copy.deepcopy(cfg)
    OmegaConf.set_struct(out, False)
    out.host.policy_file = None
    out.host.chain = [[str(op), int(level)] for op, level in chain]
    out.augment = True
    out.fork_from = None
    out.keep_checkpoints = []
    return out


def edit_child(
    out,
    *,
    train_ids: Sequence[int],
    score_ids: Sequence[int],
    iters: int,
    batch_size: int,
    resolution: float,
    seed: int,
    num_workers: int,
):
    """The child's edits on the built mmengine config ``out`` (module docstring), by item
    access, so a plain dict of the same shape takes them too."""
    r = float(resolution)
    train = out["train_dataloader"]
    train["batch_size"] = int(batch_size)
    train["num_workers"] = int(num_workers)
    train["persistent_workers"] = bool(num_workers > 0)
    tds = train["dataset"]
    tds["indices"] = [int(i) for i in train_ids]
    tds["pipeline"] = scale_pipeline(tds["pipeline"], r, insert_after="LoadAnnotations")

    # The score set: the train split's images through the val pipeline, as the test
    # loop's (``runner.test()`` after training); no val loop, so training never pauses.
    score = copy.deepcopy(out["val_dataloader"])
    sds = score["dataset"]
    for key in ("data_prefix", "ann_file", "img_suffix", "seg_map_suffix"):
        if key in tds:
            sds[key] = copy.deepcopy(tds[key])
    sds["indices"] = [int(i) for i in score_ids]
    sds["pipeline"] = scale_pipeline(sds["pipeline"], r, insert_after="LoadImageFromFile")
    score["num_workers"] = int(num_workers)
    score["persistent_workers"] = bool(num_workers > 0)
    out["test_dataloader"] = score
    out["test_evaluator"] = copy.deepcopy(out["val_evaluator"])
    out["val_dataloader"] = None
    out["val_cfg"] = None
    out["val_evaluator"] = None

    model = out["model"]
    dp = model["data_preprocessor"]
    if dp.get("size") is not None:
        dp["size"] = _scaled(dp["size"], r)
    mtest = model.get("test_cfg")
    if mtest is not None and mtest.get("mode") == "slide":
        mtest["crop_size"] = _scaled(mtest["crop_size"], r)
        mtest["stride"] = _scaled(mtest.get("stride", mtest["crop_size"]), r)

    out["train_cfg"] = dict(type="IterBasedTrainLoop", max_iters=int(iters))
    out["test_cfg"] = dict(type="TestLoop")
    out["param_scheduler"] = compress_schedule(out["param_scheduler"], int(iters))

    hooks = out["default_hooks"]
    hooks["checkpoint"] = dict(type="CheckpointHook", by_epoch=False, interval=-1, save_last=False)
    hooks["logger"] = dict(type="LoggerHook", interval=min(50, int(iters)), log_metric_by_epoch=False)
    hooks.pop("visualization", None)
    out["custom_hooks"] = [h for h in (out.get("custom_hooks") or []) if h.get("type") not in DROPPED_HOOKS]
    out["visualizer"] = dict(type="Visualizer", vis_backends=[dict(type="LocalVisBackend")])
    out["randomness"] = dict(seed=int(seed), diff_rank_seed=False)
    out["launcher"] = "none"
    out["resume"] = False
    out["load_from"] = None
    return out


def child_config(
    cfg,
    chain: Sequence[Sequence],
    work_dir: str,
    *,
    train_ids: Sequence[int],
    score_ids: Sequence[int],
    epochs: float,
    batch_size: Optional[int],
    resolution: float,
    seed: int,
    num_workers: int,
):
    """The mmengine config of one trial (module docstring) in ``work_dir``."""
    from .config import build

    batch = child_batch_size(cfg, batch_size)
    iters = child_iters(epochs, len(train_ids), batch)
    out = build(child_dictconfig(cfg, chain), work_dir)
    return edit_child(
        out,
        train_ids=train_ids,
        score_ids=score_ids,
        iters=iters,
        batch_size=batch,
        resolution=resolution,
        seed=seed,
        num_workers=num_workers,
    )


def run_trial(mm_cfg) -> dict:
    """Train the child, then score it: the evaluator's metrics on the score set (``mIoU``
    among them)."""
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    import_modules_from_strings(**mm_cfg.custom_imports)
    init_default_scope(mm_cfg.get("default_scope", "mmseg"))
    os.makedirs(mm_cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(mm_cfg)
    runner.train()
    metrics = runner.test()
    return {str(k): float(v) for k, v in dict(metrics).items()}


def train_split_size(cfg, work_dir: str) -> int:
    """How many images the train split has (the draw of the train and score ids)."""
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmseg.registry import DATASETS

    from .config import build

    mm_cfg = build(cfg, work_dir)
    import_modules_from_strings(**mm_cfg.custom_imports)
    init_default_scope(mm_cfg.get("default_scope", "mmseg"))
    dataset = dict(mm_cfg.train_dataloader.dataset)
    dataset["lazy_init"] = True
    built = DATASETS.build(dataset)
    built.full_init()
    return len(built)
