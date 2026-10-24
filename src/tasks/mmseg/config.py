"""The mmengine config for one dropaug mmseg run, built from the composed config.

    build(cfg, work_dir, launcher=, resume=) -> mmengine.config.Config

``cfg`` is ``dropaug.config.compose(...)`` / ``load(work_dir)``. The task's .py
(``configs/task/<task.config>``: model, optimizer, schedule, default_runtime; no
dataset) is the base, and the builder fills in everything the yaml groups own:

    dataset/   the train / val / test dataloaders (type, data_root, data_prefix,
               pipelines), num_classes on every decode / auxiliary head, the crop size,
               and the evaluator when the dataset names one (``mmseg.val_evaluator``)
    host/      ``model.data_preprocessor`` becomes ``GpuAugSegDataPreProcessor`` with
               ``host=dict(type=<name>, **kwargs)`` and ``augment``
    prune:     the static prune (``dropaug.config.prune_of``): the host's ``pruned`` /
               ``hold_none_prob``, RobustValLoop's ``pruned_augmentations`` /
               ``hold_none_prob``, and CollectGradientHook's ``probe_pruned``
    probe/ selector/
               ``custom_hooks``: ``CollectGradientHook`` (NORMAL) and
               ``ApplyWeightsHook`` (LOW), the priorities their classes carry, or
               ``PruneHook`` (LOW) for selector nms. The host's ``fade:`` block goes
               to CollectGradientHook's ``fade`` / ``fade_at``
    sa/        host=sensaug only: ``RobustIterBasedTrainLoop`` / ``RobustValLoop`` /
               ``SubsetTestLoop``. Any other host keeps mmengine's stock loops
    task/      max_iters, val and checkpoint intervals, AMP, ``cfg_options`` (dotted
               mmengine keys, as mmseg's ``--cfg-options``, merged last)
    cluster/   ``pretrained_cache``: backbone ``Pretrained`` URLs become local paths

Glue only: nothing here computes. Importing the module needs neither mmengine nor mmseg
(mmengine is imported in ``build``); the registry lookups of the types it names happen in
``Runner.from_cfg``, with ``tasks.mmseg.register`` imported through ``custom_imports``.

Ported from Sensaug's ``train.py``: ``_localize_pretrained_checkpoint`` (636-670) and
``build_config`` (672-1120): AMP (704-720), auto_scale_lr (722-724), work_dir and resume
(735-770), the train pipeline and data root (772-870), the visualizer (884-889), the
round interval (891-905), logger and checkpoint hooks (925-933), the test dataloader
(752, 935), the SA loop's loops (950-1010), the GPU preprocessor (1100-1110) and the
randomness (1112).

What the sensaug host's loops require of this config (``RobustValLoop`` raises at its
first round otherwise), and what the builder therefore derives rather than reads:

- ``train_cfg.val_interval`` is the host's ``round_interval`` and ``val_cfg.warmup_rounds``
  its ``warmup_rounds``
- ``train_cfg.init_sa`` and the host's ``pre_train_round`` are one value,
  ``bool(resume or sa.init_sa)``, as Sensaug derived both from ``cfg.resume or
  args.no_warmup``. ``dropaug.config.check`` refuses ``host.pre_train_round !=
  sa.init_sa``; a resume turns both on here, and the saved dropaug.yaml keeps its values
- the test loop is ``SubsetTestLoop`` (the curve search scores through it), over the val
  dataloader at batch size 1, with ``test_evaluator = val_evaluator``

Left out of Sensaug's ``build_config``, and why:

- the config-file search, the ``schedule_320k`` fallback merge and the foundation
  backbone (672-730): a dropaug task is one self-contained .py, written once per
  backbone (``configs/task/``)
- ``--aug-type`` branches (legacy CPU transforms, AutoAugment / AugMix / VIP / IDBH,
  ``random``, ``no_corr_sa``, geometric/photometric-only, perturbation_set): the host and
  the vocab replace them. ``pruned_ops`` and ``pruned_augmentations`` are the ``prune:``
  block, which reaches the host (so its warmup policy too) and the val loop
- ``GradCorrValLoop`` and the correlation hooks' gate (``fire_iters``, ``magnitudes_path``,
  ``skip_pruned``): the sensaug host owns the round grid and the loops publish R
- ``--adamw``, ``--freeze-early-layers``, ``--max-iters`` and the ACDC branch (ACDC is
  eval-only here): a task's .py or ``task.cfg_options`` says them
- the auto-resume on an existing ``last_checkpoint``: ``resume`` is explicit (slurm/
  train.sbatch refuses an existing run without ``ALLOW_RESUME``)

Not ported because it never took effect: Sensaug's LoveDA learning rate (999-1001, SA
arms only), ``cfg.optimizer.lr *= 0.1``. ``cfg.optimizer`` is a separate ConfigDict from
``optim_wrapper.optimizer``, the one the Runner builds, so Sensaug's LoveDA runs trained
at the config's full lr (checked with mmengine on its pspnet_r18 and segformer_mit-b0
LoveDA configs). These do too.

Deviations, for the record: ``randomness.seed`` is ``cfg.seed`` (Sensaug: 0); the
checkpoint interval is the task's; the visualizer keeps the local backend beside
TensorBoard.
"""

import copy
import os
from typing import Any, Optional

from dropaug.config import CONFIG_DIR, fade_of, host_kwargs, prune_of, selector_spec, to_python

__all__ = ["build", "localize_pretrained_checkpoint"]

CUSTOM_IMPORTS = ["tasks.mmseg.register"]

# Pipeline keys mmcv wants as tuples (``imrescale`` raises TypeError on a list scale);
# yaml only has lists.
_TUPLE_KEYS = ("scale", "crop_size", "img_scale", "size")

_PROBE_KEYS = ("magnitude", "magnitude_mode", "n_probe", "probe_seed", "batch_size", "fd_step", "num_workers", "split")
_ANALYSIS_KEYS = ("n_min", "normalize_per_image", "bootstrap", "bootstrap_reps", "bootstrap_seed", "fdr_alpha")
_SA_VAL_KEYS = (
    "ratio",
    "sa_curve_path",
    "uniform",
    "descending_MA",
    "remove_H",
    "weighted_augs",
    "num_levels",
    "tolerance",
)


def _tuples(node: Any) -> Any:
    """Lists of numbers under ``_TUPLE_KEYS`` as tuples, throughout a pipeline."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if (
                key in _TUPLE_KEYS
                and isinstance(value, list)
                and value
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
            ):
                out[key] = tuple(value)
            else:
                out[key] = _tuples(value)
        return out
    if isinstance(node, list):
        return [_tuples(v) for v in node]
    return node


def localize_pretrained_checkpoint(cfg, backbone: str, cache_dir: Optional[str]) -> None:
    """Redirect ``init_cfg`` ``Pretrained`` checkpoint URLs under ``cfg.model`` to
    ``<cache_dir>/<basename>``, for compute nodes without internet (Sensaug's
    ``_localize_pretrained_checkpoint``, train.py 636-670). Fails before the Runner,
    NCCL and the dataloaders spin up if a file is not staged. ``cache_dir`` None leaves
    the URLs alone."""
    if not cache_dir or "model" not in cfg:
        return

    def walk(node):
        if isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "Pretrained" and str(node.get("checkpoint", "")).startswith("http"):
            url = node["checkpoint"]
            local_path = os.path.join(cache_dir, os.path.basename(url))
            if not os.path.isfile(local_path):
                raise FileNotFoundError(
                    f"Pretrained checkpoint not cached locally: {url}\n"
                    f"Expected at: {local_path}\n"
                    "Compute nodes have no internet access -- fetch it first from a login node "
                    f"(or your machine + rsync) into {cache_dir} (backbone: {backbone})."
                )
            node["checkpoint"] = local_path
            return
        for v in node.values():
            walk(v)

    walk(cfg.model)


def _set_num_classes(head, num_classes: int) -> None:
    for h in head if isinstance(head, (list, tuple)) else [head]:
        if isinstance(h, dict):
            h["num_classes"] = num_classes


def _dataloader(base, dataset_cfg: dict, split: str, batch_size: int, num_workers: int, data_root: str) -> dict:
    mm = dataset_cfg["mmseg"]
    part = mm[split]
    # dataset-wide keys (type, and e.g. a shared data_prefix), then the split's own
    # (data_prefix, ann_file, ...); the pipeline gets mmcv's tuples
    dataset = {**mm["dataset"], **{k: v for k, v in part.items() if k != "pipeline"}}
    dataset["data_root"] = data_root
    dataset["pipeline"] = _tuples(part["pipeline"])
    if split == "val":
        dataset["test_mode"] = True
    dl = dict(base or {})
    dl["batch_size"] = batch_size
    dl["num_workers"] = num_workers
    # mmengine refuses persistent_workers with no workers
    dl["persistent_workers"] = bool(num_workers > 0 and dl.get("persistent_workers", True))
    dl["dataset"] = dataset
    return dl


def _stride_fits(stride, crop):
    return all(s <= c for s, c in zip(stride, crop))


def build(cfg, work_dir: str, *, launcher: str = "none", resume: bool = False):
    """The mmengine ``Config`` for the run ``cfg`` describes, in ``work_dir``.

    ``resume`` continues the run in ``work_dir`` (``cfg.resume = True``, ``load_from =
    None``: mmengine finds ``last_checkpoint``), and puts the SA loop's pre-train round
    on, as Sensaug did, since the host's q is not saved in checkpoints. Raises
    ``ValueError`` on a config the loops would refuse hours in."""
    from mmengine.config import Config

    task = to_python(cfg.task)
    if task.get("backend", "mmseg") != "mmseg":
        raise ValueError(f"tasks.mmseg.config builds mmseg tasks; task.backend is {task.get('backend')!r}")
    dataset = to_python(cfg.dataset)
    if "mmseg" not in dataset:
        raise ValueError(f"dataset {dataset.get('name')!r} has no mmseg section")

    base = Config.fromfile(os.path.join(CONFIG_DIR, "task", task["config"]))
    max_iters = int(task["max_iters"])
    mm = dataset["mmseg"]
    crop = tuple(mm["crop_size"])
    data_root = dataset["data_root"]
    num_workers = int(task.get("num_workers", 4))

    pruning = prune_of(cfg)
    spec = host_kwargs(cfg, pruning)
    host_name = spec.pop("name")
    is_sa = host_name == "sensaug"

    out = base
    out.launcher = launcher
    out.work_dir = str(work_dir)
    # Sensaug 727-730: the config's own pretrained entries are replaced by init_cfg
    out.pretrained = None
    out.model.pretrained = None
    localize_pretrained_checkpoint(out, task.get("backbone", ""), to_python(cfg.cluster).get("pretrained_cache"))

    # --- data (Sensaug 772-870) ---
    out.train_dataloader = _dataloader(
        out.get("train_dataloader"), dataset, "train", int(task["batch_size"]), num_workers, data_root
    )
    # val and test: batch size 1 (labels of mixed native sizes must stack on the eval path)
    out.val_dataloader = _dataloader(out.get("val_dataloader"), dataset, "val", 1, num_workers, data_root)
    out.test_dataloader = copy.deepcopy(out.val_dataloader)
    if mm.get("val_evaluator"):  # a dataset whose mmseg base reports more (synapse: mDice)
        out.val_evaluator = copy.deepcopy(mm["val_evaluator"])
    out.test_evaluator = copy.deepcopy(out.get("val_evaluator"))

    num_classes = int(dataset["num_classes"])
    _set_num_classes(out.model.decode_head, num_classes)
    if "auxiliary_head" in out.model:
        _set_num_classes(out.model.auxiliary_head, num_classes)

    # crop: the preprocessor pads training batches to it, as every Sensaug config's
    # `data_preprocessor = dict(size=crop_size)` did. That is also the probe's batch
    # (to_unit runs training=True), so a val image smaller than the crop (pascal_voc12,
    # ade20k) is padded before any op, geometric ones included, is applied. The task
    # .py files leave `size` out on purpose; it comes from here.
    # Slide inference keeps the task's own window, as Sensaug's per-file configs did.
    dp = out.model.data_preprocessor
    dp.size = crop
    mtest = out.model.get("test_cfg")
    if mtest is not None and mtest.get("mode") == "slide":
        window = tuple(mtest.get("crop_size", ()))
        stride = tuple(mtest.get("stride", window))
        if not _stride_fits(stride, window):  # a stride past the window leaves gaps
            raise ValueError(
                f"task {task['config']}: slide test_cfg stride {stride} exceeds its window "
                f"{window}, so inference would skip pixels"
            )

    # --- run settings (Sensaug 735-770, 925-933) ---
    out.randomness = dict(seed=int(cfg.seed), diff_rank_seed=False)
    out.train_cfg = dict(out.get("train_cfg") or {})
    out.train_cfg["max_iters"] = max_iters

    hooks = out.default_hooks
    hooks.logger.interval = min(200, max_iters)
    hooks.checkpoint.interval = max(1, int(task.get("checkpoint_interval") or max_iters // 20))
    hooks.checkpoint.save_best = "mIoU"
    hooks.checkpoint.max_keep_ckpts = 3

    out.resume = bool(resume)
    out.load_from = None if resume else out.get("load_from")

    if task.get("amp"):
        wrapper = out.optim_wrapper.type
        if wrapper != "AmpOptimWrapper":
            if wrapper != "OptimWrapper":
                raise ValueError(f"task.amp needs optim_wrapper.type OptimWrapper, got {wrapper!r}")
            out.optim_wrapper.type = "AmpOptimWrapper"
            out.optim_wrapper.loss_scale = "dynamic"
    if "auto_scale_lr" in out and "base_batch_size" in out.auto_scale_lr:
        out.auto_scale_lr.enable = True

    out.visualizer = dict(
        type="Visualizer", vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
    )

    # --- host, augmentation, loops ---
    sa = to_python(cfg.get("sa")) or {}
    init_sa = bool(resume or sa.get("init_sa", False))
    if is_sa:
        # dropaug.config.check refuses host.pre_train_round != sa.init_sa; a resume turns
        # both on here without touching the saved config
        spec["pre_train_round"] = init_sa
        if int(spec.get("max_iters", max_iters)) != max_iters:
            raise ValueError(f"host.max_iters {spec['max_iters']} != task.max_iters {max_iters}")
        from hosts import build_host

        host = build_host(host_name, **spec)
        out.train_cfg.update(
            type="RobustIterBasedTrainLoop", val_interval=int(host.round_interval), init_sa=init_sa
        )
        out.val_cfg = dict(
            type="RobustValLoop",
            **{k: sa[k] for k in _SA_VAL_KEYS if k in sa},
            pruned_augmentations=list(pruning.ops),
            hold_none_prob=bool(pruning.hold_none_prob),
            warmup_rounds=int(host.warmup_rounds),
        )
        out.test_cfg = dict(
            type="SubsetTestLoop",
            ratio=sa.get("test_ratio", sa.get("ratio", 0.25)),
            kid_rgb=bool(sa.get("kid_rgb", True)),
        )
    else:
        out.train_cfg.update(type="IterBasedTrainLoop", val_interval=int(task["val_interval"]))
        out.val_cfg = dict(type="ValLoop")
        out.test_cfg = dict(type="TestLoop")

    dp.type = "GpuAugSegDataPreProcessor"
    dp.host = dict(type=host_name, **spec)
    dp.augment = bool(cfg.augment)

    probe = to_python(cfg.probe)
    collect = {k: probe[k] for k in _PROBE_KEYS if k in probe and probe[k] is not None}
    collect["fade"], collect["fade_at"] = fade_of(cfg)
    collect["probe_pruned"] = bool(pruning.probe)
    analysis = probe.get("analysis") or {}
    apply_kwargs = {k: analysis[k] for k in ("mask_within_op", *_ANALYSIS_KEYS) if k in analysis}
    selector = selector_spec(cfg)
    if selector["selector"]["type"] == "nms":
        # The NMS arm has no statistical gate, so none of the analysis keys but the
        # within-op mask reach it, and its prune is latched (hooks.PruneHook).
        weights = dict(
            type="PruneHook",
            tau=selector["selector"].get("tau", 0.5),
            mask_within_op=analysis.get("mask_within_op", True),
        )
    else:
        weights = dict(
            type="ApplyWeightsHook",
            selector=selector["selector"],
            lambda_ramp=selector["lambda_ramp"],
            **apply_kwargs,
        )
    out.custom_hooks = list(out.get("custom_hooks") or []) + [
        dict(type="CollectGradientHook", **collect),
        weights,
    ]
    # merge into the task's own custom_imports (convnext, vitsam: mmpretrain.models)
    existing = out.get("custom_imports") or {}
    imports = existing.get("imports", [])
    imports = [imports] if isinstance(imports, str) else list(imports)
    imports += [m for m in CUSTOM_IMPORTS if m not in imports]
    out.custom_imports = dict(imports=imports, allow_failed_imports=False)

    # --- the task's own overrides, last ---
    if task.get("cfg_options"):
        out.merge_from_dict(task["cfg_options"])
        if int(out.train_cfg.max_iters) != max_iters:
            raise ValueError(
                "task.cfg_options changed train_cfg.max_iters; set task.max_iters instead, which "
                "the host's round grid and the probe schedule derive from"
            )
    return out
