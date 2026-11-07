#!/usr/bin/env python
"""Faster AutoAugment's policy search on one segmentation dataset, before training.

The search is ``hosts/faster_autoaugment/search.py`` (dda's WGAN-GP search, adapted to
segmentation); this is glue: compose the config, build the search segmentor and the
training data, run the search, write the policy. Nothing is computed here.

    python scripts/faa_search.py --dataset cityscapes --cluster nexus \
        --output-dir experiments/faa_search_cityscapes [host.search.epochs=20 ...]

The segmentor is ``host.search.task`` (configs/host/faster_autoaugment.yaml; default
``mmseg_segformer_b0``) with that dataset's recipe, its backbone initialized as training
initializes it (``init_weights``: the pretrained backbone), and a critic head on its
deepest feature. The data is the dataset's training split, through its training pipeline
(crop, flip, ...) and the model's preprocessor with no host augmentation, as the images a
training run hands the host. ``host.search.train_size`` images of it (the paper's 4000,
capped at the split), a fixed draw by ``host.search.seed``.

The output directory gets ``epoch_<N>.npz`` after every epoch, ``final.npz`` at the end
(the host's ``policy_file``; ``slurm/train.sbatch`` looks for
``<WORK_DIR>/faa_search_<dataset>/final.npz``), ``search_log.jsonl`` and
``search_config.json``. A directory that already holds a ``final.npz`` is refused.

**A search over a subset** (the search-then-train recipe: a trunk trains on the starting
policy to 25%, ``scripts/nms_prune.py`` turns its probe into P, then each arm searches the
ops left). ``--prune-ops "A B ..."`` leaves those ops out of the search; ``--random-prune
K`` (with ``--random-prune-seed S``) leaves out K more drawn at random from the ops the
probe gives a row (``prune.pool=probed``: never SamplePairing or the flips), the random-K
null. Both go through the training run's own prune (``dropaug.config.prune_of``), so the
draw is the one a training run with the same flags would make. The policy then holds only
the ops left (``Policy(names=...)``: fewer parameters, fewer ops a step); ``prune.json``
records the prune and the policy's size with and without it; the default output
directory gets the run name's prune suffix (``prune_tag``):
``<work-root>/faa_search_<dataset>[_prune-...][_<search-tag>]``.

One GPU, one process (no DDP). ``--dry-run`` composes and checks the config and prints
what would run, without building a model. Further ``key=value`` words are Hydra overrides
of the composed config.
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

HOST = "faster_autoaugment"


def die(message: str):
    print(f"[faa_search] ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Search a Faster AutoAugment policy on one segmentation dataset.",
        epilog="Further positional key=value words are Hydra overrides, e.g. host.search.epochs=2.",
    )
    p.add_argument("--dataset", required=True, help="a configs/dataset name")
    p.add_argument("--task", help="the search segmentor's configs/task name (default host.search.task)")
    p.add_argument("--cluster", default="local", help="a configs/cluster name (data and checkpoint paths)")
    p.add_argument("--data-root", help="override the dataset's data_root")
    p.add_argument(
        "--output-dir",
        help="where the policy goes (default <work-root>/faa_search_<dataset>[_prune-...][_<search-tag>])",
    )
    p.add_argument("--work-root", default=os.path.join(REPO, "experiments"), help="the default output dir's parent")
    p.add_argument("--search-tag", help="the default output dir's suffix, for a second search of the same setup")
    p.add_argument(
        "--prune-ops",
        metavar='"OP OP ..."',
        help="ops the search leaves out, space-separated in one word (prune.ops): NMS's P",
    )
    p.add_argument("--random-prune", type=int, metavar="K", help="K more ops left out, drawn from the probed ops (prune.random)")
    p.add_argument("--random-prune-seed", type=int, metavar="S", help="the draw's seed (prune.seed; default the config's seed)")
    p.add_argument("--device", help="default cuda when available")
    p.add_argument("--no-pretrained", action="store_true", help="skip init_weights (a smoke test with no download)")
    p.add_argument("--dry-run", action="store_true", help="check the config and print the plan, build nothing")
    p.add_argument("overrides", nargs="*", metavar="key=value", help="overrides of the config")
    return p.parse_args(argv)


def prune_overrides(args) -> list:
    """The prune flags as the training run's ``prune:`` overrides (train.py's spelling)."""
    out = []
    ops = (args.prune_ops or "").split()
    if ops:
        bad = [op for op in ops if any(c in op for c in "'\\,[] ")]
        if bad:
            die(f"--prune-ops entries {bad} are not op names")
        out.append("prune.ops=[" + ",".join(f"'{op}'" for op in ops) + "]")
    if args.random_prune:
        if args.random_prune < 0:
            die(f"--random-prune must be >= 0, got {args.random_prune}")
        # The random-K null matched to NMS: never an op the probe gives no row.
        out += [f"prune.random={args.random_prune}", "prune.pool=probed"]
        if args.random_prune_seed is not None:
            out.append(f"prune.seed={args.random_prune_seed}")
    elif args.random_prune_seed is not None:
        die("--random-prune-seed is for --random-prune")
    return out


def compose_config(args):
    from hydra.errors import HydraException
    from omegaconf import OmegaConf

    from dropaug.config import ConfigError, check, compose, group_options, search_of

    for group, value in (("dataset", args.dataset), ("cluster", args.cluster), ("task", args.task)):
        if value and value not in group_options(group):
            die(f"--{group} {value!r} is not one of configs/{group}/: {group_options(group)}")
    base = [f"cluster={args.cluster}", f"dataset={args.dataset}", f"host={HOST}", "augment=false"] + prune_overrides(args)
    if args.data_root:
        data_root = os.path.abspath(args.data_root)
        if "'" in data_root or not os.path.isdir(data_root):
            die(f"--data-root {data_root} is not a directory Hydra can take")
        base.append(f"dataset.data_root='{data_root}'")
    try:
        # The search block names the task; compose once to read it, then with it.
        cfg = compose(base + list(args.overrides))
        task = args.task or search_of(cfg)["task"]
        if task not in group_options("task"):
            die(f"host.search.task {task!r} is not one of configs/task/: {group_options('task')}")
        cfg = compose(base + [f"task={task}"] + list(args.overrides))
        if cfg.task.backend != "mmseg":
            die(f"task {task!r} is {cfg.task.backend}; the search is wired for mmseg segmentors only")
        for warning in check(cfg, training=False):
            print(f"[faa_search] WARNING: {warning}", file=sys.stderr)
    except (ConfigError, ValueError, HydraException) as error:
        die(str(error))
    search = search_of(cfg)
    OmegaConf.resolve(cfg)
    return cfg, task, search


def resolve_search_prune(cfg):
    """``(prune, ops searched, tag)``: the training run's prune machinery on the search's
    config (``dropaug.config.prune_of`` / ``prune_tag``)."""
    from dropaug.config import ConfigError, prune_of, prune_tag
    from hosts.faster_autoaugment.host import VOCAB_NAMES

    try:
        prune = prune_of(cfg)
        tag = prune_tag(cfg, prune)
    except (ConfigError, ValueError) as error:
        die(str(error))
    kept = tuple(n for n in VOCAB_NAMES if n not in set(prune.ops))
    if not kept:
        die("the prune leaves the search no op")
    return prune, kept, tag


def build_search(cfg, out_dir: str, device: str, pretrained: bool):
    """``(backend, model, dataset)``: the mmseg backend (its preprocessor makes the clean
    unit batch), the search segmentor wrapped as task model + critic, and the training
    dataset."""
    import torch
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmseg.registry import DATASETS

    from hosts.faster_autoaugment.search import SegmentorCritic
    from tasks.mmseg.backend import MMSegBackend, pooled_features
    from tasks.mmseg.config import build

    mm_cfg = build(cfg, out_dir)
    import_modules_from_strings(**mm_cfg.custom_imports)
    backend = MMSegBackend(mm_cfg, split="train", device=device)
    segmentor = backend.build_model()
    if pretrained:
        segmentor.init_weights()
    init_default_scope(mm_cfg.get("default_scope", "mmseg"))
    dataset = DATASETS.build(mm_cfg.train_dataloader.dataset)
    crop = dataset[0]["inputs"].shape[-2:]
    with torch.no_grad():
        dummy = torch.zeros(1, 3, *crop, device=device)
        feature_dim = int(pooled_features(segmentor, backend._normalize(dummy)).shape[1])
    model = SegmentorCritic(segmentor, feature_dim).to(device)
    return backend, model, dataset


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg, task, search = compose_config(args)
    prune, names, tag = resolve_search_prune(cfg)
    default = f"faa_search_{args.dataset}{tag}" + (f"_{args.search_tag}" if args.search_tag else "")
    out_dir = os.path.abspath(args.output_dir or os.path.join(args.work_root, default))
    if os.path.exists(os.path.join(out_dir, "final.npz")):
        die(f"{out_dir} already holds a finished search (final.npz); pick another --output-dir")
    print(
        f"[faa_search] dataset {cfg.dataset.name} ({cfg.dataset.data_root}), search segmentor {task} "
        f"(recipe {cfg.choices.get('recipe')}), out {out_dir}\n[faa_search] search: {search}"
    )
    if prune:
        print(f"[faa_search] searching {len(names)} of 26 ops; left out: {list(prune.ops)}")
    print(f"OUT_DIR={out_dir}")
    if args.dry_run:
        return 0

    import numpy as np
    import torch
    from mmengine.dataset import pseudo_collate

    from hosts.faster_autoaugment.policy import Policy
    from hosts.faster_autoaugment.search import SearchConfig
    from hosts.faster_autoaugment.search import search as run_search
    from vocab.diff_ops import LABEL_IGNORE_INDEX  # mmseg's ignore index, what the preprocessor pads with

    seed = int(search["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    backend, model, dataset = build_search(cfg, out_dir, device, not args.no_pretrained)

    size = min(int(search["train_size"]), len(dataset))
    subset = torch.utils.data.Subset(dataset, sorted(np.random.default_rng(seed).permutation(len(dataset))[:size].tolist()))
    batch_size = int(search["batch_size"])
    if batch_size < 2 or batch_size % 2:
        die(f"host.search.batch_size must be even (half real, half augmented), got {batch_size}")

    def epoch_batches(epoch):
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=pseudo_collate,
            num_workers=int(search["num_workers"]),
            generator=torch.Generator().manual_seed(seed * 1000 + epoch),
        )
        for batch in loader:
            images01, label = backend.to_unit(batch)
            yield images01.to(device), label.to(device)

    policy = Policy(
        names,
        num_sub_policies=int(cfg.host.num_sub_policies),
        operation_count=int(cfg.host.operation_count),
        temperature=float(search["temperature"]),
        bernoulli_temperature=float(search["bernoulli_temperature"]),
        num_chunks=int(search["num_chunks"]),
    ).to(device)
    config = SearchConfig(
        epochs=int(search["epochs"]),
        task_factor=float(search["task_factor"]),
        gp_factor=float(search["gp_factor"]),
        lr=float(search["lr"]),
        policy_lr=float(search["policy_lr"]),
    )
    if prune:
        from hosts.faster_autoaugment.host import VOCAB_NAMES
        from stats import write_json_atomic

        full = Policy(
            num_sub_policies=policy.num_sub_policies, operation_count=policy.operation_count
        ).param_count()
        write_json_atomic(
            os.path.join(out_dir, "prune.json"),
            prune.record(VOCAB_NAMES, policy_params={"all_ops": full, "searched": policy.param_count()}),
        )
        print(f"[faa_search] policy parameters: {full} over all 26 ops, {policy.param_count()} over the {len(names)} searched")
    meta = {
        "dataset": cfg.dataset.name,
        "task": task,
        "recipe": cfg.choices.get("recipe"),
        "train_size": size,
        "batch_size": batch_size,
        "num_chunks": int(search["num_chunks"]),
        "seed": seed,
    }
    run_search(
        model,
        policy,
        epoch_batches,
        normalize=backend._normalize,
        num_classes=len(dataset.metainfo["classes"]),
        ignore_index=LABEL_IGNORE_INDEX,
        config=config,
        out_dir=out_dir,
        meta=meta,
        log=lambda line: print(f"[faa_search] {line}", flush=True),
    )
    print(f"[faa_search] wrote {os.path.join(out_dir, 'final.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
