#!/usr/bin/env python
"""AdaAug's policy search on one segmentation dataset, before training.

The search is ``src/adaaug/`` (AdaAug's ``search.py``, adapted to segmentation); this is
glue: compose the config, build the search segmentor gf, its training data and the
held-out images, run the search, write the policy. Nothing is computed here.

    python scripts/adaaug_search.py --dataset cityscapes --cluster nexus \
        --output-dir experiments/adaaug_search_cityscapes [host.search.epochs=200 ...]

gf is ``host.search.task`` (configs/host/adaaug.yaml; default ``mmseg_segformer_b0``) with
that dataset's recipe and its backbone initialized as training initializes it
(``init_weights``: the pretrained backbone), trained from there with the recipe's
optimizer (its ``optim_wrapper``) under AdaAug's cosine schedule. Its training images are
the dataset's training split through its training pipeline (crop, flip, ...) and the
model's preprocessor with no host augmentation: ``host.search.train_size`` of them (capped
at the split), a fixed draw by ``host.search.seed``. The held-out images h explores on are
the ``host.search.explore_split`` split (default val; AdaAug holds out the rest of its
reduced training set, which a 2975-image Cityscapes doesn't leave), read through the
*training* pipeline as AdaAug's search set is (its ``transform_train_pre``: crop and flip),
a fresh random batch each explore.

The output directory gets ``final/`` at the end (``gf.pth``, ``h.pth``, ``meta.json`` and
``gf_config.py``, gf's config for ``tasks/mmseg/adaaug_hooks.py`` to rebuild it with a plain
preprocessor): the host's ``policy_file`` (train.sbatch ``ADAAUG_POLICY=<dir>``).
``state.pt`` after every epoch (a rerun with the same flags resumes), ``epoch_<N>/`` every
``host.search.save_every`` epochs, ``search_log.jsonl`` and ``search_config.json``. A
directory that already holds ``final/meta.json`` is refused.

**A search over a subset** (the search-then-train recipe: a trunk trains on the starting
policy to 25%, ``scripts/nms_prune.py`` turns its probe into P, then each arm searches the
ops left). ``--prune-ops "A B ..."`` leaves those ops out of the search; ``--random-prune
K`` (with ``--random-prune-seed S``) leaves out K more drawn at random from the ops the
probe gives a row (``prune.pool=probed``: never Flip), the random-K null. Both go through
the training run's own prune (``dropaug.config.prune_of``), so the draw is the one a
training run with the same flags would make. h then has only the outputs left
(``hosts.adaaug.searched_outputs``: fewer parameters, fewer explore copies); ``prune.json``
records the prune and h's size and copies with and without it; the default output
directory gets the run name's prune suffix (``prune_tag``):
``<work-root>/adaaug_search_<dataset>[_prune-...][_<search-tag>]``.

One GPU, one process (no DDP). ``--dry-run`` composes and checks the config and prints
what would run and its pass counts, without building a model. Further ``key=value`` words
are Hydra overrides of the composed config.
"""

import argparse
import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

HOST = "adaaug"


def die(message: str):
    print(f"[adaaug_search] ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Search an AdaAug policy on one segmentation dataset.",
        epilog="Further positional key=value words are Hydra overrides, e.g. host.search.epochs=2.",
    )
    p.add_argument("--dataset", required=True, help="a configs/dataset name")
    p.add_argument("--task", help="the search segmentor's configs/task name (default host.search.task)")
    p.add_argument("--cluster", default="local", help="a configs/cluster name (data and checkpoint paths)")
    p.add_argument("--data-root", help="override the dataset's data_root")
    p.add_argument(
        "--output-dir",
        help="where the policy goes (default <work-root>/adaaug_search_<dataset>[_prune-...][_<search-tag>])",
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
        if cfg.host.get("policy_file") != "initial":
            die(f"the search starts from the starting policy; host.policy_file is {cfg.host.get('policy_file')!r}")
        for warning in check(cfg, training=False):
            print(f"[adaaug_search] WARNING: {warning}", file=sys.stderr)
    except (ConfigError, ValueError, HydraException) as error:
        die(str(error))
    search = search_of(cfg)
    OmegaConf.resolve(cfg)
    return cfg, task, search


def resolve_search_prune(cfg):
    """``(prune, outputs searched, tag)``: the training run's prune machinery on the
    search's config (``dropaug.config.prune_of`` / ``prune_tag``)."""
    from dropaug.config import ConfigError, prune_of, prune_tag
    from hosts.adaaug import searched_outputs

    try:
        prune = prune_of(cfg)
        tag = prune_tag(cfg, prune)
        outputs = searched_outputs(prune.ops)
    except (ConfigError, ValueError, KeyError) as error:
        die(str(error))
    if len(outputs) <= 1:
        die("the prune leaves the search no op")
    return prune, outputs, tag


def _unwrap_dataset(cfg):
    """A dataset config under mmengine's wrappers (RepeatDataset and the like)."""
    while "dataset" in cfg and "type" in cfg and str(cfg["type"]).endswith("Dataset") and "data_prefix" not in cfg:
        cfg = cfg["dataset"]
    return cfg


def explore_dataset_cfg(mm_cfg, split: str):
    """The training dataset's config with ``split``'s files: its images through the
    training pipeline (AdaAug's search set takes ``transform_train_pre``)."""
    train = copy.deepcopy(_unwrap_dataset(mm_cfg.train_dataloader.dataset))
    other = _unwrap_dataset(mm_cfg.get(f"{split}_dataloader").dataset)
    if "data_prefix" not in train or "data_prefix" not in other:
        die(f"can't read the {split} split through the training pipeline: no data_prefix in the dataset configs")
    train["data_prefix"] = copy.deepcopy(other["data_prefix"])
    for key in ("ann_file", "data_root"):
        if key in other:
            train[key] = other[key]
    return train


def gf_config(mm_cfg):
    """gf's config for the training hook to rebuild it: the search segmentor with a plain
    ``SegDataPreProcessor`` (the GPU-augmenting one would build a host), scope and imports."""
    from mmengine.config import Config

    model = copy.deepcopy(mm_cfg.model)
    dp = dict(model.get("data_preprocessor") or {})
    plain = {k: dp[k] for k in ("mean", "std", "bgr_to_rgb", "pad_val", "seg_pad_val", "size", "size_divisor") if k in dp}
    model["data_preprocessor"] = dict(type="SegDataPreProcessor", **plain)
    fields = dict(model=model, default_scope=mm_cfg.get("default_scope", "mmseg"))
    imports = [m for m in (mm_cfg.get("custom_imports") or {}).get("imports", []) if m != "tasks.mmseg.register"]
    if imports:
        fields["custom_imports"] = dict(imports=imports, allow_failed_imports=False)
    return Config(fields)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg, task, search = compose_config(args)
    prune, outputs, tag = resolve_search_prune(cfg)
    default = f"adaaug_search_{args.dataset}{tag}" + (f"_{args.search_tag}" if args.search_tag else "")
    out_dir = os.path.abspath(args.output_dir or os.path.join(args.work_root, default))
    if os.path.exists(os.path.join(out_dir, "final", "meta.json")):
        die(f"{out_dir} already holds a finished search (final/meta.json); pick another --output-dir")

    from adaaug.search import pass_counts

    counts = pass_counts(
        train_images=int(search["train_size"]),
        batch_size=int(search["batch_size"]),
        epochs=int(search["epochs"]),
        search_freq=int(search["search_freq"]),
        explore_batch=int(search["explore_batch"]),
        n_outputs=len(outputs),
    )
    print(
        f"[adaaug_search] dataset {cfg.dataset.name} ({cfg.dataset.data_root}), search segmentor {task} "
        f"(recipe {cfg.choices.get('recipe')}), out {out_dir}\n[adaaug_search] search: {search}"
    )
    if prune:
        print(f"[adaaug_search] searching {len(outputs)} of 17 outputs; left out: {list(prune.ops)}")
    print(f"[adaaug_search] at most (train_size before the split's cap): {counts}")
    print(f"OUT_DIR={out_dir}")
    if args.dry_run:
        return 0

    import numpy as np
    import torch
    from mmengine.dataset import pseudo_collate
    from mmengine.optim import build_optim_wrapper
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings
    from mmseg.registry import DATASETS

    import adaaug
    import madaug
    from hosts import build_host
    from tasks.mmseg.backend import MMSegBackend, pooled_features
    from tasks.mmseg.config import build
    from vocab import build_vocab
    from vocab.diff_ops import LABEL_IGNORE_INDEX  # mmseg's ignore index, what the preprocessor pads with

    seed = int(search["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    mm_cfg = build(cfg, out_dir)
    import_modules_from_strings(**mm_cfg.custom_imports)
    backend = MMSegBackend(mm_cfg, split="train", device=device)
    segmentor = backend.build_model()
    if not args.no_pretrained:
        segmentor.init_weights()
    init_default_scope(mm_cfg.get("default_scope", "mmseg"))
    dataset = DATASETS.build(mm_cfg.train_dataloader.dataset)
    explore_set = DATASETS.build(explore_dataset_cfg(mm_cfg, str(search["explore_split"])))
    crop = dataset[0]["inputs"].shape[-2:]
    with torch.no_grad():
        dummy = torch.zeros(1, 3, *crop, device=device)
        feature_dim = int(pooled_features(segmentor, backend._normalize(dummy)).shape[1])

    host = build_host(HOST, policy_file="initial", pruned=list(prune.ops))
    if tuple(host.outputs) != tuple(outputs):
        die(f"the host's outputs {host.outputs} are not the search's {outputs}")
    ops = build_vocab({"ops": list(host.vocabulary())})
    policy_cfg = cfg.host.policy
    torch.manual_seed(seed)
    policy = madaug.Projection(
        feature_dim, n_layers=int(policy_cfg.n_layers), n_hidden=int(policy_cfg.n_hidden), n_outputs=len(outputs)
    ).to(device)
    optimizer = build_optim_wrapper(segmentor, copy.deepcopy(mm_cfg.optim_wrapper)).optimizer

    size = min(int(search["train_size"]), len(dataset))
    subset = torch.utils.data.Subset(dataset, sorted(np.random.default_rng(seed).permutation(len(dataset))[:size].tolist()))
    batch_size = int(search["batch_size"])
    steps_per_epoch = size // batch_size
    if steps_per_epoch < 1:
        die(f"host.search.batch_size {batch_size} is more than the {size} training images")

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

    explore_size = int(search["explore_batch"])
    if explore_size > len(explore_set):
        die(f"host.search.explore_batch {explore_size} is more than the {len(explore_set)} held-out images")

    def explore_batch(epoch, k):
        # A fresh random batch each explore (AdaAug's next(iter(search_queue)) on a
        # shuffled loader), seeded by (seed, epoch, k) so a resumed search draws the same.
        picks = np.random.default_rng([seed, epoch, k]).choice(len(explore_set), size=explore_size, replace=False)
        images01, label = backend.to_unit(pseudo_collate([explore_set[int(i)] for i in picks]))
        return images01.to(device), label.to(device)

    from stats import write_json_atomic

    if prune:
        full = adaaug.policy_params(feature_dim, 17, int(policy_cfg.n_layers), int(policy_cfg.n_hidden))
        searched = sum(p.numel() for p in policy.parameters())
        write_json_atomic(
            os.path.join(out_dir, "prune.json"),
            prune.record(
                host.vocabulary(),
                policy_params={"all_ops": full, "searched": searched},
                explore_copies_per_image={"all_ops": 17, "searched": len(outputs)},
            ),
        )
        print(f"[adaaug_search] h: {full} parameters over all 17 outputs, {searched} over the {len(outputs)} searched")
    final = os.path.join(out_dir, "final")
    os.makedirs(final, exist_ok=True)
    gf_config(mm_cfg).dump(os.path.join(final, "gf_config.py"))
    meta = {
        "dataset": cfg.dataset.name,
        "task": task,
        "recipe": cfg.choices.get("recipe"),
        "feature_dim": feature_dim,
        "n_hidden": int(policy_cfg.n_hidden),
        "train_size": size,
        "batch_size": batch_size,
        "explore_split": str(search["explore_split"]),
        "explore_images": len(explore_set),
        "seed": seed,
        "pass_counts": adaaug.pass_counts(
            train_images=size,
            batch_size=batch_size,
            epochs=int(search["epochs"]),
            search_freq=int(search["search_freq"]),
            explore_batch=explore_size,
            n_outputs=len(outputs),
        ),
    }
    print(f"[adaaug_search] pass counts: {meta['pass_counts']}")
    config = adaaug.SearchConfig(
        epochs=int(search["epochs"]),
        search_freq=int(search["search_freq"]),
        explore_batch=explore_size,
        policy_lr=float(search["policy_lr"]),
        policy_weight_decay=float(search["policy_weight_decay"]),
        grad_clip=float(search["grad_clip"]),
        lr_min_ratio=float(search["lr_min_ratio"]),
        save_every=int(search["save_every"]),
        log_interval=int(search["log_interval"]),
        seed=seed,
    )
    adaaug.search(
        segmentor,
        policy,
        host,
        ops,
        epoch_batches,
        explore_batch,
        normalize=backend._normalize,
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
        config=config,
        out_dir=out_dir,
        meta=meta,
        ignore_index=LABEL_IGNORE_INDEX,
        log=lambda line: print(f"[adaaug_search] {line}", flush=True),
    )
    print(f"[adaaug_search] wrote {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
