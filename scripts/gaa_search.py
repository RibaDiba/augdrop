#!/usr/bin/env python
"""Greedy AutoAugment's policy search on one segmentation dataset, before training.

The search loop is ``src/greedyaa/`` (Algorithm 1: which chains to try); each trial is a
child network trained on one chain and scored on held-out training images
(``tasks/mmseg/greedyaa_trial.py``). This is glue: compose the config, draw the train and
score images, run the trials, write the policy. Nothing is computed here.

    python scripts/gaa_search.py --dataset cityscapes --backbone segformer --cluster nexus \\
        [--prune-ops "greedy_aa.Rotate_pos ..."] [host.search.levels=2 ...]

The child is the task the search is composed with (``--backbone`` / ``--task``, as
train.py; the run's own, as the paper's children share the trained network's
architecture) with that dataset's recipe. ``host.search`` (configs/host/
greedy_autoaugment.yaml) holds the budget. ``train_size`` + ``score_size`` images of the
train split are drawn once (seeded), disjoint; every child trains on the first, augmented
by its chain, and is scored (mIoU) on the second, clean. The val split is never read.

**Trials run in parallel**, ``host.search.gpus`` at a time, each a subprocess of this
script (``--trial <dir>``) on one GPU (``CUDA_VISIBLE_DEVICES``), so no state passes from
one child to the next. Only this process writes ``trials.jsonl`` (one line a scored
trial). A level's trials are independent; the next level waits for all of them (its base
is their best).

**A search over a subset** (the search-then-train recipe: a trunk trains with no policy to
25%, ``scripts/nms_prune.py`` turns its probe into P, then each arm searches the ops left):
``--prune-ops "A B ..."`` leaves those vocab ops out (every (op, level) pair that applies
one), ``--random-prune K`` (``--random-prune-seed S``) K more drawn from the probed ops,
the random-K null; both through the training run's own prune (``dropaug.config.prune_of``).
A pruned search has fewer trials a level (``prune.json``: ``trials_per_level``).

The output directory, ``<work-root>/gaa_search_<dataset>_<backbone>[_prune-...][_<tag>]``:
``search_config.json`` (settings and the drawn ids), ``config.yaml`` (the composed
config), ``trials.jsonl``, ``trials/<level>-<index>/`` (each child's spec, log and
result), ``level_<c>.json`` after each level (the policy from levels 1..c: ``level_1.json``
is the k = 1 search's), ``final.json`` at the end (the host's ``policy_file``;
``slurm/train.sbatch`` looks for ``<WORK_DIR>/gaa_search_<dataset>_<backbone>/final.json``),
``prune.json`` with a prune. A directory with a ``final.json`` is refused; one without
resumes: scored trials are kept, a trial whose child finished but went unrecorded is read
back from its ``result.json``, the rest rerun. A failed child stops the search (after the
running ones finish); rerunning retries it.

``--dry-run`` composes and checks the config and prints the plan (trial counts), building
nothing. ``--time-trial`` trains and scores one child (level 1's first candidate, under
``time_trial/``, not recorded) and prints its seconds and the search's projected hours.
Further ``key=value`` words are Hydra overrides of the composed config.
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

HOST = "greedy_autoaugment"
METRIC = "mIoU"


def die(message: str):
    print(f"[gaa_search] ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def log(message: str):
    print(f"[gaa_search] {message}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Search a Greedy AutoAugment policy on one segmentation dataset.",
        epilog="Further positional key=value words are Hydra overrides, e.g. host.search.levels=1.",
    )
    p.add_argument("--dataset", help="a configs/dataset name")
    p.add_argument("--backbone", help="the child's task by its backbone (train.py --backbone; default the config's task)")
    p.add_argument("--task", help="the child's configs/task name (alternative to --backbone)")
    p.add_argument("--cluster", default="local", help="a configs/cluster name (data and checkpoint paths)")
    p.add_argument("--data-root", help="override the dataset's data_root")
    p.add_argument(
        "--output-dir",
        help="where the search goes (default <work-root>/gaa_search_<dataset>_<backbone>[_prune-...][_<search-tag>])",
    )
    p.add_argument("--work-root", default=os.path.join(REPO, "experiments"), help="the default output dir's parent")
    p.add_argument("--search-tag", help="the default output dir's suffix, for a second search of the same setup")
    p.add_argument(
        "--prune-ops",
        metavar='"OP OP ..."',
        help="vocab ops the search leaves out, space-separated in one word (prune.ops): NMS's P",
    )
    p.add_argument("--random-prune", type=int, metavar="K", help="K more ops left out, drawn from the probed ops (prune.random)")
    p.add_argument("--random-prune-seed", type=int, metavar="S", help="the draw's seed (prune.seed; default the config's seed)")
    p.add_argument("--gpus", type=int, help="children at once, one per GPU (default host.search.gpus)")
    p.add_argument("--dry-run", action="store_true", help="check the config and print the plan, build nothing")
    p.add_argument("--time-trial", action="store_true", help="train and score one child, print the projected search time")
    p.add_argument("--trial", metavar="DIR", help=argparse.SUPPRESS)  # internal: one child, in DIR
    p.add_argument("overrides", nargs="*", metavar="key=value", help="overrides of the config")
    args = p.parse_args(argv)
    if not args.trial and not args.dataset:
        p.error("--dataset is required")
    if args.backbone and args.task:
        p.error("give --backbone or --task, not both")
    return args


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

    from dropaug.config import ConfigError, check, compose, group_options, search_of, task_for_backbone

    for group, value in (("dataset", args.dataset), ("cluster", args.cluster), ("task", args.task)):
        if value and value not in group_options(group):
            die(f"--{group} {value!r} is not one of configs/{group}/: {group_options(group)}")
    base = [f"cluster={args.cluster}", f"dataset={args.dataset}", f"host={HOST}", "augment=false"] + prune_overrides(args)
    try:
        if args.backbone:
            base.append(f"task={task_for_backbone(args.backbone)}")
        elif args.task:
            base.append(f"task={args.task}")
        if args.data_root:
            data_root = os.path.abspath(args.data_root)
            if "'" in data_root or not os.path.isdir(data_root):
                die(f"--data-root {data_root} is not a directory Hydra can take")
            base.append(f"dataset.data_root='{data_root}'")
        cfg = compose(base + list(args.overrides))
        if cfg.task.backend != "mmseg":
            die(f"task {cfg.choices.get('task')!r} is {cfg.task.backend}; the search is wired for mmseg segmentors only")
        for warning in check(cfg, training=False):
            print(f"[gaa_search] WARNING: {warning}", file=sys.stderr)
    except (ConfigError, ValueError, HydraException) as error:
        die(str(error))
    search = search_of(cfg)
    if search is None:
        die("the greedy_autoaugment host config has no search: block")
    OmegaConf.resolve(cfg)
    return cfg, search


def resolve_search_prune(cfg):
    """``(prune, candidates, tag)``: the training run's prune machinery on the search's
    config, and the (search op, level) pairs it leaves."""
    from dropaug.config import ConfigError, prune_of, prune_tag
    from hosts.greedy_autoaugment import candidates

    try:
        prune = prune_of(cfg)
        tag = prune_tag(cfg, prune)
    except (ConfigError, ValueError) as error:
        die(str(error))
    kept = candidates(prune.ops)
    if not kept:
        die("the prune leaves the search no (op, level) pair")
    return prune, kept, tag


#: The settings a resumed search must share with the one that wrote the directory.
_RESUME_KEYS = ("levels", "epochs", "train_size", "score_size", "resolution", "batch_size", "seed")


def _write_json(path: str, record) -> None:
    """Write ``record`` as JSON to ``path`` atomically."""
    from stats import write_json_atomic

    write_json_atomic(path, record)


def _read_records(path: str) -> list:
    """The scored-trial records in the JSONL file ``path`` (empty if it does not exist)."""
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _visible_gpus(count: int) -> list:
    """The first ``count`` visible GPU ids (``CUDA_VISIBLE_DEVICES``, else 0..count-1); dies if fewer."""
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    gpus = visible or [str(i) for i in range(count)]
    if len(gpus) < count:
        die(f"host.search.gpus is {count}, but only {len(gpus)} GPU(s) are visible ({gpus})")
    return gpus[:count]


# --- one child (``--trial``) ---------------------------------------------------------------


def run_child(trial_dir: str) -> int:
    """Train and score the child ``<trial_dir>/spec.json`` describes; write ``result.json``."""
    from omegaconf import OmegaConf

    from tasks.mmseg.greedyaa_trial import child_config, run_trial

    with open(os.path.join(trial_dir, "spec.json")) as f:
        spec = json.load(f)
    with open(spec["search_config"]) as f:
        search_config = json.load(f)
    cfg = OmegaConf.load(spec["config"])
    settings = search_config["settings"]
    mm_cfg = child_config(
        cfg,
        spec["chain"],
        os.path.join(trial_dir, "work"),
        train_ids=search_config["train_ids"],
        score_ids=search_config["score_ids"],
        epochs=float(settings["epochs"]),
        batch_size=settings.get("batch_size"),
        resolution=float(settings["resolution"]),
        seed=int(settings["seed"]),
        num_workers=int(settings.get("num_workers", 4)),
    )
    start = time.time()
    metrics = run_trial(mm_cfg)
    if METRIC not in metrics:
        print(f"[gaa_search] child metrics {sorted(metrics)} hold no {METRIC}", file=sys.stderr)
        return 2
    _write_json(
        os.path.join(trial_dir, "result.json"),
        {
            "key": spec["key"],
            "chain": spec["chain"],
            "score": metrics[METRIC],
            "metrics": metrics,
            "iters": int(mm_cfg.train_cfg.max_iters),
            "batch_size": int(mm_cfg.train_dataloader.batch_size),
            "seconds": round(time.time() - start, 1),
        },
    )
    return 0


def _launch(trial, out_dir: str, gpu: str, sub: str = "trials"):
    """Start one child subprocess for ``trial`` on ``gpu``; returns ``(proc, logfile, trial_dir)``."""
    trial_dir = os.path.join(out_dir, sub, trial.key)
    os.makedirs(trial_dir, exist_ok=True)
    _write_json(
        os.path.join(trial_dir, "spec.json"),
        {
            "key": trial.key,
            "level": trial.level,
            "index": trial.index,
            "chain": [list(step) for step in trial.chain],
            "config": os.path.join(out_dir, "config.yaml"),
            "search_config": os.path.join(out_dir, "search_config.json"),
        },
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    logfile = open(os.path.join(trial_dir, "log.txt"), "w")
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--trial", trial_dir],
        env=env,
        stdout=logfile,
        stderr=subprocess.STDOUT,
        cwd=REPO,
    )
    return proc, logfile, trial_dir


def _result(trial, trial_dir: str):
    """The child's ``result.json`` if it exists and is for ``trial``'s chain, else None."""
    path = os.path.join(trial_dir, "result.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        result = json.load(f)
    if [list(s) for s in trial.chain] != result.get("chain"):
        return None
    return result


# --- the search ------------------------------------------------------------------------------


def run_level(search, pending, out_dir: str, gpus: list, records_path: str) -> list:
    """Run ``pending`` trials, ``len(gpus)`` at a time; record each as it finishes. Returns
    the keys of the trials that failed (the rest are recorded)."""
    queue = list(pending)
    running = {}  # gpu -> (trial, proc, logfile, dir)
    failed = []
    with open(records_path, "a") as records:
        while queue or running:
            for gpu in gpus:
                if gpu not in running and queue and not failed:
                    trial = queue.pop(0)
                    proc, logfile, trial_dir = _launch(trial, out_dir, gpu)
                    running[gpu] = (trial, proc, logfile, trial_dir)
            if failed and not running:
                break
            time.sleep(2.0)
            for gpu, (trial, proc, logfile, trial_dir) in list(running.items()):
                if proc.poll() is None:
                    continue
                logfile.close()
                del running[gpu]
                result = _result(trial, trial_dir) if proc.returncode == 0 else None
                if result is None:
                    failed.append(trial.key)
                    log(f"trial {trial.key} failed (exit {proc.returncode}); see {trial_dir}/log.txt")
                    continue
                search.record(trial, float(result["score"]))
                records.write(json.dumps(trial.record(result["score"], seconds=result.get("seconds"))) + "\n")
                records.flush()
                log(
                    f"level {trial.level} trial {trial.index + 1}/{len(search.candidates)} {trial.chain}: "
                    f"{METRIC} {result['score']:.2f} ({result.get('seconds')} s)"
                )
    return failed


def write_policy(path: str, search, settings: dict, meta: dict, upto_level: int) -> None:
    from hosts.greedy_autoaugment import save_policy

    save_policy(
        path,
        search.policy(int(settings["top"]), upto_level=upto_level),
        set_size=int(settings["set_size"]),
        alpha=float(settings["alpha"]),
        meta={**meta, "upto_level": int(upto_level)},
    )


def prepare(cfg, settings: dict, out_dir: str, candidates, prune) -> dict:
    """``search_config.json``: the settings and the drawn train / score ids, written once;
    a resumed search must match it."""
    import numpy as np
    from omegaconf import OmegaConf

    path = os.path.join(out_dir, "search_config.json")
    pruned = sorted(prune.ops)
    if os.path.isfile(path):
        with open(path) as f:
            existing = json.load(f)
        mismatched = [k for k in _RESUME_KEYS if existing["settings"].get(k) != settings.get(k)]
        if existing.get("pruned_ops") != pruned:
            mismatched.append("pruned_ops")
        if existing.get("task") != cfg.choices.get("task") or existing.get("dataset") != cfg.dataset.name:
            mismatched.append("task / dataset")
        if mismatched:
            die(f"{out_dir} holds a search with other settings ({mismatched}); pick another --output-dir")
        return existing

    from tasks.mmseg.greedyaa_trial import train_split_size

    size = train_split_size(cfg, os.path.join(out_dir, "_split"))
    need = int(settings["train_size"]) + int(settings["score_size"])
    if need > size:
        die(f"train_size + score_size = {need} images, but the train split has {size}")
    order = np.random.default_rng(int(settings["seed"])).permutation(size)
    train_ids = sorted(int(i) for i in order[: int(settings["train_size"])])
    score_ids = sorted(int(i) for i in order[int(settings["train_size"]) : need])
    record = {
        "dataset": cfg.dataset.name,
        "task": cfg.choices.get("task"),
        "backbone": cfg.task.backbone,
        "recipe": cfg.choices.get("recipe"),
        "settings": settings,
        "pruned_ops": pruned,
        "trials_per_level": len(candidates),
        "train_split_size": size,
        "train_ids": train_ids,
        "score_ids": score_ids,
    }
    OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))
    _write_json(path, record)
    return record


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.trial:
        return run_child(os.path.abspath(args.trial))

    from greedyaa import GreedySearch, trial_counts

    cfg, search_block = compose_config(args)
    prune, cands, tag = resolve_search_prune(cfg)
    settings = {k: search_block[k] for k in sorted(search_block)}
    if args.gpus is not None:
        settings["gpus"] = int(args.gpus)
    levels = int(settings["levels"])
    default = f"gaa_search_{cfg.dataset.name}_{cfg.task.backbone}{tag}" + (f"_{args.search_tag}" if args.search_tag else "")
    out_dir = os.path.abspath(args.output_dir or os.path.join(args.work_root, default))
    if os.path.exists(os.path.join(out_dir, "final.json")):
        die(f"{out_dir} already holds a finished search (final.json); pick another --output-dir")
    counts = trial_counts(len(cands), levels)
    from hosts.greedy_autoaugment import candidates as all_candidates

    full = len(all_candidates())
    log(
        f"dataset {cfg.dataset.name} ({cfg.dataset.data_root}), child {cfg.choices.get('task')} "
        f"(recipe {cfg.choices.get('recipe')}), out {out_dir}"
    )
    log(f"search: {settings}")
    log(
        f"trials: {counts['per_level']} a level x {levels} levels = {counts['total']} "
        f"({full} a level over all ops)"
        + (f"; left out: {list(prune.ops)}" if prune else "")
    )
    print(f"OUT_DIR={out_dir}")
    if args.dry_run:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    prepare(cfg, settings, out_dir, cands, prune)
    if prune:
        from hosts.greedy_autoaugment import VOCAB_NAMES

        _write_json(
            os.path.join(out_dir, "prune.json"),
            prune.record(VOCAB_NAMES, trials_per_level={"all_ops": full, "searched": len(cands)}),
        )
    gpus = _visible_gpus(int(settings["gpus"]))

    search = GreedySearch(cands, levels)
    if args.time_trial:
        trial = search.pending()[0]
        start = time.time()
        proc, logfile, trial_dir = _launch(trial, out_dir, gpus[0], sub="time_trial")
        proc.wait()
        logfile.close()
        result = _result(trial, trial_dir) if proc.returncode == 0 else None
        if result is None:
            die(f"the timed child failed (exit {proc.returncode}); see {trial_dir}/log.txt")
        seconds = time.time() - start
        hours = seconds * counts["total"] / len(gpus) / 3600.0
        log(
            f"one child: {seconds:.0f} s wall ({result['seconds']} s train + score, {result['iters']} iters at "
            f"batch {result['batch_size']}), {METRIC} {result['score']:.2f}"
        )
        log(f"projected search: {counts['total']} trials / {len(gpus)} GPUs = {hours:.1f} h")
        return 0

    records_path = os.path.join(out_dir, "trials.jsonl")
    search = GreedySearch(cands, levels, _read_records(records_path))
    meta = {
        "source": "search",
        "dataset": cfg.dataset.name,
        "task": cfg.choices.get("task"),
        "levels": levels,
        "trials_per_level": len(cands),
        "pruned_ops": " ".join(sorted(prune.ops)),
    }
    while True:
        # Levels finished before a stop get their policy now if it was never written.
        for level in range(1, min(search.level, levels + 1)):
            path = os.path.join(out_dir, f"level_{level}.json")
            if not os.path.isfile(path):
                write_policy(path, search, settings, meta, level)
                log(f"wrote {path}")
        if search.done:
            break
        pending = search.pending()
        # A child that finished but went unrecorded (the driver stopped first) is read back.
        with open(records_path, "a") as records:
            for trial in list(pending):
                result = _result(trial, os.path.join(out_dir, "trials", trial.key))
                if result is not None:
                    search.record(trial, float(result["score"]))
                    records.write(json.dumps(trial.record(result["score"], seconds=result.get("seconds"))) + "\n")
                    pending.remove(trial)
        if not pending:
            # Read-back finished the level; the top of the loop writes its policy and re-checks done.
            continue
        base = search.base(search.level)
        log(
            f"level {search.level}/{levels}: {len(pending)} trials to run"
            + (f", base {list(base.chain)} ({METRIC} {search.score(base):.2f})" if base is not None else "")
        )
        failed = run_level(search, pending, out_dir, gpus, records_path)
        if failed:
            die(f"{len(failed)} trial(s) failed ({failed[:5]}); rerun to retry them")
    write_policy(os.path.join(out_dir, "final.json"), search, settings, meta, levels)
    log(f"wrote {os.path.join(out_dir, 'final.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
