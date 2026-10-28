#!/usr/bin/env python
"""The gradient probe, post hoc, on one checkpoint: one frozen sweep -> G -> R, with
Sensaug's logs and dropaug's own G / R files.

dropaug's form of Sensaug's ``scripts/compute_grad_corr.py``. That script rebuilt a
Runner and drove the grad-corr hooks' private methods through a proxy; here the probe
is a library (``probe``, ``stats``), so this is glue: rebuild the model, load the
checkpoint, sweep, analyze, write. Nothing is computed here. The checkpoint comes in one
of two ways:

    # a dropaug run, rebuilt from its dropaug.yaml and probed with its own host's ops
    python scripts/probe.py --work-dir experiments/<run> --checkpoint <run>/best_mIoU_iter_8000.pth \
        --output-dir experiments/<run>/probe/best_mIoU_iter_8000 [flags] [probe.fd_step=0.1 ...]

    # any mmseg checkpoint (e.g. a Sensaug experiment of any --aug-type), probed with --method's ops
    python scripts/probe.py --checkpoint <exp>/best_mIoU_iter_8000.pth --backbone segformer \
        --dataset cityscapes --method randaugment --output-dir <exp>/probe/best_mIoU_iter_8000_randaugment \
        [--cluster nexus] [--data-root /path/to/cityscapes] [--magnitudes-path F] [--non-strict] \
        [--dry-run] [flags] [probe.analysis.n_min=50 vocab=... ...]

flags: [--magnitude 0.5] [--magnitude-mode mode] [--n-probe 500] [--probe-seed 0]
[--batch-size 1] [--split val]. They are shorthand for ``probe.*`` overrides (keys
documented in ``configs/probe/default.yaml``); the positional ``key=value`` words come
after them. On a run they are dotted overrides of its saved config, and groups cannot be
switched. On a checkpoint they are Hydra overrides of the composed config, so a group
can be switched (``vocab=randaug_default`` with ``--method uniform``); ``check`` refuses
a combination that doesn't fit.

The checkpoint mode. ``--method`` is a ``configs/host`` option (sensaug, randaugment,
trivialaugment, autoaugment, uniform, madaug, or a ``*_diff`` host: trivialaugment_diff,
randaugment_diff, autoaugment_diff, augmix_diff, smartsampling_diff). It is the probe's choice and need not be how the
model was trained: a Sensaug ``none`` or ``augmix`` model is probed with whichever
method's ops are named. The method's host config gives:

- the ops, its whole vocab: sensaug and uniform the 32 DIFF32 ops, randaugment and
  trivialaugment 13 each, autoaugment 10, the ``*_diff`` hosts 22 each, autoaugment_diff 15,
  smartsampling_diff 13, madaug 16 (R is over all of them)
- the magnitude, its own (``tasks.magnitudes``): RandAugment's bin M, TrivialAugment's
  uniform bin per image, AutoAugment's policy bins, randaugment_diff its published level
  for ``--dataset``, AugMix a bin uniform on its first 3; sensaug the SA snapshot (below);
  madaug its policy's level on each probe image, the policy read from the checkpoint
  (``madaug_policy``; a checkpoint without one, any other method's, gets probe.magnitude);
  uniform, and every method under ``--magnitude-mode fixed``, ``probe.magnitude``
- the fade, its ``fade:`` block: randaugment's and autoaugment's ops are all faded in,
  trivialaugment's only where there is no derivative (its enhance ops go through their
  exact tangent, its geometric ops through finite differences), DIFF32 differentiated

The model is the ``--backbone`` (or ``--task``) task with the recipe for that (task,
dataset) pair applied, i.e. the model Sensaug trained on that pair. It is loaded
strictly: a missing, unexpected or mis-shaped key exits 2 with mmengine's list, since a
partly loaded model would be probed with random weights where the keys missed.
``--non-strict`` only logs them. ``--data-root`` is the dataset's directory (default:
``<cluster.data_root>/<dataset>``). The same ``--n-probe`` and ``--probe-seed`` give every
model on a dataset the same probe images (``ProbeSet.build`` takes the ids sorted), so
their Rs are on one set of images. ``--dry-run`` composes and checks the config, resolves
the ops, fade and magnitudes, prints them, and exits before any model is built.

Writes into ``--output-dir`` (not the run directory, where the in-training probe's
logs already are):

    aug_gradient_log.txt     one JSONL record per batch (APPENDED to: see --allow-existing)
    corr_matrix_log.json     R_raw / R_scalenorm / loadings ... (Sensaug's shape + dropaug's fields)
    corr_bootstrap_log.txt   per-cell CIs and FDR q-values
    corr_redundancy_log.txt  per-op red(a), scored but applied to nothing (no training loop)
    G.npz, R.json            dropaug.io files, with vocab / probe-set / checkpoint hashes
    probe_set.json           the images the columns of G are
    probe_config.yaml        the config probed with (not dropaug.yaml: this is no run dir)

Magnitudes of the sensaug host. With ``probe.magnitude_mode`` other than ``fixed``, each
op is probed at the SA levels of the latest ``corr_magnitudes.json`` record (written by
the SA loop, Sensaug's or dropaug's) at or before the checkpoint's iteration, as the
live hook does (``sa.probe_magnitudes``); ops the snapshot lacks, or a run without the
file, use ``probe.magnitude``. The file is ``--magnitudes-path``, else
``<work-dir>/corr_magnitudes.json`` on a run and ``<checkpoint's dir>/corr_magnitudes.json``
on a checkpoint. No other method reads one. The source and mode are recorded in every
output.

Exit status 2 on a bad invocation: a mix of the two modes, a missing work dir,
``dropaug.yaml``, checkpoint or magnitudes file, a config that doesn't compose or fit
together, an output dir that already holds a gradient log, a backend that is not wired,
a checkpoint that doesn't fit the model.

Differences from compute_grad_corr.py: the model is rebuilt from ``dropaug.yaml`` or
from the task + recipe, not a dumped mmengine .py; no Runner (``MMSegBackend`` builds
the model and the val dataset); the checkpoint is always explicit (slurm/probe.sbatch
resolves best / last); the probe set is the run's ``probe_set.json`` when it matches,
else built from ``--n-probe`` and ``--probe-seed`` (Sensaug swept the whole val set); an
existing gradient log is refused unless ``--allow-existing``; the corr_magnitudes.json
record is the latest one at or before the checkpoint's iteration (Sensaug: the file's
seed, whichever); no ``--ref-magnitude`` (``probe.magnitude`` is it).
"""

import argparse
import json
import logging
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

GRADIENT_LOG = "aug_gradient_log.txt"
PROBE_SET_FILE = "probe_set.json"
CORR_MAGNITUDES_FILE = "corr_magnitudes.json"
CONFIG_FILE = "probe_config.yaml"

#: The two ways to name the checkpoint (``mode_of``).
MODE_RUN = "run"
MODE_CHECKPOINT = "checkpoint"
#: Flags that only the checkpoint mode reads.
CHECKPOINT_FLAGS = ("backbone", "task", "dataset", "method", "cluster", "data_root", "non_strict")


def die(message: str):
    print(f"probe.py: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="The gradient probe on one checkpoint: frozen sweep -> G -> R. Give either --work-dir "
        "(a dropaug run) or --backbone/--task, --dataset and --method (any mmseg checkpoint).",
        epilog="Further positional key=value words are overrides of the config, e.g. probe.fd_step=0.1.",
    )
    p.add_argument("--checkpoint", required=True, help="checkpoint file to probe")
    p.add_argument("--output-dir", required=True, help="where the probe's logs and G / R files go")

    run = p.add_argument_group("a dropaug run")
    run.add_argument("--work-dir", help="the run directory (holds dropaug.yaml)")

    ckpt = p.add_argument_group("any mmseg checkpoint")
    ckpt.add_argument("--backbone", help="the task whose yaml says backbone: NAME (segformer, pspnet, ...)")
    ckpt.add_argument("--task", help="a configs/task option, e.g. mmseg_segformer_b0 (alternative to --backbone)")
    ckpt.add_argument("--dataset", help="a configs/dataset option: the probe's images, and with the task the recipe")
    ckpt.add_argument("--method", help="a configs/host option whose ops, magnitudes and fade the probe uses")
    ckpt.add_argument("--cluster", help="a configs/cluster option: data paths (default: local)")
    ckpt.add_argument("--data-root", help="the dataset's directory (default: <cluster.data_root>/<dataset>)")
    ckpt.add_argument(
        "--non-strict",
        action="store_true",
        help="load a checkpoint that doesn't fit the model, only logging the missing / unexpected keys",
    )

    p.add_argument("--magnitude", type=float, help="probe.magnitude: a strength in [0, 1] for every op")
    p.add_argument("--magnitude-mode", help="probe.magnitude_mode: mode | sampled_shared | sampled_independent | fixed")
    p.add_argument(
        "--magnitudes-path",
        help=f"the sensaug method's SA snapshot ({CORR_MAGNITUDES_FILE}); default: the run's, or the checkpoint's dir's",
    )
    p.add_argument("--n-probe", type=int, help="probe.n_probe: images in the probe set (default: the config's)")
    p.add_argument("--probe-seed", type=int, help="probe.probe_seed")
    p.add_argument("--batch-size", type=int, help="probe.batch_size (also the bootstrap's cluster size)")
    p.add_argument("--split", help="probe.split: the clean split to probe (val | test)")
    p.add_argument(
        "--allow-existing",
        action="store_true",
        help=f"write into an output dir that already holds {GRADIENT_LOG}. The log is APPENDED to, so "
        "R recomputed from it would pool two Gs",
    )
    p.add_argument("--dry-run", action="store_true", help="resolve and print the ops, fade and magnitudes, then exit")
    p.add_argument("overrides", nargs="*", metavar="key=value", help="overrides of the config")
    return p.parse_args(argv)


def mode_of(args) -> str:
    """``MODE_RUN`` or ``MODE_CHECKPOINT``; exits 2 on a mix of the two or half of one."""
    given = [f"--{flag.replace('_', '-')}" for flag in CHECKPOINT_FLAGS if getattr(args, flag)]
    if args.work_dir:
        if given:
            die(f"{', '.join(given)} name a checkpoint's model; a run (--work-dir) is rebuilt from its dropaug.yaml")
        return MODE_RUN
    if args.backbone and args.task:
        die("give --backbone or --task, not both")
    missing = [
        flag
        for flag, value in (("--backbone or --task", args.backbone or args.task), ("--dataset", args.dataset), ("--method", args.method))
        if not value
    ]
    if missing:
        die(f"give --work-dir (a dropaug run), or a checkpoint's model: missing {', '.join(missing)}")
    return MODE_CHECKPOINT


def overrides_of(args) -> list:
    flags = {
        "magnitude": args.magnitude,
        "magnitude_mode": args.magnitude_mode,
        "n_probe": args.n_probe,
        "probe_seed": args.probe_seed,
        "batch_size": args.batch_size,
        "split": args.split,
    }
    out = [f"probe.{key}={value}" for key, value in flags.items() if value is not None]
    return out + list(args.overrides)


def compose_overrides(args) -> list:
    """The Hydra overrides that pick the checkpoint mode's groups, as train.py's flags do."""
    from dropaug.config import group_options, task_for_backbone

    for group, value in (("task", args.task), ("dataset", args.dataset), ("host", args.method), ("cluster", args.cluster)):
        if value and value not in group_options(group):
            flag = "--method" if group == "host" else f"--{group}"
            die(f"{flag} {value!r} is not one of configs/{group}/: {group_options(group)}")
    out = [f"cluster={args.cluster or 'local'}"]
    out.append(f"task={task_for_backbone(args.backbone) if args.backbone else args.task}")
    out += [f"dataset={args.dataset}", f"host={args.method}"]
    if args.data_root:
        data_root = os.path.abspath(args.data_root)
        if "'" in data_root:
            die(f"--data-root {data_root}: a path with a quote can't be passed to Hydra")
        if not os.path.isdir(data_root):
            die(f"--data-root {data_root} is not a directory")
        out.append(f"dataset.data_root='{data_root}'")
    return out


def config_of(args, mode: str, work_dir):
    """The run's saved config, or the checkpoint mode's composed and checked one."""
    from hydra.errors import HydraException

    from dropaug.config import ConfigError, check, compose, load

    try:
        if mode == MODE_RUN:
            return load(work_dir, overrides_of(args))
        cfg = compose(compose_overrides(args) + overrides_of(args))
        for warning in check(cfg, training=False):
            print(f"[probe] WARNING: {warning}", file=sys.stderr)
        return cfg
    except (ConfigError, ValueError, HydraException) as error:
        die(str(error))


def make_backend(cfg, work_dir: str, split: str, checkpoint: str, strict: bool = False):
    """``(backend, ckpt_meta)``: the task backend with its model built and the
    checkpoint loaded. The one place that touches a training framework (so a test
    replaces just this)."""
    name = cfg.task.backend
    if name != "mmseg":
        die(
            f"task backend {name!r} is not wired into scripts/probe.py (only mmseg is); "
            f"a cls run can be probed from python with tasks.cls.ClsBackend + probe.sweep"
        )
    from mmengine.utils import import_modules_from_strings

    from tasks.mmseg.backend import MMSegBackend
    from tasks.mmseg.config import build

    mm_cfg = build(cfg, work_dir)
    # registers GpuAugSegDataPreProcessor and the task's own modules before MODELS.build
    import_modules_from_strings(**mm_cfg.custom_imports)
    backend = MMSegBackend(mm_cfg, split=split)
    backend.build_model()
    try:
        meta = backend.load_ckpt(checkpoint, strict=strict)
    except RuntimeError as error:
        if not strict:
            raise
        die(
            f"{checkpoint} doesn't fit the {cfg.task.backbone} model built for {cfg.dataset.name} "
            f"(task {cfg.choices.get('task')}, recipe {cfg.choices.get('recipe')}):\n{error}\n"
            f"  (a different backbone or dataset? --non-strict loads it anyway)"
        )
    return backend, meta


def pick_probe_set(work_dir, data, n_probe, probe_seed):
    """The run's own ``probe_set.json`` when it is the set these settings would build
    (same source, size and seed), verified against the dataset, so this G is on the
    images the in-training Gs were; else (and with no run) a fresh one."""
    import probe

    path = None if work_dir is None else os.path.join(work_dir, PROBE_SET_FILE)
    if path is not None and os.path.exists(path):
        saved = probe.ProbeSet.load(path)
        want = len(data.ids) if n_probe is None else int(n_probe)
        same_seed = n_probe is None or saved.seed == probe_seed
        if saved.source == data.source and len(saved) == want and same_seed:
            saved.verify(data.ids)
            return saved, "the run's probe_set.json"
    return probe.ProbeSet.build(data.ids, n_probe, seed=probe_seed, source=data.source), "built"


def snapshot_path(args, mode: str, work_dir, checkpoint: str):
    """The sensaug method's corr_magnitudes.json, or None when there is none."""
    if args.magnitudes_path:
        path = os.path.abspath(args.magnitudes_path)
        if not os.path.isfile(path):
            die(f"--magnitudes-path {path} is not a file")
        return path
    path = os.path.join(work_dir if mode == MODE_RUN else os.path.dirname(checkpoint), CORR_MAGNITUDES_FILE)
    return path if os.path.exists(path) else None


def sa_snapshot(path, ckpt_iter):
    """``({op: {levels, probs, mode}}, iter)`` from the latest corr_magnitudes.json
    record at or before ``ckpt_iter`` (the latest when it is unknown), else ``({}, None)``."""
    if path is None:
        return {}, None
    with open(path) as f:
        records = json.load(f)
    if ckpt_iter is not None:
        records = [r for r in records if int(r["iter"]) <= int(ckpt_iter)]
    if not records:
        return {}, None
    record = max(records, key=lambda r: int(r["iter"]))
    return record["magnitudes"], int(record["iter"])


def dry_run_report(cfg, ops, fade, faded, fade_at, magnitudes, snapshot_file) -> str:
    """What a sweep would probe: each op's row method (``probe.resolve_methods``) and
    one image's magnitude."""
    import numpy as np
    import probe

    methods = probe.resolve_methods(ops, fade)
    drawn = magnitudes.draw(ops, 1, np.random.default_rng(0))
    lines = [
        f"[probe] dry run: method {cfg.host.name}, task {cfg.choices.get('task')}, dataset {cfg.dataset.name} "
        f"({cfg.dataset.data_root}), recipe {cfg.choices.get('recipe')}",
        f"[probe] {len(ops)} ops, {len(faded)} faded at a={fade_at:g}; magnitudes "
        f"{magnitudes.source}/{magnitudes.mode}" + (f" (SA snapshot {snapshot_file})" if snapshot_file else ""),
    ]
    for name in ops:
        value = drawn.get(name)
        shown = "-" if value is None else f"{float(np.asarray(value).reshape(-1)[0]):.4g}"
        lines.append(f"    {name:40s} {methods[name]:11s} {shown}")
    return "\n".join(lines)


def policy_levels(host, magnitude_mode, checkpoint, backend, probe_set, data, pcfg):
    """A policy method's (madaug) own levels on every probe image, from the policy saved
    in the checkpoint; None for any other method, under ``fixed``, or when the checkpoint
    holds no policy (then every op is at probe.magnitude, and this says so)."""
    import sa

    if not hasattr(host, "method_magnitudes") or magnitude_mode == sa.MODE_FIXED:
        return None
    from tasks.mmseg.madaug_hooks import policy_from_checkpoint, probe_set_levels

    device = next(backend.model.parameters()).device
    policy = policy_from_checkpoint(checkpoint, device)
    if policy is None:
        print(
            f"[probe] NOTE: {checkpoint} holds no MADAug policy; every op at "
            f"probe.magnitude={pcfg.get('magnitude')!r}",
            file=sys.stderr,
        )
        return None
    return probe_set_levels(
        backend, policy, probe_set, data.dataset, int(pcfg.get("batch_size", 1)), int(pcfg.get("num_workers", 0))
    )


def run(args) -> int:
    mode = mode_of(args)
    out_dir = os.path.abspath(args.output_dir)
    checkpoint = os.path.abspath(args.checkpoint)
    work_dir = os.path.abspath(args.work_dir) if mode == MODE_RUN else None

    from dropaug.config import RUN_FILE, fade_of, magnitudes_of, to_python

    if mode == MODE_RUN:
        if not os.path.isdir(work_dir):
            die(f"--work-dir {work_dir} does not exist")
        if not os.path.exists(os.path.join(work_dir, RUN_FILE)):
            die(f"{os.path.join(work_dir, RUN_FILE)} does not exist; {work_dir} was not trained by train.py")
    if not os.path.isfile(checkpoint):
        die(f"--checkpoint {checkpoint} is not a file")
    log_path = os.path.join(out_dir, GRADIENT_LOG)
    if os.path.exists(log_path) and not args.allow_existing and not args.dry_run:
        die(
            f"{log_path} already exists: a second sweep would be appended to it and R recomputed "
            f"from the file would pool two Gs. Choose another --output-dir, or pass --allow-existing"
        )

    cfg = config_of(args, mode, work_dir)

    import numpy as np
    import probe
    import sa
    import stats
    from dropaug import io as dio
    from dropaug.config import host_of
    from omegaconf import OmegaConf
    from selection import Redundancy, measured_red
    from tasks.magnitudes import probe_magnitudes
    from vocab import get_op

    pcfg = to_python(cfg.probe)
    magnitude_mode = pcfg.get("magnitude_mode", sa.MODE_MODAL)
    if magnitude_mode not in sa.MAGNITUDE_MODES:
        die(f"probe.magnitude_mode {magnitude_mode!r} is not one of {sa.MAGNITUDE_MODES}")
    try:
        fallback = magnitudes_of(cfg.probe.magnitude)
        host = host_of(cfg)
        ops = {name: get_op(name) for name in host.vocabulary()}
        fallback.draw(ops, 1, np.random.default_rng(0))  # a bad magnitude fails now
    except (KeyError, ValueError) as error:
        die(f"probe.magnitude={pcfg.get('magnitude')!r}: {error}")
    fade, fade_at = fade_of(cfg)
    try:
        faded = probe.faded_ops(ops, fade)
        fade_at = probe.check_fade_at(fade_at)
    except ValueError as error:
        die(f"host.fade={to_python(cfg.host.get('fade'))!r}: {error}")

    # Only the sensaug host has an SA loop, so only it reads an SA snapshot.
    reads_snapshot = cfg.host.name == "sensaug" and magnitude_mode != sa.MODE_FIXED
    if args.magnitudes_path and not reads_snapshot:
        die(
            f"--magnitudes-path is the sensaug method's SA snapshot; method {cfg.host.name!r} with "
            f"magnitude mode {magnitude_mode!r} doesn't read one"
        )
    snapshot_file = snapshot_path(args, mode, work_dir, checkpoint) if reads_snapshot else None

    if args.dry_run:
        snapshot, _ = sa_snapshot(snapshot_file, None)
        magnitudes = probe_magnitudes(host, snapshot, magnitude_mode, fallback)
        print(dry_run_report(cfg, ops, fade, faded, fade_at, magnitudes, snapshot_file))
        return 0

    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(out_dir, CONFIG_FILE), resolve=True)
    strict = mode == MODE_CHECKPOINT and not args.non_strict
    backend, meta = make_backend(cfg, work_dir or out_dir, pcfg.get("split", "val"), checkpoint, strict=strict)
    ckpt_iter = meta.get("iter") if isinstance(meta, dict) else None
    data = backend.probe_data()
    probe_set, probe_set_how = pick_probe_set(work_dir, data, pcfg.get("n_probe"), int(pcfg.get("probe_seed", 0)))
    probe_set.save(os.path.join(out_dir, PROBE_SET_FILE))

    snapshot, snapshot_iter = sa_snapshot(snapshot_file, ckpt_iter)
    levels = policy_levels(host, magnitude_mode, checkpoint, backend, probe_set, data, pcfg)
    magnitudes = probe_magnitudes(host, snapshot, magnitude_mode, fallback, levels=levels)
    print(
        f"[probe] method {cfg.host.name}: {len(ops)} ops ({len(faded)} faded at a={fade_at:g}), probe set "
        f"{probe_set.digest[:12]} ({len(probe_set)} images of "
        f"{data.source}; {probe_set_how}), checkpoint iter {ckpt_iter}, magnitudes "
        f"{magnitudes.source}/{magnitudes.mode}"
        + (f" (SA snapshot of iter {snapshot_iter}, {snapshot_file})" if snapshot else "")
    )
    if reads_snapshot and not snapshot:
        print(
            f"[probe] NOTE: no SA snapshot ({snapshot_file or 'no ' + CORR_MAGNITUDES_FILE}); every op at "
            f"probe.magnitude={pcfg.get('magnitude')!r}",
            file=sys.stderr,
        )
    if os.path.exists(log_path):
        print(f"[probe] WARNING: appending to the existing {log_path} (--allow-existing)", file=sys.stderr)

    sweep = probe.sweep(
        backend,
        probe_set,
        data.dataset,
        ops,
        magnitudes,
        checkpoint=checkpoint,
        batch_size=int(pcfg.get("batch_size", 1)),
        probe_seed=int(pcfg.get("probe_seed", 0)),
        fd_step=pcfg.get("fd_step", probe.DEFAULT_STEP),
        num_workers=int(pcfg.get("num_workers", 0)),
        log_path=log_path,
        meta={
            "iter": ckpt_iter,
            "role": "active",
            "posthoc": True,
            "model_from": mode,
            "method": str(cfg.host.name),
            "backbone": str(cfg.task.backbone),
            "dataset": str(cfg.dataset.name),
            "magnitudes_snapshot_iter": snapshot_iter,
            "magnitudes_path": snapshot_file if snapshot else None,
        },
        fade=fade,
        fade_at=fade_at,
    )

    analysis_cfg = dict(pcfg.get("analysis") or {})
    mask_within_op = bool(analysis_cfg.pop("mask_within_op", True))
    analysis = stats.analyze_sweep(sweep, **analysis_cfg)
    print(f"[probe] {analysis.summary()}")

    stats.write_analysis(out_dir, analysis)
    if analysis.skipped is None:
        score = measured_red(Redundancy(analysis.names, analysis.primary, survives=analysis.survives, mask_within_op=mask_within_op))
        stats.append_jsonl(
            os.path.join(out_dir, stats.REDUNDANCY_LOG),
            stats.redundancy_record(
                score,
                analysis,
                mask_within_op,
                selector="none",
                lam=0.0,
                lam_target=0.0,
                applied=False,
                reason="post hoc probe: no training loop to reweight",
                q=None,
                pruned=[],
            ),
        )
    else:
        print(f"[probe] WARNING: R skipped ({analysis.skipped}); only G was written", file=sys.stderr)

    dio.write_G(os.path.join(out_dir, "G.npz"), sweep, ops)
    dio.write_R(os.path.join(out_dir, "R.json"), analysis, ops)
    if analysis.withheld:
        print(f"[probe] NOTE: R is withheld from any selector: {analysis.withheld}")
    for name in (GRADIENT_LOG, "corr_matrix_log.json", "corr_bootstrap_log.txt", "corr_redundancy_log.txt", "G.npz", "R.json", PROBE_SET_FILE, CONFIG_FILE):
        if os.path.exists(os.path.join(out_dir, name)):
            print(f"wrote: {os.path.join(out_dir, name)}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
