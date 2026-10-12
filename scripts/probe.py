#!/usr/bin/env python
"""The gradient probe, post hoc, on one checkpoint of a finished run: one frozen
sweep -> G -> R, with Sensaug's logs and dropaug's own G / R files.

dropaug's form of Sensaug's ``scripts/compute_grad_corr.py``. That script rebuilt a
Runner and drove the grad-corr hooks' private methods through a proxy; here the probe
is a library (``probe``, ``stats``), so this is glue: rebuild the run from its
``dropaug.yaml``, load the checkpoint, sweep, analyze, write. Nothing is computed here.

    python scripts/probe.py --work-dir experiments/<run> --checkpoint <run>/best_mIoU_iter_8000.pth \
        --output-dir experiments/<run>/probe/best_mIoU_iter_8000 [--magnitude 0.5] [--n-probe 500] \
        [--probe-seed 0] [--batch-size 1] [--split val] [probe.fd_step=0.1 probe.analysis.n_min=50 ...]

Flags are shorthand for ``probe.*`` overrides of the run's saved config (the
positional ``key=value`` words are further dotted overrides, applied after the flags;
groups cannot be switched). The keys are documented in ``configs/probe/default.yaml``.

Writes into ``--output-dir`` (not the run directory, where the in-training probe's
logs already are):

    aug_gradient_log.txt     one JSONL record per batch (APPENDED to: see --allow-existing)
    corr_matrix_log.json     R_raw / R_scalenorm / loadings ... (Sensaug's shape + dropaug's fields)
    corr_bootstrap_log.txt   per-cell CIs and FDR q-values
    corr_redundancy_log.txt  per-op red(a), scored but applied to nothing (no training loop)
    G.npz, R.json            dropaug.io files, with vocab / probe-set / checkpoint hashes
    probe_set.json           the images the columns of G are

Magnitudes. With ``probe.magnitude_mode`` other than ``fixed`` and a
``<work-dir>/corr_magnitudes.json`` (written by the SA loop of a sensaug-host run), each
op is probed at the SA levels published at or before the checkpoint's iteration, as the
live hook does (``sa.probe_magnitudes``); ops the snapshot lacks use
``probe.magnitude``. A randaugment / trivialaugment / autoaugment run is probed at its
host's own magnitude bins (``tasks.magnitudes``), and every other run at
``probe.magnitude``. The source and mode are recorded in every output.

Fading. The ops the run's host config names in its ``fade:`` block (``host.fade.ops``,
e.g. ``non_differentiable``) are faded in at that magnitude and differentiated along
the fade at ``host.fade.at`` (``probe.fade``); ``host.fade.ops=null`` turns it off.

Exit status 2 on a bad invocation: a missing work dir, ``dropaug.yaml`` or checkpoint,
an output dir that already holds a gradient log, a backend that is not wired.

Differences from compute_grad_corr.py: the run is rebuilt from ``dropaug.yaml``, not a
dumped mmengine .py; no Runner (``MMSegBackend`` builds the model and the val dataset);
the checkpoint is always explicit (slurm/probe.sbatch resolves best / last); the probe
set is the run's ``probe_set.json`` when it matches, else built from ``--n-probe`` and
``--probe-seed`` (Sensaug swept the whole val set); an existing gradient log is refused
unless ``--allow-existing``; the corr_magnitudes.json record is the latest one at or
before the checkpoint's iteration (Sensaug: the file's seed, whichever); no
``--ref-magnitude`` (``probe.magnitude`` is it).
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


def die(message: str):
    print(f"probe.py: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="The gradient probe on one checkpoint of a finished run: frozen sweep -> G -> R.",
        epilog="Further positional key=value words are dotted overrides of the run's config, e.g. probe.fd_step=0.1.",
    )
    p.add_argument("--work-dir", required=True, help="the run directory (holds dropaug.yaml)")
    p.add_argument("--checkpoint", required=True, help="checkpoint file to probe")
    p.add_argument("--output-dir", required=True, help="where the probe's logs and G / R files go")
    p.add_argument("--magnitude", type=float, help="probe.magnitude: a strength in [0, 1] for every op")
    p.add_argument("--magnitude-mode", help="probe.magnitude_mode: mode | sampled_shared | sampled_independent | fixed")
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
    p.add_argument("overrides", nargs="*", metavar="key=value", help="dotted overrides of the run's config")
    return p.parse_args(argv)


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


def make_backend(cfg, work_dir: str, split: str, checkpoint: str):
    """``(backend, ckpt_meta)``: the run's task backend with its model built and the
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
    meta = backend.load_ckpt(checkpoint)
    return backend, meta


def pick_probe_set(work_dir: str, data, n_probe, probe_seed):
    """The run's own ``probe_set.json`` when it is the set these settings would build
    (same source, size and seed), verified against the dataset, so this G is on the
    images the in-training Gs were; else a fresh one."""
    import probe

    path = os.path.join(work_dir, PROBE_SET_FILE)
    if os.path.exists(path):
        saved = probe.ProbeSet.load(path)
        want = len(data.ids) if n_probe is None else int(n_probe)
        same_seed = n_probe is None or saved.seed == probe_seed
        if saved.source == data.source and len(saved) == want and same_seed:
            saved.verify(data.ids)
            return saved, "the run's probe_set.json"
    return probe.ProbeSet.build(data.ids, n_probe, seed=probe_seed, source=data.source), "built"


def sa_snapshot(work_dir: str, ckpt_iter):
    """``({op: {levels, probs, mode}}, iter)`` from the latest corr_magnitudes.json
    record at or before ``ckpt_iter`` (the latest when it is unknown), else ``({}, None)``."""
    path = os.path.join(work_dir, CORR_MAGNITUDES_FILE)
    if not os.path.exists(path):
        return {}, None
    with open(path) as f:
        records = json.load(f)
    if ckpt_iter is not None:
        records = [r for r in records if int(r["iter"]) <= int(ckpt_iter)]
    if not records:
        return {}, None
    record = max(records, key=lambda r: int(r["iter"]))
    return record["magnitudes"], int(record["iter"])


def run(args) -> int:
    work_dir = os.path.abspath(args.work_dir)
    out_dir = os.path.abspath(args.output_dir)
    checkpoint = os.path.abspath(args.checkpoint)

    from dropaug.config import RUN_FILE, ConfigError, fade_of, load, magnitudes_of, to_python, vocab_of

    if not os.path.isdir(work_dir):
        die(f"--work-dir {work_dir} does not exist")
    if not os.path.exists(os.path.join(work_dir, RUN_FILE)):
        die(f"{os.path.join(work_dir, RUN_FILE)} does not exist; {work_dir} was not trained by train.py")
    if not os.path.isfile(checkpoint):
        die(f"--checkpoint {checkpoint} is not a file")
    log_path = os.path.join(out_dir, GRADIENT_LOG)
    if os.path.exists(log_path) and not args.allow_existing:
        die(
            f"{log_path} already exists: a second sweep would be appended to it and R recomputed "
            f"from the file would pool two Gs. Choose another --output-dir, or pass --allow-existing"
        )

    try:
        cfg = load(work_dir, overrides_of(args))
    except (ConfigError, ValueError) as error:
        die(str(error))

    import numpy as np
    import probe
    import sa
    import stats
    from dropaug import io as dio
    from dropaug.config import host_of
    from selection import Redundancy, measured_red
    from tasks.magnitudes import probe_magnitudes
    from vocab import get_op

    pcfg = to_python(cfg.probe)
    mode = pcfg.get("magnitude_mode", sa.MODE_MODAL)
    if mode not in sa.MAGNITUDE_MODES:
        die(f"probe.magnitude_mode {mode!r} is not one of {sa.MAGNITUDE_MODES}")
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

    os.makedirs(out_dir, exist_ok=True)
    backend, meta = make_backend(cfg, work_dir, pcfg.get("split", "val"), checkpoint)
    ckpt_iter = meta.get("iter") if isinstance(meta, dict) else None
    data = backend.probe_data()
    probe_set, probe_set_how = pick_probe_set(work_dir, data, pcfg.get("n_probe"), int(pcfg.get("probe_seed", 0)))
    probe_set.save(os.path.join(out_dir, PROBE_SET_FILE))

    snapshot, snapshot_iter = ({}, None) if mode == sa.MODE_FIXED else sa_snapshot(work_dir, ckpt_iter)
    magnitudes = probe_magnitudes(host, snapshot, mode, fallback)
    print(
        f"[probe] {len(ops)} ops ({len(faded)} faded at a={fade_at:g}), probe set "
        f"{probe_set.digest[:12]} ({len(probe_set)} images of "
        f"{data.source}; {probe_set_how}), checkpoint iter {ckpt_iter}, magnitudes "
        f"{magnitudes.source}/{magnitudes.mode}"
        + (f" (SA snapshot of iter {snapshot_iter})" if snapshot else "")
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
            "magnitudes_snapshot_iter": snapshot_iter,
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
    for name in (GRADIENT_LOG, "corr_matrix_log.json", "corr_bootstrap_log.txt", "corr_redundancy_log.txt", "G.npz", "R.json", PROBE_SET_FILE):
        if os.path.exists(os.path.join(out_dir, name)):
            print(f"wrote: {os.path.join(out_dir, name)}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
