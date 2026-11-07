#!/usr/bin/env python
"""The NMS prune, post hoc, on one logged probe: sweep -> R_residual -> NMS -> the prune set P.

The computation is PruneHook's, from the same library functions
(``selection.nms_analysis`` / ``nms_select``), on a sweep read back from an
``aug_gradient_log.txt``. Nothing is computed here. It exists for the probe-and-prune
recipe: the trunk (the control run, selector ``none``) probes at a fixed magnitude, this
turns the decision probe into P, and the pruned branch is a fork of the trunk's
checkpoint at that iteration with ``PRUNE_OPS=P`` (train.py ``--fork-from``).

    # one of a run's in-training probes
    python scripts/nms_prune.py --work-dir experiments/<run> --iter 40000 [--tau 0.3 0.5 0.7]
    # a scripts/probe.py output directory (one sweep)
    python scripts/nms_prune.py --probe-dir experiments/<run>/probe/iter_40000
    # the stability ablation: P at several of a run's probes, and their pairwise Jaccard
    python scripts/nms_prune.py --work-dir experiments/<run> --iter 40000 88000 136000

Every probe is pruned fresh (M starts empty), so the prune sets of several probes are
independent: PruneHook's latch is a training-time rule. The host's prior (its
``base_probs()``, rebuilt from the run's config) is the pool NMS draws from, as in the
hook. The run's static prune is set aside, as the hook sets it aside. Ops with no
usable row (unprobed: SamplePairing, the flips) are exempt and kept.

Writes ``--output-dir`` (default ``<run>/nms/iter_<N>`` for one probe of a run, or the
probe dir): ``nms_prune.json`` (per tau: P, who suppressed whom at what r, the ranking,
the exempt ops, plus the source log's sha256 and the probe set, vocab and magnitude
setting) and ``R.json`` (``dropaug.io.write_R``: R_residual and its labels). With
several iterations: one subdirectory per iteration and ``jaccard.json``. Prints a
``PRUNE_OPS="..."`` line for each tau. Exit status 2 on a bad invocation.
"""

import argparse
import itertools
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

GRADIENT_LOG = "aug_gradient_log.txt"
PROBE_CONFIG = "probe_config.yaml"
RESULT_FILE = "nms_prune.json"
JACCARD_FILE = "jaccard.json"


def die(message: str):
    print(f"nms_prune.py: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NMS on one logged probe: the prune set P (PruneHook's computation, post hoc).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--work-dir", help="a dropaug run (its dropaug.yaml and aug_gradient_log.txt)")
    src.add_argument("--probe-dir", help="a scripts/probe.py output dir (its probe_config.yaml and aug_gradient_log.txt)")
    p.add_argument("--iter", type=int, nargs="+", help="(--work-dir) the probe iteration(s); several give the Jaccard table")
    p.add_argument("--tau", type=float, nargs="+", help="NMS cutoff(s) on R_residual (default selection.TAU, 0.5)")
    p.add_argument("--no-mask-within-op", action="store_true", help="let X_pos / X_neg suppress each other")
    p.add_argument("--output-dir", help="default <run>/nms/iter_<N> (one probe), <run>/nms (several), or the probe dir")
    return p.parse_args(argv)


def config_and_log(args):
    """``(cfg, log path, default output dir)``."""
    from omegaconf import OmegaConf

    from dropaug.config import RUN_FILE, load

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
        if not os.path.exists(os.path.join(work_dir, RUN_FILE)):
            die(f"{work_dir} holds no {RUN_FILE}; not a dropaug run")
        if not args.iter:
            die("--work-dir needs --iter: which of the run's probes")
        return load(work_dir), os.path.join(work_dir, GRADIENT_LOG), os.path.join(work_dir, "nms")
    if args.iter:
        die("--iter is for --work-dir; a probe dir holds one sweep")
    probe_dir = os.path.abspath(args.probe_dir)
    path = os.path.join(probe_dir, PROBE_CONFIG)
    if not os.path.exists(path):
        die(f"{probe_dir} holds no {PROBE_CONFIG}; not a scripts/probe.py output dir")
    return OmegaConf.load(path), os.path.join(probe_dir, GRADIENT_LOG), probe_dir


def pick_sweeps(sweeps, iters, log_path):
    """``{iter: sweep}``: the sweep of each asked-for iteration (the last one logged at it),
    or, with no iterations, the log's only sweep."""
    if iters is None:
        if len(sweeps) != 1:
            die(f"{log_path} holds {len(sweeps)} sweeps; a probe dir should hold one")
        return {sweeps[0].meta.get("iter"): sweeps[0]}
    out = {}
    for n in iters:
        at = [s for s in sweeps if s.meta.get("iter") == n]
        if not at:
            logged = sorted({s.meta.get("iter") for s in sweeps if s.meta.get("iter") is not None})
            die(f"{log_path} has no probe at iter {n}; it has {logged}")
        if len(at) > 1:
            print(f"[nms] WARNING: {len(at)} sweeps at iter {n} (a re-probe after a resume?); using the last", file=sys.stderr)
        out[n] = at[-1]
    return out


def prune_record(analysis, results, sweep, vocab, log_path) -> dict:
    from dropaug.io.hashing import file_hash, vocab_hash

    return {
        "iter": sweep.meta.get("iter"),
        "role": sweep.meta.get("role"),
        "checkpoint": sweep.checkpoint,
        "source": {"gradient_log": log_path, "sha256": file_hash(log_path)},
        "probe_set": sweep.probe_set,
        "vocab_hash": vocab_hash(vocab),
        "magnitude_source": sweep.magnitude_source,
        "magnitude_mode": sweep.magnitude_mode,
        "n_images": analysis.n_images,
        "matrix": "R_residual",
        "unprobed": list(analysis.unprobed),
        "static_pruned": list(analysis.static_pruned),
        "by_tau": {f"{tau:g}": {**result.record(), "applied": selection.applied} for tau, (selection, result) in results.items()},
    }


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    # Two empty prune sets agree.
    return 1.0 if not (a | b) else len(a & b) / len(a | b)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg, log_path, default_out = config_and_log(args)
    if not os.path.exists(log_path):
        die(f"{log_path} does not exist")

    import numpy as np

    import stats
    from dropaug import io as dio
    from dropaug.config import host_of, vocab_of
    from selection import TAU, nms_analysis, nms_select

    taus = args.tau or [TAU]
    vocab = vocab_of(cfg)
    host = host_of(cfg)
    rel = dict(zip(host.vocabulary(), host.base_probs()))
    sweeps = pick_sweeps(dio.sweeps_from_gradient_log(log_path), args.iter, log_path)
    several = len(sweeps) > 1
    out_root = os.path.abspath(args.output_dir or default_out)

    prune_sets = {}
    for n, sweep in sweeps.items():
        if sweep.magnitude_mode != "fixed":
            print(
                f"[nms] NOTE: the probe at iter {n} took magnitudes {sweep.magnitude_source}/{sweep.magnitude_mode}, "
                f"not a fixed reference (probe.magnitude_mode=fixed)",
                file=sys.stderr,
            )
        analysis = nms_analysis(sweep, static_pruned=host.pruned())
        if not np.isfinite(np.asarray(analysis.loss_increase, dtype=float)).any():
            die(f"the probe at iter {n} logged no losses, so NMS has nothing to rank by (a log written before the probe kept them)")
        results = {
            tau: nms_select(analysis, rel, tau=tau, mask_within_op=not args.no_mask_within_op) for tau in taus
        }
        # One directory per iteration, except a probe dir's one sweep, or one probe written where it was asked to go.
        flat = args.probe_dir or (args.output_dir and not several)
        out_dir = out_root if flat else os.path.join(out_root, f"iter_{n}")
        os.makedirs(out_dir, exist_ok=True)
        stats.write_json_atomic(os.path.join(out_dir, RESULT_FILE), prune_record(analysis, results, sweep, vocab, log_path))
        dio.write_R(os.path.join(out_dir, "R.json"), analysis, vocab)
        print(f"[nms] iter {n}: {analysis.summary()}")
        for tau, (_, result) in results.items():
            prune_sets[(n, tau)] = result.pruned
            suppressed = ", ".join(f"{op} (r={r:.2f} with {by})" for op, (by, r) in result.suppressed.items())
            print(f"[nms] iter {n}, tau={tau:g}: pruned {len(result.pruned)}: {suppressed or 'nothing'}")
            print(f'PRUNE_OPS="{" ".join(result.pruned)}"')
        print(f"wrote: {os.path.join(out_dir, RESULT_FILE)}")

    if several:
        table = {
            f"{tau:g}": [
                {"iters": [a, b], "jaccard": jaccard(prune_sets[(a, tau)], prune_sets[(b, tau)])}
                for a, b in itertools.combinations(sweeps, 2)
            ]
            for tau in taus
        }
        os.makedirs(out_root, exist_ok=True)
        stats.write_json_atomic(os.path.join(out_root, JACCARD_FILE), {"prune_sets": {f"{n}@{t:g}": list(p) for (n, t), p in prune_sets.items()}, "jaccard": table})
        for tau, rows in table.items():
            print(f"[nms] tau={tau}: " + ", ".join(f"J({a},{b})={row['jaccard']:.2f}" for row in rows for a, b in [row["iters"]]))
        print(f"wrote: {os.path.join(out_root, JACCARD_FILE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
