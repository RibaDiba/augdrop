#!/usr/bin/env python
"""Train one dropaug run: compose the config, check it, write the run directory, launch.

    python train.py --cluster nexus --backbone segformer --dataset cityscapes \\
        --host sensaug --selector mrmr --lam 0.25 --seed 0 --work-dir runs/x
    python train.py --backbone pspnet --dataset ade20k --host uniform task.max_iters=400
    python train.py +experiment=cityscapes_uniform_mrmr probe.magnitude=0.3
    python train.py <same flags> --dry-run            # compose + check + write dropaug.yaml, no training
    torchrun --nproc-per-node=4 train.py <same flags> --launcher pytorch

Every flag is shorthand for a Hydra override of configs/train.yaml, applied in this
order, and the positional ``overrides`` come after them, so they win:

    --cluster NAME          cluster=NAME
    --cluster-config PATH   cluster=<stem of PATH>   (PATH must be configs/cluster/<name>.yaml)
    --backbone B            task=<the task whose yaml says backbone: B>
    --task T                task=T
    --dataset D / --host H / --selector S     dataset=D / host=H / selector=S
    --lam F                 selector.lam=F          (selector none is the control: lam 0)
    --seed N                seed=N
    --launcher L            launcher=L              (none | pytorch)
    --resume                resume=true
    --descending-MA         sa.descending_MA=true   (Sensaug's flag; sensaug host only: the
                            most training mass goes on each op's most damaging level,
                            see src/sa/pdf.py)
    --no-warmup             sa.init_sa=true, and host.warmup_rounds=0 where the host has
                            one (Sensaug's flag: one val round before training; the host's
                            pre_train_round follows sa.init_sa)
    --no-inv-aug            sa.remove_H=true   (Sensaug's flag: lighter_H / darker_H stay
                            out of the SA curve and every pdf; still probed, and still in
                            the warmup policy, as in Sensaug)
    --pruned-augmentations OP [OP ...]
                            prune.ops=[OP,...]  (Sensaug's flag: ops out of training for
                            the whole run, on any host; see configs/train.yaml's prune:)
    --random-prune-count N  prune.random=N      (Sensaug's `null` arm: N more ops drawn at
                            random before training)
    --random-prune-seed S   prune.seed=S        (default: the run's seed)
    --random-prune-pool P   prune.pool=P        (all, Sensaug's; or probed: the draw also
                            leaves out the ops the probe gives no row)
    --no-hold-none-prob     prune.hold_none_prob=false  (Sensaug's default: P(none) rises
                            as ops are pruned)
    --probe-pruned          prune.probe=true    (the probe still measures the pruned ops)
    --fork-from CKPT        fork_from=<CKPT, absolute>  (start from another run's iter_<N>.pth;
                            see the fork rule below)
    --keep-checkpoints N [N ...]
                            keep_checkpoints=[N,...]  (those checkpoints outlive
                            max_keep_ckpts, hard-linked into <run>/keep/)

Both ``--flag value`` and ``--flag=value`` work.

The run directory is ``--work-dir`` (the run's own directory), else ``work_dir=`` from
the config, else ``<cluster.work_root>/<run_name>``. Rules:

- ``<run>/dropaug.yaml`` is the resolved config probe.py / eval.py rebuild from. If it
  exists and equals the new one (slurm runs ``--dry-run``, then the real launch, with the
  same arguments) that is fine. If it differs, train.py exits 2 and lists the keys,
  unless ``--resume``.
- ``--resume`` continues the run in place: the saved dropaug.yaml is used as it is (never
  rewritten, so the run keeps the values it started with; the flags that would have
  changed it are listed as a note) and mmseg resumes from ``last_checkpoint``. Without a
  saved file, the composed config is saved with ``resume: true``. It works with
  ``--dry-run``.
- Only rank 0 (env RANK absent or 0) writes: dropaug.yaml, a copy of the task's mmengine
  .py, and ``mmseg_config.py`` (the built config). Other ranks write nothing.
- With a static prune, rank 0 also writes ``<run>/prune.json`` (``hosts.Prune.record``:
  the named ops, the pool, the seed, the draw, what is kept). An existing one that
  disagrees with the config's prune exits 2, so a resume can't silently prune
  different ops.
- ``--fork-from <run>/[keep/]iter_<N>.pth`` starts a new run (a new directory, named
  ``..._fork-i<N>``) where another run's checkpoint left off: mmengine resumes from that
  file (iteration, optimizer, scheduler, message hub, the MADAug policy), so the two runs
  share their first N iterations. The source run's ``dropaug.yaml`` (next to the
  checkpoint, or one up from ``keep/``) must equal this config except in the prune, the
  selector, the probe schedule, ``keep_checkpoints``, the host's ``policy_file`` (the
  search-then-train continuation) and the run's own bookkeeping (``FORK_KEYS``); anything
  else exits 2 with the keys. The fork's own ``--resume``
  continues from its own ``last_checkpoint`` once it has one (and then checks neither the
  source checkpoint nor its run, which may be gone), else from ``fork_from``.
- ``--dry-run`` stops after that, printing a summary; it needs no mmengine. A cls task
  (no cls training loop yet) exits 2 after the dry-run stage. A config error (ConfigError,
  an unknown group option, a bad override) exits 2 with the message on stderr.

Sensaug's train.py flags that are not flags here, and what replaces each:

    --cluster-config (required)   --cluster NAME (configs/cluster/); --cluster-config PATH still works
    --rounds-config               host.n_rounds / warmup_rounds / corr_cadence (configs/host/sensaug.yaml)
    --backbone / --dataset        kept
    --aug-type                    host=<sensaug|uniform|randaugment|...>; augment=false trains clean
    --random-aug                  host=uniform
    --no-corr-sa                  host=uniform with a selector (no SA loop)
    --uniform / --weighted-augs   sa.uniform / sa.weighted_augs
    --hold-none-prob              the default (prune.hold_none_prob=true); --no-hold-none-prob
                                  is Sensaug's default
    --random-prune-method         prune.random > 0 is the `null` arm; selector=random_k is a
                                  different null, redrawn from R at every probe
    --geometric-only / --photometric-only
                                  a vocab config (vocab=<name>); not ported
    --corr-lambda / --corr-lambda-ramp
                                  --lam / selector.lambda_ramp
    --corr-downweight-method      --selector (mrmr, kl_reweight, ...)
    --corr-red-mode, --corr-keep-within-op, --corr-skip-pruned-eval
                                  selector.* / probe.* keys (see configs/selector, configs/probe)
    --corr-interval               host.probe_interval
    --corr-magnitudes / --corr-magnitude-mode
                                  probe.magnitude / probe.magnitude_mode
    --corr-sync-sa                not ported (the SA snapshot magnitude modes)
    --max-iters                   task.max_iters=N
    --amp / --auto-scale-lr / --adamw / --freeze-early-layers / --use-foundation-backbone
                                  task.amp, or task.cfg_options={dotted.mmengine.key: value}
    --local-rank                  torchrun sets env vars; not needed

What Sensaug's train() did that this one does not: ``set_manual_seed(0)`` (the builder sets
mmengine's ``randomness`` from ``seed``), the ``warn_ignored_*`` helpers (check() warns),
and ``set_train_spec`` (GpuAugSegDataPreProcessor owns its host and starts from the
host's own policy). It keeps touching ``runner.val_loop`` / ``test_loop`` before
``train()`` so a bad loop config fails at start-up.
"""

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

EXIT_CONFIG = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Train one dropaug run. Flags are shorthand for Hydra overrides of "
        "configs/train.yaml; any other argument is a Hydra override applied after them "
        "(task.max_iters=400, +experiment=NAME). See the module docstring.",
        allow_abbrev=False,
        usage="train.py [flags] [hydra overrides ...]",
    )
    p.add_argument("--cluster", help="configs/cluster/<NAME>.yaml (data and checkpoint paths)")
    p.add_argument("--cluster-config", metavar="PATH", help="path of a configs/cluster/<name>.yaml; same as --cluster <name>")
    p.add_argument("--backbone", help="picks the task whose yaml says backbone: NAME (segformer, pspnet, ...)")
    p.add_argument("--task", help="a configs/task option, e.g. mmseg_segformer_b0 (alternative to --backbone)")
    p.add_argument("--dataset", help="a configs/dataset option")
    p.add_argument("--host", help="a configs/host option (sensaug, uniform, ...)")
    p.add_argument("--selector", help="a configs/selector option; none is the lambda=0 control")
    p.add_argument("--lam", type=float, help="selector.lam, the selector's target strength")
    p.add_argument("--seed", type=int, help="the run's seed")
    p.add_argument("--work-dir", help="the run directory itself (default <cluster.work_root>/<run_name>)")
    p.add_argument("--launcher", choices=("none", "pytorch"), help="pytorch under torchrun")
    p.add_argument("--resume", action="store_true", help="continue the run in the work dir from last_checkpoint")
    p.add_argument(
        "--descending-MA",
        dest="descending_MA",
        action="store_true",
        help="sa.descending_MA=true (sensaug host): the most training mass goes on each op's most damaging level",
    )
    p.add_argument("--no-warmup", action="store_true", help="sa.init_sa=true (and host.warmup_rounds=0 where the host has one)")
    p.add_argument(
        "--no-inv-aug",
        dest="no_inv_aug",
        action="store_true",
        help="sa.remove_H=true (sensaug host): lighter_H / darker_H stay out of the SA curve and the pdf",
    )
    p.add_argument(
        "--pruned-augmentations",
        nargs="+",
        metavar="OP",
        help="prune.ops=[OP,...]: ops out of training for the whole run, on any host",
    )
    p.add_argument("--random-prune-count", type=int, metavar="N", help="prune.random=N: N more ops drawn at random before training")
    p.add_argument("--random-prune-seed", type=int, metavar="S", help="prune.seed=S: the random draw's seed (default the run's seed)")
    p.add_argument(
        "--random-prune-pool",
        choices=("all", "probed"),
        help="prune.pool: all (Sensaug's), or probed (the draw also leaves out the ops the probe gives no row)",
    )
    p.add_argument(
        "--no-hold-none-prob",
        dest="no_hold_none_prob",
        action="store_true",
        help="prune.hold_none_prob=false: P(no augmentation) rises as ops are pruned (Sensaug's default)",
    )
    p.add_argument("--probe-pruned", action="store_true", help="prune.probe=true: the probe still measures the pruned ops")
    p.add_argument("--fork-from", metavar="CKPT", help="fork_from: start a new run from another run's iter_<N>.pth")
    p.add_argument(
        "--keep-checkpoints",
        nargs="+",
        type=int,
        metavar="N",
        help="keep_checkpoints=[N,...]: these iterations' checkpoints outlive max_keep_ckpts (<run>/keep/)",
    )
    p.add_argument("--dry-run", action="store_true", help="compose, check and write dropaug.yaml, then exit")
    p.add_argument("overrides", nargs="*", help="Hydra overrides, applied after the flags")
    return p


def _fail(message: str) -> None:
    print(f"train.py: error: {message}", file=sys.stderr)
    sys.exit(EXIT_CONFIG)


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)


def flag_overrides(args, config_dir: str):
    """(flag overrides, user overrides). The flags' order is fixed; see the docstring."""
    from dropaug.config import task_for_backbone

    out = []
    if args.cluster and args.cluster_config:
        _fail("give --cluster or --cluster-config, not both")
    if args.cluster:
        out.append(f"cluster={args.cluster}")
    if args.cluster_config:
        path = os.path.realpath(args.cluster_config)
        name = os.path.basename(path)
        if os.path.dirname(path) != os.path.realpath(os.path.join(config_dir, "cluster")) or not name.endswith(".yaml"):
            _fail(
                f"--cluster-config {args.cluster_config}: must be one of configs/cluster/<name>.yaml "
                f"(the cluster is a config group); use --cluster <name>"
            )
        out.append(f"cluster={name[: -len('.yaml')]}")
    if args.backbone and args.task:
        _fail("give --backbone or --task, not both")
    if args.backbone:
        out.append(f"task={task_for_backbone(args.backbone)}")
    if args.task:
        out.append(f"task={args.task}")
    for flag in ("dataset", "host", "selector"):
        if getattr(args, flag):
            out.append(f"{flag}={getattr(args, flag)}")
    if args.lam is not None:
        out.append(f"selector.lam={args.lam:g}")
    if args.seed is not None:
        out.append(f"seed={args.seed}")
    if args.launcher:
        out.append(f"launcher={args.launcher}")
    if args.resume:
        out.append("resume=true")
    if args.descending_MA:
        out.append("sa.descending_MA=true")
    if args.no_warmup:
        out.append("sa.init_sa=true")
    if args.no_inv_aug:
        out.append("sa.remove_H=true")
    if args.pruned_augmentations:
        bad = [op for op in args.pruned_augmentations if not op or any(c in op for c in "'\\,[] ")]
        if bad:
            _fail(f"--pruned-augmentations: {bad} are not op names")
        out.append("prune.ops=[" + ",".join(f"'{op}'" for op in args.pruned_augmentations) + "]")
    if args.random_prune_count is not None:
        out.append(f"prune.random={args.random_prune_count}")
    if args.random_prune_seed is not None:
        out.append(f"prune.seed={args.random_prune_seed}")
    if args.random_prune_pool:
        out.append(f"prune.pool={args.random_prune_pool}")
    if args.no_hold_none_prob:
        out.append("prune.hold_none_prob=false")
    if args.probe_pruned:
        out.append("prune.probe=true")
    if args.fork_from:
        path = os.path.abspath(args.fork_from)
        if "'" in path or "\\" in path:
            _fail(f"--fork-from {path}: a path with a quote or backslash can't be passed to Hydra")
        out.append(f"fork_from='{path}'")
    if args.keep_checkpoints:
        out.append("keep_checkpoints=[" + ",".join(str(n) for n in args.keep_checkpoints) + "]")
    return out


def compose_config(args):
    from dropaug.config import CONFIG_DIR, compose

    flags = flag_overrides(args, CONFIG_DIR)
    user = list(args.overrides)
    cfg = compose(flags + user)
    if args.no_warmup and "warmup_rounds" in cfg.host:
        # The sensaug host has warmup_rounds; the others have no such key to set.
        cfg = compose(flags + ["host.warmup_rounds=0"] + user)
    return cfg


def _plain(cfg):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg, resolve=True)


def _flatten(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out or {prefix.rstrip("."): {}}
    return {prefix.rstrip("."): node}


def diff_keys(old, new, ignore=("resume",)):
    a, b = _flatten(_plain(old)), _flatten(_plain(new))
    return [
        (k, a.get(k, "<absent>"), b.get(k, "<absent>"))
        for k in sorted(set(a) | set(b))
        if a.get(k, "<absent>") != b.get(k, "<absent>") and k.split(".")[0] not in ignore
    ]


#: What a fork may change from the run it forks (``fork_problems``): its prune, its
#: selector, when it probes, which checkpoints it keeps, and the run's own bookkeeping.
#: Also ``host.policy_file``: the search-then-train recipe's continuation trains the
#: trunk's checkpoint (on Faster AutoAugment's starting policy) on the policy a search
#: then found. Key prefixes of the flattened, resolved config.
FORK_KEYS = (
    "prune.",
    "selector.",
    "choices.selector",
    "probe.schedule.",
    "host.probe_interval",
    "host.probe_iters",
    "host.control_iters",
    "host.policy_file",
    "keep_checkpoints",
    "fork_from",
    "work_dir",
    "resume",
)


def fork_source(path: str) -> str:
    """The run directory a fork's checkpoint belongs to: the checkpoint's directory, or
    the one above it for a kept checkpoint (``<run>/keep/iter_<N>.pth``)."""
    directory = os.path.dirname(os.path.abspath(path))
    return os.path.dirname(directory) if os.path.basename(directory) == "keep" else directory


def fork_problems(source, cfg) -> list:
    """The keys where a fork's config differs from its source run's beyond ``FORK_KEYS``:
    ``[(key, source value, fork value)]``, empty when the fork is a fork."""
    return [
        (key, old, new)
        for key, old, new in diff_keys(source, cfg, ignore=())
        if not any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in FORK_KEYS)
    ]


def check_fork(cfg, work_dir) -> None:
    """Exit 2 unless ``cfg.fork_from`` is a checkpoint of another dropaug run whose config
    this one changes only in ``FORK_KEYS``."""
    from dropaug.config import RUN_FILE, load

    path = str(cfg.fork_from)
    if not os.path.isfile(path):
        _fail(f"--fork-from {path} is not a file")
    source_dir = fork_source(path)
    if os.path.abspath(source_dir) == os.path.abspath(work_dir):
        _fail(f"--fork-from {path} is in this run's own directory; a fork goes in a new one")
    if not os.path.exists(os.path.join(source_dir, RUN_FILE)):
        _fail(f"--fork-from {path}: {source_dir} holds no {RUN_FILE}, so it isn't a dropaug run to fork")
    problems = fork_problems(load(source_dir), cfg)
    if problems:
        lines = "\n".join(f"    {k}: source {o!r}, fork {n!r}" for k, o, n in problems)
        _fail(
            f"a fork may change only the prune, the selector, the probe schedule, keep_checkpoints and "
            f"host.policy_file; "
            f"this config also differs from {os.path.join(source_dir, RUN_FILE)} in:\n{lines}"
        )


def resolve_work_dir(cfg, args) -> str:
    from dropaug.config import run_name

    if args.work_dir:
        path = args.work_dir
    elif cfg.get("work_dir"):
        path = cfg.work_dir
    else:
        path = os.path.join(cfg.cluster.work_root, run_name(cfg))
    return os.path.abspath(path)


def prune_record(cfg):
    """``prune.json``'s contents for the run's static prune, or None with nothing pruned."""
    from dropaug.config import host_of, prune_of, vocab_of
    from dropaug.io.hashing import vocab_hash

    pruning = prune_of(cfg)
    if not pruning:
        return None
    vocab = vocab_of(cfg)
    host = host_of(cfg, pruning)
    # `kept` follows what the host realizes (autoaugment prunes ops by implication), `pruned` the config.
    realized = host.pruned()
    # A host whose prune acts per stage says what each stage lost (faster_autoaugment).
    extra = {"stages": host.prune_report()} if hasattr(host, "prune_report") else {}
    return pruning.record(list(vocab), realized, host=str(cfg.host.name), vocab_hash=vocab_hash(vocab), **extra)


def write_prune_record(cfg, work_dir) -> None:
    """Write ``<run>/prune.json``, or refuse (exit 2) when one is there and records a
    different prune. ``vocab_hash`` is provenance only: an op's definition can change
    without changing the draw."""
    import json

    from hosts import PRUNE_FILE

    path = os.path.join(work_dir, PRUNE_FILE)
    record = prune_record(cfg)
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        same = lambda r: {k: v for k, v in (r or {}).items() if k != "vocab_hash"}  # noqa: E731
        if same(saved) != same(record):
            _fail(
                f"{path} records the prune {saved.get('pruned')}, but this config prunes "
                f"{None if record is None else record['pruned']}. Use a new --work-dir."
            )
        return
    if record is not None:
        os.makedirs(work_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, path)


def prepare(args):
    """Compose, check and (rank 0) write the run directory. Returns (cfg, work_dir)."""
    from omegaconf import OmegaConf

    from dropaug.config import RUN_FILE, CONFIG_DIR, check, load, save

    cfg = compose_config(args)
    work_dir = resolve_work_dir(cfg, args)
    cfg.work_dir = work_dir
    run_file = os.path.join(work_dir, RUN_FILE)
    rank0 = _rank() == 0

    if args.resume and os.path.exists(run_file):
        # The run keeps the values it started with.
        saved = load(work_dir)
        changed = diff_keys(saved, cfg)
        if changed and rank0:
            print(f"train.py: note: resuming with the saved {RUN_FILE}; these flags/overrides differ and are ignored:")
            for key, old, new in changed:
                print(f"    {key}: saved {old!r}, now {new!r}")
        cfg = saved
        cfg.work_dir = work_dir
    elif rank0 and os.path.exists(run_file):
        changed = diff_keys(load(work_dir), cfg)
        if changed:
            lines = "\n".join(f"    {k}: saved {o!r}, now {n!r}" for k, o, n in changed)
            _fail(
                f"{run_file} exists and differs from this config:\n{lines}\n"
                f"Use a new --work-dir, or --resume to continue the saved run as it is."
            )

    warnings = check(cfg, training=True)
    # A fork resuming from a checkpoint of its own no longer reads fork_from
    # (tasks.mmseg.config.build), and its config was checked against the source when it
    # started: the source may be gone by now.
    own_checkpoint = args.resume and os.path.exists(os.path.join(work_dir, "last_checkpoint"))
    if cfg.get("fork_from") and not own_checkpoint:
        check_fork(cfg, work_dir)
    if rank0:
        for w in warnings:
            print(f"train.py: warning: {w}", file=sys.stderr)
        if not os.path.exists(run_file):
            save(cfg, work_dir)
        write_prune_record(cfg, work_dir)
        task_py = cfg.task.get("config")
        if task_py:
            src = os.path.join(CONFIG_DIR, "task", task_py)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(work_dir, os.path.basename(task_py)))
    return cfg, work_dir


def summary(cfg, work_dir) -> str:
    from dropaug.config import fork_iter, prune_of, run_name

    pruning = prune_of(cfg)
    lines = [] if not pruning else [
        f"prune     {list(pruning.ops)}"
        + (f" (named {list(pruning.named)})" if pruning.named and pruning.drawn else "")
        + (f" (drew {pruning.random} of {len(pruning.pool)} at seed {pruning.seed})" if pruning.random else "")
        + f", P(none) {'held' if pruning.hold_none_prob else 'renormalized'}"
        + (", still probed" if pruning.probe else ", not probed")
    ]
    if pruning:
        from dropaug.config import host_of

        host = host_of(cfg, pruning)
        if hasattr(host, "prune_report"):
            report = host.prune_report()
            share = report["stage_share_pruned"]
            lines.append(
                f"stages    the prune held {max(share):.0%} of a stage's mass at most, {sum(share) / len(share):.0%} "
                f"on average; {len(report['flagged'])} stage(s) lost over half"
                + (" (their mass held: it went to their other ops)" if report["hold_stage_mass"] else "")
            )
    return "\n".join(
        [
            f"run dir   {work_dir}",
            f"run_name  {run_name(cfg)}",
            f"task      {cfg.choices.get('task')} (backbone {cfg.task.backbone}, backend {cfg.task.backend})",
            f"dataset   {cfg.dataset.name}",
            f"host      {cfg.host.name}",
            f"selector  {cfg.choices.get('selector', cfg.selector.name)} (lam {float(cfg.selector.lam):g}"
            + (", control)" if cfg.selector.get("control") else ")"),
            f"seed      {cfg.seed}  launcher {cfg.launcher}  resume {bool(cfg.resume)}  augment {bool(cfg.augment)}",
            *lines,
            *([f"fork      from {cfg.fork_from} (iteration {fork_iter(cfg.fork_from)})"] if cfg.get("fork_from") else []),
            *([f"keep      checkpoints at {list(cfg.keep_checkpoints)}"] if cfg.get("keep_checkpoints") else []),
        ]
    )


def train_mmseg(cfg, work_dir, args):
    from tasks.mmseg.config import build  # lazy: needs mmengine

    mm_cfg = build(cfg, work_dir, launcher=cfg.launcher, resume=bool(args.resume or cfg.resume))
    if _rank() == 0:
        mm_cfg.dump(os.path.join(work_dir, "mmseg_config.py"))
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings

    # custom_imports only runs when mmengine parses a config file; this one is built in
    # memory, so register GpuAugSegDataPreProcessor, the loops and hooks by hand.
    import_modules_from_strings(**mm_cfg.custom_imports)
    runner = Runner.from_cfg(mm_cfg)
    runner.val_loop  # Sensaug's: build the loops now, so a bad loop config fails at start-up
    runner.test_loop
    runner.train()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stray = [o for o in args.overrides if o.startswith("--")]
    if stray:
        _fail(f"unknown flag(s) {stray}; see --help (Sensaug's other flags are Hydra overrides, see the docstring)")

    from hydra.errors import HydraException

    from dropaug.config import ConfigError

    try:
        cfg, work_dir = prepare(args)
    except (ConfigError, HydraException) as error:
        _fail(str(error))

    if _rank() == 0:
        print(summary(cfg, work_dir))
    if args.dry_run:
        print("dry run: config composed and checked, nothing trained")
        return 0
    backend = cfg.task.backend
    if backend == "mmseg":
        try:
            train_mmseg(cfg, work_dir, args)
        except ConfigError as error:
            _fail(str(error))
        return 0
    _fail(f"task backend {backend!r} has no training loop yet (only mmseg trains; cls is probe/eval only)")


if __name__ == "__main__":
    sys.exit(main())
