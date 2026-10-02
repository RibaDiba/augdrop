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
    --uniform / --weighted-augs / --no-inv-aug / --hold-none-prob
                                  sa.uniform / sa.weighted_augs / sa.remove_H / sa.hold_none_prob
    --pruned-augmentations        host.pruned_ops=[op,...]   (sensaug and uniform hosts: out of
                                  training from iteration 0, the SA curve and pdfs, and the probe)
    --random-prune-count/-method/-seed
                                  selector=random_k selector.<key>=..., or host.pruned_ops
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


def resolve_work_dir(cfg, args) -> str:
    from dropaug.config import run_name

    if args.work_dir:
        path = args.work_dir
    elif cfg.get("work_dir"):
        path = cfg.work_dir
    else:
        path = os.path.join(cfg.cluster.work_root, run_name(cfg))
    return os.path.abspath(path)


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
    if rank0:
        for w in warnings:
            print(f"train.py: warning: {w}", file=sys.stderr)
        if not os.path.exists(run_file):
            save(cfg, work_dir)
        task_py = cfg.task.get("config")
        if task_py:
            src = os.path.join(CONFIG_DIR, "task", task_py)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(work_dir, os.path.basename(task_py)))
    return cfg, work_dir


def summary(cfg, work_dir) -> str:
    from dropaug.config import run_name

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
        ]
    )


def train_mmseg(cfg, work_dir, args):
    from tasks.mmseg.config import build  # lazy: needs mmengine

    mm_cfg = build(cfg, work_dir, launcher=cfg.launcher, resume=bool(args.resume or cfg.resume))
    if _rank() == 0:
        mm_cfg.dump(os.path.join(work_dir, "mmseg_config.py"))
    from mmengine.runner import Runner

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
