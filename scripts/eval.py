#!/usr/bin/env python
"""``dropaug.eval``'s robustness suite on one checkpoint of a finished run.

Replaces Sensaug's ``test_robust.py`` (and the per-op table of
``eval_all_augmentations.py``). Glue only: rebuild the run from its ``dropaug.yaml``,
load the checkpoint, and hand ``dropaug.eval.robustness.run`` the suite.

    python scripts/eval.py --work-dir experiments/<run> --checkpoint <run>/best_mIoU_iter_8000.pth \
        --out experiments/<run>/eval/best_mIoU_iter_8000.jsonl [--dataset acdc] [--overwrite] \
        [--no-corruptions] [eval.strengths=[0.5,1.0] eval.batch_size=2 ...]

The conditions are ``suite(vocab_of(cfg), **cfg.eval)``: clean, every vocab op at every
strength, the on-the-fly ImageNet-C corruptions, and Sensaug's combination levels. One
JSONL record per condition is appended to ``--out`` as it is measured, and a rerun
resumes (skips what is there). One file is one checkpoint. The positional ``key=value``
words are dotted overrides of the run's config (``eval.*`` documented in
configs/train.yaml); groups cannot be switched.

``--dataset NAME`` scores a domain-shift target instead of the run's val split: one of
the training dataset's ``shift_targets`` (ACDC, IDD), composed on its own and swapped in.
Its label space must match. Records carry that dataset's name, so one ``--out`` can hold
both.

Corruptions need ``imagecorruptions`` and the combination needs ``cv2``, both optional.
If one is missing the script exits 2 BEFORE evaluating anything, naming it. Pass
``--no-corruptions`` to drop the corruption and combination conditions explicitly
(the op conditions and clean still run). The corruptions are applied in the
DataLoader's workers (``eval.num_workers``; ``MMSegBackend.corrupted_dataset``), after
the pipeline's Resize, as test_robust.py's ImageNetCTransform was.

Differences from test_robust.py: the suite is ``dropaug.eval``'s (ops over the run's own
vocab at ``eval.strengths``, not SA's adaptive level search, which is not ported); the
domain-shift sets are the ``shift_targets`` of the dataset config (Dark Zurich and
Nighttime Driving are not ported); scores come from the segmentor's own ``inference``
(``MMSegBackend.predict``), at the label's resolution.

Exit status 2 on a bad invocation: a missing work dir, ``dropaug.yaml`` or checkpoint,
a dataset that is not a shift target, a missing optional dependency, a backend that is
not wired.
"""

import argparse
import logging
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

#: eval.* keys that configure the loader, not the suite.
LOADER_KEYS = ("batch_size", "num_workers")
SUITE_KEYS = ("strengths", "corruptions", "combination", "both_directions", "seed")


def die(message: str):
    print(f"eval.py: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="dropaug.eval's robustness suite on one checkpoint of a finished run.",
        epilog="Further positional key=value words are dotted overrides of the run's config, e.g. eval.strengths=[0.5,1.0].",
    )
    p.add_argument("--work-dir", required=True, help="the run directory (holds dropaug.yaml)")
    p.add_argument("--checkpoint", required=True, help="checkpoint file to score")
    p.add_argument("--out", required=True, help="results JSONL (one file per checkpoint; resumes)")
    p.add_argument("--dataset", help="score this shift target (one of the run dataset's shift_targets) instead of val")
    p.add_argument("--overwrite", action="store_true", help="re-measure conditions already in --out")
    p.add_argument(
        "--no-corruptions",
        action="store_true",
        help="drop the corruption and combination conditions (they need imagecorruptions / cv2)",
    )
    p.add_argument("overrides", nargs="*", metavar="key=value", help="dotted overrides of the run's config")
    return p.parse_args(argv)


def shift_target(cfg, name: str):
    """``cfg`` with ``dataset`` swapped for the shift target ``name``, composed on its
    own with the run's cluster. The run's train section rides along, since the mmseg
    builder reads every split of the dataset it is given (the post-hoc path never builds
    the train loader)."""
    from omegaconf import OmegaConf

    from dropaug.config import compose

    if name == cfg.dataset.name:
        return cfg
    targets = list(cfg.dataset.get("shift_targets") or [])
    if name not in targets:
        die(f"--dataset {name!r} is neither the run's dataset ({cfg.dataset.name!r}) nor one of its shift_targets {targets}")
    overrides = [f"dataset={name}"]
    cluster = (cfg.get("choices") or {}).get("cluster")
    if cluster:
        overrides.append(f"cluster={cluster}")
    target = compose(overrides).dataset
    if int(target.num_classes) != int(cfg.dataset.num_classes):
        die(f"{name} has {target.num_classes} classes, the run's dataset {cfg.dataset.num_classes}")
    if "train" not in target.mmseg:
        target.mmseg.train = cfg.dataset.mmseg.train
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    out.dataset = OmegaConf.to_container(target, resolve=True)
    return out


def make_backend(cfg, work_dir: str, checkpoint: str):
    """``(backend, ckpt_meta)``: the run's task backend on its val split, model built and
    checkpoint loaded. The one place that touches a training framework (so a test
    replaces just this)."""
    name = cfg.task.backend
    if name != "mmseg":
        die(f"task backend {name!r} is not wired into scripts/eval.py (only mmseg is)")
    from mmengine.utils import import_modules_from_strings

    from tasks.mmseg.backend import MMSegBackend
    from tasks.mmseg.config import build

    mm_cfg = build(cfg, work_dir)
    import_modules_from_strings(**mm_cfg.custom_imports)  # GpuAugSegDataPreProcessor etc.
    backend = MMSegBackend(mm_cfg, split="val")
    backend.build_model()
    meta = backend.load_ckpt(checkpoint)
    return backend, meta


def run(args) -> int:
    work_dir = os.path.abspath(args.work_dir)
    checkpoint = os.path.abspath(args.checkpoint)
    out = os.path.abspath(args.out)

    from dropaug.config import RUN_FILE, ConfigError, load, run_name, to_python, vocab_of

    if not os.path.isdir(work_dir):
        die(f"--work-dir {work_dir} does not exist")
    if not os.path.exists(os.path.join(work_dir, RUN_FILE)):
        die(f"{os.path.join(work_dir, RUN_FILE)} does not exist; {work_dir} was not trained by train.py")
    if not os.path.isfile(checkpoint):
        die(f"--checkpoint {checkpoint} is not a file")

    try:
        cfg = load(work_dir, list(args.overrides))
    except (ConfigError, ValueError) as error:
        die(str(error))
    dataset_name = args.dataset or cfg.dataset.name
    try:
        if args.dataset:
            cfg = shift_target(cfg, args.dataset)
    except (ConfigError, ValueError) as error:
        die(str(error))

    from dropaug.eval import robustness

    ecfg = to_python(cfg.eval)
    suite_kwargs = {k: ecfg[k] for k in SUITE_KEYS if k in ecfg}
    unknown = sorted(set(ecfg) - set(SUITE_KEYS) - set(LOADER_KEYS))
    if unknown:
        die(f"unknown eval keys {unknown}; expected {list(SUITE_KEYS + LOADER_KEYS)}")
    if args.no_corruptions:
        suite_kwargs.update(corruptions=[], combination=False)

    # Before anything is built or loaded: a missing imagecorruptions / cv2 fails here.
    try:
        conditions = robustness.suite(vocab_of(cfg), **suite_kwargs)
    except (ImportError, KeyError, ValueError) as error:
        die(f"{error}\n  (pass --no-corruptions to run only the clean and op conditions)")
    print(f"[eval] {len(conditions)} conditions on {dataset_name}" + (" (no corruptions)" if args.no_corruptions else ""))

    backend, meta = make_backend(cfg, work_dir, checkpoint)
    ckpt_iter = meta.get("iter") if isinstance(meta, dict) else None
    data = backend.probe_data()

    from torch.utils.data import DataLoader

    loader_kwargs = dict(
        batch_size=int(ecfg.get("batch_size", 1)),
        shuffle=False,
        collate_fn=getattr(backend, "collate", None),
        num_workers=int(ecfg.get("num_workers", 0)),
    )
    loader = DataLoader(data.dataset, **loader_kwargs)

    def corrupted_loader(corrupt_image):
        """The corruptions run in the loader's workers, as Sensaug's did."""
        return DataLoader(backend.corrupted_dataset(data.dataset, corrupt_image), **loader_kwargs)

    num_classes = int(cfg.dataset.num_classes)
    try:
        arm = run_name(cfg)
    except Exception:  # a hand-edited config without the keys run_name reads
        arm = None
    records = robustness.run(
        backend,
        loader,
        conditions,
        lambda: robustness.SegMetric(num_classes),
        out,
        dataset=dataset_name,
        checkpoint=checkpoint,
        meta={"run": arm, "iter": ckpt_iter, "work_dir": work_dir, "split": data.source},
        overwrite=args.overwrite,
        corrupted_loader=corrupted_loader if hasattr(backend, "corrupted_dataset") else None,
    )
    print(f"[eval] {len(records)} records in {out}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
