"""Op-matched robustness: score one trained model under every perturbation condition.

Replaces three Sensaug scripts. test_robust.py evaluated clean, every op at a level
grid, 7 ImageNet-C corruptions and "combination", resumably, then ran
eval_results_to_csv. scripts/eval_all_perturbations.py evaluated every diff32 op at a
fixed grid whether or not training pruned it, and wrote results.jsonl + summary.csv.
scripts/eval_all_augmentations.py contributed its per-op table. All three drove one
mmengine Runner per (op, level), through a val-pipeline transform or the GPU data
preprocessor. Here an evaluation is a plain function of (adapter, loader, condition)
and knows no training framework, like the probe's sweep.

"Op-matched" means the test perturbations are the vocab's own OpSpecs: the same
implementation and units the training augmentation drew from. So a pruned run is
judged on exactly the ops it dropped, which is the comparison eval_all_perturbations.py
existed for, and every op of the vocab is evaluated, pruned or not. Magnitudes don't
transfer between implementations of the "same" op, so ``randaug_default.Rotate`` and
``rotate_pos`` are two different conditions, not one condition at two scales.

THE ADAPTER extends the probe's, so one object can serve both:

    adapter.model                     The bare nn.Module. Put in eval() for every pass.
    adapter.to_unit(batch)            -> (images01, label), exactly as for the probe.
    adapter.predict(images01, batch)  -> scores: (B, K) for classification, or
                                         (B, K, H, W) at the LABEL's resolution for
                                         segmentation. Normalizes internally.
    adapter.target(label, batch)      -> what the scores are graded against. [optional]
                                         Defaults to the label as the perturbation
                                         returned it (segmentation). A classification
                                         adapter returns the batch's (B,) class ids.

A CONDITION is one row of the eval: clean, one op at one level, or one corruption at
one severity. ``suite`` builds test_robust.py's set over a dropaug vocab:

    conditions = suite(build_vocab())    # clean, 32 ops x 5 strengths,
                                         # 7 ImageNet-C x 5 severities, combination x 6
    records = run(adapter, loader, conditions, lambda: SegMetric(19),
                  "results.jsonl", checkpoint="best_mIoU.pth")
    write_csv("summary.csv", summarize(records))

``run`` appends one JSON record per condition to ``results_path`` and skips any
condition already there, so a walltime-killed job resumes where it stopped. One
results file holds one checkpoint. Mixing weights in one table is the eval's version
of pooling sweeps.

Invariants carried over (each was a Sensaug bug or guard):
- eval() for every pass, and every module's own mode restored afterwards.
- A geometric op moves the label with the image. The vocab op does the warp, and
  ``apply_op`` refuses an op that didn't. Scores are graded against the moved label,
  with out-of-frame pixels ignored.
- A perturbation that leaves the first batch bit-identical is logged and flagged in its
  record (Sensaug's verify_perturbation_effective). A silently clean eval reads as
  perfect robustness.
- Randomness is pinned per (seed, condition, image position) and never consumes the
  caller's torch or numpy RNG streams.
- Every record carries its level, the op-unit magnitude and the implementation.

Not ported: Sensaug's adaptive SA level search and its betabinom-ranked pdf
(eval_all_augmentations.py, i.e. the SA loop); the mmseg glue (checkpoint discovery,
config recovery, the label-before-Resize pipeline), which belongs in an mmseg adapter;
DDP; and the LaTeX table formatter. The domain-shift sets test.py switched to (ACDC,
IDD, Dark Zurich, Nighttime Driving) need no code of their own here. Each is another
loader, run under its own ``dataset`` name.
"""

import csv
import functools
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from probe.frozen_sweep import frozen, strength_to_magnitude
from probe.grad_collect import apply_op, op_seed, preserved_rng, require_eval, unit_batch

from .corruptions import (
    ALEXNET_ERR,
    COMBINATION,
    SENSAUG_UNSEEN,
    SEVERITIES,
    CorruptFn,
    ImageCorruption,
    category,
    check_backend,
    corruption_perturbation,
    corruption_source,
    default_severities,
    mce,
)

logger = logging.getLogger(__name__)

Tensor = torch.Tensor
# (clean images01, label, images seen before this batch) -> (images01, label)
Perturb = Callable[[Tensor, Optional[Tensor], int], tuple]
Levels = Union[Sequence[float], Mapping[str, Sequence[float]]]

CLEAN = "clean"
# "corruption" is corrupted on the fly from the run's loader; "prerendered" brings its
# own loader. Their numbers differ even for the same corruption, so they never share a key.
KINDS = ("clean", "op", "corruption", "prerendered")
LEVEL_KINDS = ("none", "strength", "magnitude", "severity")

#: sensaug/testing/shared_levels_diff32.json's grid.
DEFAULT_STRENGTHS = (0.2, 0.4, 0.6, 0.8, 1.0)

#: vocab.diff_ops.LABEL_IGNORE_INDEX: what geometric ops fill out-of-frame label pixels
#: with, and Cityscapes' ignore id. Not imported, so the eval runs without kornia.
IGNORE_INDEX = 255


# --- conditions -----------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One row of the eval: what is done to each batch, and the labels its record carries.

    ``perturb(images01, label, index)`` receives the clean unit batch, where ``index`` is
    how many images the pass saw before this batch (for per-image seeding). ``loader``,
    when set, replaces the run's loader for this condition (a pre-rendered corruption
    set). It is only called if the condition actually runs. ``corrupt_image`` is set on
    an on-the-fly corruption: the per-image function ``perturb`` applies, which ``run``
    hands to its ``corrupted_loader`` instead when it has one.
    """

    name: str
    kind: str
    level_kind: str = "none"
    level: Optional[float] = None
    perturb: Optional[Perturb] = field(default=None, repr=False, compare=False)
    loader: Optional[Callable[[], Iterable]] = field(default=None, repr=False, compare=False)
    info: Mapping[str, Any] = field(default_factory=dict, compare=False)
    corrupt_image: Optional[ImageCorruption] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind {self.kind!r} not in {KINDS}")
        if self.level_kind not in LEVEL_KINDS:
            raise ValueError(f"{self.name}: level_kind {self.level_kind!r} not in {LEVEL_KINDS}")
        if (self.level is None) != (self.level_kind == "none"):
            raise ValueError(f"{self.name}: level {self.level!r} does not fit level_kind {self.level_kind!r}")

    @property
    def key(self) -> tuple:
        return (self.kind, self.name, self.level_kind, self.level)


def clean() -> Condition:
    return Condition(CLEAN, "clean")


def op_conditions(
    ops: Mapping[str, Any],
    levels: Levels = DEFAULT_STRENGTHS,
    *,
    level_kind: str = "strength",
    both_directions: bool = False,
    seed: int = 0,
) -> list[Condition]:
    """One condition per (op, level), in vocab order.

    ``levels`` is shared by every op, or is a mapping ``{op: levels}``, for instance a
    per-op grid loaded from JSON like Sensaug's test_levels files. With
    ``level_kind="strength"`` a level is a fraction of the way from the op's identity to
    the far end of its range (``probe.strength_to_magnitude``). With ``"magnitude"`` it
    is in the op's own units. ``both_directions`` adds the mirrored strength -s for ops
    whose identity lies inside the range (randaug_default.Rotate: +-30 degrees), since a
    positive strength only ever reaches one side. An op with no magnitude gets a single
    condition.
    """
    if level_kind not in ("strength", "magnitude"):
        raise ValueError(f"op level_kind must be 'strength' or 'magnitude', got {level_kind!r}")
    if isinstance(levels, Mapping):
        unknown = sorted(set(levels) - set(ops))
        if unknown:
            raise ValueError(f"levels name ops not in the vocab: {unknown}")

    conditions = []
    for name, op in ops.items():
        info = {
            "family": getattr(op, "family", None),
            "op_kind": getattr(op, "kind", None),
            "source": getattr(op, "source", None),
            "seed": seed,
        }
        if getattr(op, "mag_range", None) is None:
            conditions.append(
                Condition(name, "op", perturb=_op_perturb(name, op, None, seed, None), info={**info, "magnitude": None})
            )
            continue
        if isinstance(levels, Mapping):
            if name not in levels:
                raise ValueError(f"no levels for op {name!r}")
            op_levels = [float(level) for level in levels[name]]
        else:
            op_levels = [float(level) for level in levels]
        if both_directions and level_kind == "strength" and _near_end(op) is not None:
            op_levels += [-level for level in op_levels if level > 0]
        for level in op_levels:
            magnitude = _magnitude(name, op, level, level_kind)
            conditions.append(
                Condition(
                    name,
                    "op",
                    level_kind,
                    level,
                    perturb=_op_perturb(name, op, magnitude, seed, level),
                    info={**info, "magnitude": magnitude},
                )
            )
    return conditions


def corruption_conditions(
    names: Sequence[str] = SENSAUG_UNSEEN,
    severities: Optional[Sequence[int]] = None,
    *,
    seed: int = 0,
    fn: Optional[CorruptFn] = None,
) -> list[Condition]:
    """One condition per (corruption, severity), corrupted on the fly from the run's own
    loader: in the loader's workers when ``run`` has a ``corrupted_loader``, else in this
    process (``corruptions.corruption_perturbation``). ``severities`` None means 1-5, or
    1-6 for combination. ``fn`` replaces ``corruptions.corrupt``. Without it, each
    corruption's backend is imported here, so a missing dependency fails before the
    first eval."""
    conditions = []
    for name in names:
        if fn is None:
            check_backend(name)
        for severity in default_severities(name) if severities is None else severities:
            conditions.append(
                Condition(
                    name,
                    "corruption",
                    "severity",
                    int(severity),
                    perturb=corruption_perturbation(name, int(severity), seed=seed, fn=fn),
                    corrupt_image=ImageCorruption(name, int(severity), seed=seed, fn=fn),
                    info={
                        "category": category(name),
                        # A name, never repr(fn): a repr can hold an address, and the
                        # source must match on resume.
                        "source": corruption_source(name) if fn is None else getattr(fn, "__name__", type(fn).__name__),
                        "seed": seed,
                    },
                )
            )
    return conditions


def prerendered_conditions(
    make_loader: Callable[[str, int], Iterable],
    names: Sequence[str],
    severities: Sequence[int] = SEVERITIES,
    *,
    source: str,
) -> list[Condition]:
    """One condition per (corruption, severity) of a released benchmark.
    ``make_loader(name, severity)`` returns its loader, e.g. a DataLoader over
    ``corruptions.ImageNetC(root, name, severity)``, and is only called when the
    condition runs. ``source`` names the release (``"ImageNet-C"``)."""
    return [
        Condition(
            name,
            "prerendered",
            "severity",
            int(severity),
            loader=functools.partial(make_loader, name, int(severity)),
            info={"category": category(name), "source": source},
        )
        for name in names
        for severity in severities
    ]


def suite(
    ops: Mapping[str, Any],
    *,
    strengths: Sequence[float] = DEFAULT_STRENGTHS,
    corruptions: Sequence[str] = SENSAUG_UNSEEN,
    combination: bool = True,
    both_directions: bool = False,
    seed: int = 0,
) -> list[Condition]:
    """test_robust.py's suite over a dropaug vocab: clean; every op at every strength;
    the ImageNet-C corruptions test_robust used, at severities 1-5 on the fly; and
    Sensaug's six combination settings. Pass ``corruptions=corruptions.IMAGENET_C`` for
    all 15."""
    names = list(corruptions) + ([COMBINATION] if combination else [])
    return [
        clean(),
        *op_conditions(ops, strengths, both_directions=both_directions, seed=seed),
        *corruption_conditions(names, seed=seed),
    ]


def _op_perturb(name, op, magnitude, seed, level) -> Perturb:
    def perturb(images01, label, index):
        mag = None
        if magnitude is not None:
            mag = torch.full((images01.shape[0],), magnitude, dtype=images01.dtype, device=images01.device)
        return apply_op(name, op, images01, label, mag, op_seed(seed, f"{name}|{level}|{index}"))

    return perturb


def _near_end(op) -> Optional[float]:
    """The end of ``mag_range`` that ``strength_to_magnitude`` does NOT reach, or None
    when the identity sits on it (Solarize, Posterize, the unsigned [0, 1] ops)."""
    lo, hi = op.mag_range
    far = strength_to_magnitude(op, 1.0)
    near = lo if far == hi else hi
    return None if near == op.identity_at else near


def _magnitude(name, op, level, level_kind) -> float:
    lo, hi = op.mag_range
    if level_kind == "magnitude":
        if not lo <= level <= hi:
            raise ValueError(f"magnitude {level} is outside op {name!r}'s mag_range [{lo}, {hi}]")
        return float(level)
    if level >= 0:
        return float(strength_to_magnitude(op, level))
    near = _near_end(op)
    if near is None or level < -1.0:
        raise ValueError(f"op {name!r} has no negative direction for strength {level}")
    return float(op.identity_at + (-level) * (near - op.identity_at))


# --- metrics --------------------------------------------------------------------------


class SegMetric:
    """mIoU / aAcc / mAcc in percent, accumulated over the whole pass, as mmseg's
    IoUMetric computes them. A class that appears in neither labels nor predictions has
    IoU 0/0 and is left out of the mean (NaN), not counted as 0.

    Like IoUMetric it keeps three per-class pixel counts (intersection, label, prediction),
    not a K x K confusion matrix. Counting into K * K bins is the same arithmetic, but
    past 1000 bins torch's CUDA bincount drops to one global atomic per pixel, and most
    pixels land in a few bins. On ADE20K (K = 150, 22,500 bins) the K x K count took
    738 ms per image on an A5000, against 11 ms for the forward pass; these counts take
    3.7 ms, with the same mIoU to the last bit (checked 2026-10-15)."""

    primary = "mIoU"

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX):
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.counts = None  # (3, K): intersection, label, prediction

    def update(self, scores: Tensor, target: Tensor) -> None:
        k = self.num_classes
        if target.dim() == 4:
            target = target.squeeze(1)
        if scores.dim() != 4 or scores.shape[1] != k:
            raise ValueError(f"segmentation scores must be (B, {k}, H, W), got {tuple(scores.shape)}")
        if scores.shape[0] != target.shape[0] or scores.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"scores {tuple(scores.shape)} and target {tuple(target.shape)} disagree. "
                f"predict must return scores at the label's resolution (mmseg resizes its "
                f"prediction to the label's shape)"
            )
        target = target.to(scores.device).long()
        valid = target != self.ignore_index
        stray = valid & ((target < 0) | (target >= k))
        if bool(stray.any()):
            raise ValueError(
                f"the target holds class ids outside [0, {k}) that are not the ignore index "
                f"{self.ignore_index}: {sorted(target[stray].unique().tolist())[:5]}"
            )
        pred = scores.argmax(dim=1)[valid]
        target = target[valid]
        counts = torch.stack(
            [
                torch.bincount(target[pred == target], minlength=k),
                torch.bincount(target, minlength=k),
                torch.bincount(pred, minlength=k),
            ]
        )
        self.counts = counts if self.counts is None else self.counts + counts

    def _per_class(self):
        if self.counts is None:
            raise ValueError("no batches were scored")
        intersect, area_label, area_pred = self.counts.double().cpu().numpy()
        union = area_label + area_pred - intersect
        with np.errstate(invalid="ignore", divide="ignore"):
            return intersect, area_label, intersect / union, intersect / area_label

    def compute(self) -> dict:
        intersect, area_label, iou, acc = self._per_class()
        with np.errstate(invalid="ignore", divide="ignore"):
            aacc = intersect.sum() / area_label.sum()
        return {"mIoU": 100.0 * _nanmean(iou), "aAcc": 100.0 * float(aacc), "mAcc": 100.0 * _nanmean(acc)}

    def per_class(self) -> dict:
        _, _, iou, acc = self._per_class()
        return {"IoU": [100.0 * float(x) for x in iou], "Acc": [100.0 * float(x) for x in acc]}


class ClsMetric:
    """Top-k accuracy in percent."""

    primary = "top1"

    def __init__(self, topk: Sequence[int] = (1, 5)):
        self.topk = tuple(int(k) for k in topk)
        self.correct = {k: 0 for k in self.topk}
        self.total = 0

    def update(self, scores: Tensor, target: Tensor) -> None:
        if scores.dim() != 2:
            raise ValueError(f"classification scores must be (B, K), got {tuple(scores.shape)}")
        target = torch.as_tensor(target, device=scores.device).long().reshape(-1)
        if target.shape[0] != scores.shape[0]:
            raise ValueError(f"{scores.shape[0]} score rows for {target.shape[0]} targets")
        k_max = min(max(self.topk), scores.shape[1])
        hits = scores.topk(k_max, dim=1).indices == target[:, None]
        for k in self.topk:
            self.correct[k] += int(hits[:, : min(k, k_max)].any(dim=1).sum())
        self.total += int(target.shape[0])

    def compute(self) -> dict:
        if not self.total:
            raise ValueError("no batches were scored")
        return {f"top{k}": 100.0 * self.correct[k] / self.total for k in self.topk}


# --- evaluation -----------------------------------------------------------------------


def evaluate(adapter, loader: Iterable, metric, perturb: Optional[Perturb] = None) -> dict:
    """One pass of ``adapter``'s frozen model over ``loader``, every batch through
    ``perturb``. Returns ``{"metrics", "per_class", "n_images", "unchanged"}``, where
    ``unchanged`` is whether ``perturb`` left the first batch bit-identical (None when
    there is no perturb)."""
    n_images, unchanged = 0, None
    with torch.no_grad(), preserved_rng(), frozen(adapter.model):
        require_eval(adapter.model)
        for batch in loader:
            images01, label = unit_batch(adapter, batch)
            if perturb is not None:
                perturbed, label = perturb(images01, label, n_images)
                if unchanged is None:
                    unchanged = perturbed.shape == images01.shape and bool(torch.equal(perturbed, images01))
                images01 = perturbed
            metric.update(adapter.predict(images01, batch), _target(adapter, label, batch))
            n_images += int(images01.shape[0])
    if n_images == 0:
        raise ValueError("the loader yielded no images")
    per_class = metric.per_class() if hasattr(metric, "per_class") else None
    return {"metrics": metric.compute(), "per_class": per_class, "n_images": n_images, "unchanged": unchanged}


def run(
    adapter,
    loader: Iterable,
    conditions: Iterable[Condition],
    metric: Callable[[], Any],
    results_path: str,
    *,
    dataset: str = "val",
    checkpoint: Any = None,
    meta: Optional[Mapping] = None,
    overwrite: bool = False,
    corrupted_loader: Optional[Callable[[ImageCorruption], Iterable]] = None,
) -> list[dict]:
    """Evaluate every condition and return one record per condition, in order.

    ``corrupted_loader(corrupt_image)``, when given, returns a loader over the same
    images in the same order as ``loader``, with ``corrupt_image(image, position)``
    applied to each image as it is read (a backend's ``corrupted_dataset`` under a
    DataLoader). The on-the-fly corruptions then run in the loader's workers, alongside
    the GPU, instead of one image at a time in this process. The pixels are the same
    either way, so a results file can mix the two.

    Each new record is appended to ``results_path`` (JSONL) as soon as it is measured.
    A condition whose record is already there is skipped, but only if it was measured
    the same way (same op, magnitude, source and seed): a changed condition raises
    instead of reusing a stale number. ``overwrite`` re-measures everything, and the
    newest record per condition wins when the file is read back. ``metric()`` builds a
    fresh accumulator per condition. ``checkpoint`` and ``meta`` are copied into every
    record and must be JSON-serializable. Record them, e.g. ``meta={"arm": "control"}``
    or ``meta={"pruned_ops": [...]}`` for ``compare``.
    """
    conditions = list(conditions)
    keys = [condition.key for condition in conditions]
    duplicates = sorted({str(key) for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"conditions repeat: {duplicates}")

    existing = load_results(results_path) if os.path.exists(results_path) else []
    others = {record.get("checkpoint") for record in existing} - {checkpoint}
    if others:
        raise ValueError(
            f"{results_path} holds results for checkpoint(s) {sorted(map(str, others))}, "
            f"not {checkpoint!r}. One results file is one checkpoint."
        )
    done = {} if overwrite else latest(existing)
    meta = dict(meta or {})

    directory = os.path.dirname(os.path.abspath(results_path))
    os.makedirs(directory, exist_ok=True)
    records = []
    with open(results_path, "a") as log:
        for i, condition in enumerate(conditions, 1):
            key = (dataset, *condition.key)
            if key in done:
                _check_resumable(done[key], condition)
                records.append(done[key])
                continue
            accumulator = metric()
            source, perturb = loader, condition.perturb
            if condition.loader is not None:
                source = condition.loader()
            elif condition.corrupt_image is not None and corrupted_loader is not None:
                source, perturb = corrupted_loader(condition.corrupt_image), None
            result = evaluate(adapter, source, accumulator, perturb)
            if result["unchanged"]:
                logger.warning(
                    "[%s] %s %s=%s left the first batch bit-identical; its score is the clean score",
                    dataset,
                    condition.name,
                    condition.level_kind,
                    condition.level,
                )
            record = {
                "dataset": dataset,
                "checkpoint": checkpoint,
                "kind": condition.kind,
                "condition": condition.name,
                "level_kind": condition.level_kind,
                "level": condition.level,
                "info": _jsonable(condition.info),
                "primary": getattr(accumulator, "primary", None),
                "metrics": {name: _finite(value) for name, value in result["metrics"].items()},
                "per_class": _per_class_json(result["per_class"]),
                "n_images": result["n_images"],
                "unchanged": result["unchanged"],
                "meta": meta,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            records.append(record)
            primary = record["primary"]
            logger.info(
                "[%s] %d/%d %s %s=%s: %s=%s",
                dataset,
                i,
                len(conditions),
                condition.name,
                condition.level_kind,
                condition.level,
                primary,
                record["metrics"].get(primary),
            )

    counts = {r["n_images"] for r in records if r["dataset"] == dataset and r["kind"] != "prerendered"}
    if len(counts) > 1:
        logger.warning(
            "[%s] conditions on the same loader scored different numbers of images %s; "
            "their scores are not over the same set",
            dataset,
            sorted(counts),
        )
    return records


def load_results(path: str) -> list[dict]:
    """Every record in a results file, in file order."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def record_key(record: Mapping) -> tuple:
    return (record["dataset"], record["kind"], record["condition"], record["level_kind"], record["level"])


def latest(records: Iterable[Mapping]) -> dict:
    """``{record_key: record}``, the last record per key winning."""
    return {record_key(record): record for record in records}


def _target(adapter, label, batch):
    target = getattr(adapter, "target", None)
    if target is not None:
        return target(label, batch)
    if label is None:
        raise ValueError(
            "adapter.to_unit returned no label; a classification adapter must define "
            "target(label, batch) returning the class ids"
        )
    return label


def _check_resumable(record, condition):
    expected = _jsonable(condition.info)
    if record.get("info") != expected:
        raise ValueError(
            f"{condition.name} {condition.level_kind}={condition.level} is already in the results "
            f"file, measured as {record.get('info')}, but this run measures it as {expected}. "
            f"Use a new results file, or overwrite=True."
        )


# --- tables ---------------------------------------------------------------------------


def summarize(
    records: Iterable[Mapping],
    *,
    metric: Optional[str] = None,
    pruned_ops: Optional[Iterable[str]] = None,
) -> list[dict]:
    """One row per (dataset, kind, condition, level_kind), in first-seen order: the
    metric at each level, their mean, the worst level, and ``drop_vs_clean`` (clean
    minus mean, so positive is a loss). ``metric`` defaults to each record's primary.
    With ``pruned_ops``, a ``pruned`` column marks the ops the run trained without, the
    column eval_all_perturbations.py's summary.csv carried."""
    records = list(latest(records).values())
    pruned = None if pruned_ops is None else set(pruned_ops)

    groups: dict[tuple, list] = {}
    for record in records:
        groups.setdefault(
            (record["dataset"], record["kind"], record["condition"], record["level_kind"]), []
        ).append(record)
    clean_values = {
        dataset: rs[0]["metrics"].get(metric or rs[0]["primary"])
        for (dataset, kind, _, _), rs in groups.items()
        if kind == "clean"
    }

    rows = []
    for (dataset, kind, name, level_kind), rs in groups.items():
        metric_name = metric or rs[0]["primary"]
        values = {r["level"]: r["metrics"].get(metric_name) for r in rs}
        got = {level: value for level, value in values.items() if value is not None}
        mean = float(np.mean(list(got.values()))) if got else None
        worst_level = min(got, key=got.get) if got else None
        clean_value = clean_values.get(dataset)
        info = rs[0].get("info", {})
        row = {
            "dataset": dataset,
            "condition": name,
            "kind": kind,
            "family": info.get("family"),
            "op_kind": info.get("op_kind"),
            "category": info.get("category"),
            "metric": metric_name,
            "level_kind": level_kind,
            "n_levels": len(got),
            "mean": mean,
            "worst_level": worst_level,
            "worst": got.get(worst_level),
            "drop_vs_clean": None if clean_value is None or mean is None else clean_value - mean,
        }
        if pruned is not None:
            row["pruned"] = kind == "op" and name in pruned
        if level_kind != "none":
            for level in sorted(values):
                row[f"{level_kind}={level}"] = values[level]
        rows.append(row)
    return rows


def mce_from_records(
    records: Iterable[Mapping],
    *,
    dataset: str,
    kind: str = "prerendered",
    metric: str = "top1",
    baseline: Mapping[str, float] = ALEXNET_ERR,
) -> dict:
    """``corruptions.mce`` over one dataset's corruption records. The AlexNet baseline
    only means anything for the released ImageNet-C files (``kind="prerendered"``)."""
    accuracies: dict[str, dict] = {}
    for record in latest(records).values():
        if record["dataset"] == dataset and record["kind"] == kind:
            accuracies.setdefault(record["condition"], {})[record["level"]] = record["metrics"][metric]
    return mce({name: [by_level[s] for s in sorted(by_level)] for name, by_level in accuracies.items()}, baseline)


def write_csv(path: str, rows: Sequence[Mapping]) -> None:
    """``rows`` as CSV. The columns are the union of the rows' keys in first-seen order,
    and a row missing a column leaves it blank."""
    columns: list[str] = []
    for row in rows:
        columns += [key for key in row if key not in columns]
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _nanmean(x: np.ndarray) -> float:
    finite = x[~np.isnan(x)]
    return float(finite.mean()) if finite.size else math.nan


def _finite(value) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def _per_class_json(per_class):
    if per_class is None:
        return None
    return {name: [_finite(v) for v in values] for name, values in per_class.items()}


def _jsonable(value):
    return json.loads(json.dumps(value))
