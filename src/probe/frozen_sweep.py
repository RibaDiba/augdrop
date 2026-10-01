"""Checkpoints x probe set -> G, one G per checkpoint.

Ported from Sensaug's ``CollectGradientHook._sweep`` and its post-hoc driver
scripts/compute_grad_corr.py. Sensaug fired the sweep from inside a training loop
(``after_train_iter``) and needed an mmengine Runner proxy to call it post hoc. Here
the sweep is a plain function of (adapter, probe set, ops, magnitudes), which makes
the post-hoc case the direct one. A training loop that wants R mid-run calls
``sweep`` when its host's ``should_probe(iteration)`` says so.

G is (A ops x N probe images). Every column of one G describes the SAME weights and
the SAME magnitude setting:

- The model is put in eval() for the whole sweep and nothing steps it.
  ``torch.autograd.grad`` never populates parameter ``.grad``. Every module's train/
  eval flag is restored exactly afterwards.
- Magnitudes come from a generator seeded once per sweep and advanced across batches.
  Reseeding per batch would hand every batch the same draw (at batch size 1, one
  magnitude for the whole set). Because the seed is per sweep, every checkpoint in
  ``sweep_checkpoints`` is probed at identical magnitudes, and its Gs differ by the
  weights alone.
- One ``Sweep`` is one G. Nothing here concatenates sweeps. Pooling checkpoints mixes
  convergence drift into R (Simpson's paradox), and pooling magnitude settings mixes
  measurements that are not commensurable.

Batch boundaries are kept, as one (B,) chunk per batch per op, because the downstream
cluster bootstrap resamples whole batches. A batch that fails transiently (a
RuntimeError such as CUDA OOM) is left out of G whole and listed in
``failed_batches``, so the rows stay aligned. A ``ProbeError`` is never transient,
and neither is a failure on every one of the first batches.
"""

import contextlib
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from .finite_diff import DEFAULT_STEP
from .grad_collect import (
    UNPROBED,
    FdStep,
    ModelAdapter,
    ProbeError,
    collect_grads,
    op_method,
    preserved_rng,
)
from .probe_set import ProbeSet

logger = logging.getLogger(__name__)

# (ops, batch size, generator) -> {op name: (B,) magnitudes in the op's own units}.
# Only ops with a magnitude need an entry.
MagnitudeDraw = Callable[[Mapping[str, Any], int, np.random.Generator], Mapping[str, np.ndarray]]

# Batches that may fail before any succeeds. Past this the failure is systematic (a
# shape bug raises RuntimeError too), not transient, and the sweep stops.
_MAX_LEADING_FAILURES = 3


def strength_to_magnitude(op, strength: float) -> float:
    """The magnitude a fraction ``strength`` of the way from ``op.identity_at`` to the
    far end of ``op.mag_range``. 0 is the identity and 1 is the strongest magnitude
    the range allows.

    Toward the far end, and toward ``hi`` on a tie. That matches torchvision's
    auto-augment bins: Solarize and Posterize (identity at the top of the range)
    strengthen downward, and a symmetric signed op like Rotate strengthens in its
    positive direction. On Sensaug's unsigned [0, 1] ops the strength IS the
    magnitude.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    lo, hi = op.mag_range
    anchor = op.identity_at
    # A tie up to float error is a tie: 1.0 - 0.1 exceeds 1.9 - 1.0 by one ulp.
    below_is_farther = (anchor - lo) - (hi - anchor) > 1e-9 * (hi - lo)
    far = lo if below_is_farther else hi
    return anchor + strength * (far - anchor)


@dataclass(frozen=True)
class Magnitudes:
    """Where each op's probe magnitude comes from, and the two labels every G carries.

    Magnitudes are not commensurable: at a fixed 0.5, Sensaug's ``blur`` moved a pixel
    by at most 78/255 and ``noise`` by 251/255, and the vocab's families don't even
    share units. A G whose magnitudes are unlabelled can't be compared with anything,
    so ``source`` and ``mode`` are required.

    ``draw`` is called once per batch with the sweep's own generator. A sampled mode
    (Sensaug's ``resolve_magnitudes``) plugs in here.
    """

    draw: MagnitudeDraw
    source: str
    mode: str

    @classmethod
    def fixed(cls, value, *, source: str = "fixed") -> "Magnitudes":
        """The same magnitude for every image, in each op's own units: one value for
        all ops, or a mapping with one value per op."""
        per_op = {n: float(v) for n, v in value.items()} if isinstance(value, Mapping) else None

        def draw(ops, batch_size, rng):
            out = {}
            for name, op in ops.items():
                if op_method(op) == UNPROBED:
                    continue
                if per_op is not None and name not in per_op:
                    raise ValueError(f"no fixed magnitude for op {name!r}")
                out[name] = np.full(batch_size, per_op[name] if per_op is not None else float(value))
            return out

        return cls(draw=draw, source=source, mode="fixed")

    @classmethod
    def relative(cls, strength: float, *, source: str = "relative") -> "Magnitudes":
        """The same STRENGTH for every op (see ``strength_to_magnitude``), converted
        into each op's own units. What makes one reference meaningful across a vocab
        whose families measure magnitude in degrees, pixels and enhance factors. The
        same strength is still not the same distortion for two different ops."""

        def draw(ops, batch_size, rng):
            return {
                name: np.full(batch_size, strength_to_magnitude(op, strength))
                for name, op in ops.items()
                if op_method(op) != UNPROBED
            }

        return cls(draw=draw, source=source, mode="fixed")


@dataclass
class Sweep:
    """One G: every op x every successfully probed image, on one model state at one
    magnitude setting.

    ``grads[op]`` and ``magnitudes[op]`` hold one (B,) chunk per batch, and
    ``image_ids[k]`` holds batch k's ids. ``matrix()`` stacks them into G. An op whose
    ``methods`` entry is ``"unprobed"`` has no magnitude, and its rows are NaN.
    """

    names: list[str]
    checkpoint: Any
    probe_set: str
    magnitude_source: str
    magnitude_mode: str
    probe_seed: int
    methods: dict[str, str]
    meta: dict = field(default_factory=dict)
    grads: dict[str, list[np.ndarray]] = field(default_factory=dict)
    magnitudes: dict[str, list[np.ndarray]] = field(default_factory=dict)
    image_ids: list[tuple[str, ...]] = field(default_factory=list)
    failed_batches: list[dict] = field(default_factory=list)

    def __post_init__(self):
        for name in self.names:
            self.grads.setdefault(name, [])
            self.magnitudes.setdefault(name, [])

    @property
    def n_images(self) -> int:
        return sum(len(ids) for ids in self.image_ids)

    @property
    def n_batches(self) -> int:
        return len(self.image_ids)

    @property
    def unprobed(self) -> list[str]:
        return [name for name in self.names if self.methods[name] == UNPROBED]

    def ids(self) -> list[str]:
        """G's column labels, in column order."""
        return [image_id for batch in self.image_ids for image_id in batch]

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """``(G, batch_ids)``: G is (A, N) with rows in ``names`` order, and
        ``batch_ids[j]`` is the batch column j came from, the cluster the bootstrap
        resamples.

        Rows line up because every op is probed on the same batch and a batch is kept
        or dropped for all ops together. That is an invariant, so a mismatch raises
        instead of being truncated away, which would misalign the columns silently.
        """
        rows = [
            np.concatenate(self.grads[name]) if self.grads[name] else np.empty(0)
            for name in self.names
        ]
        lengths = {row.shape[0] for row in rows}
        if lengths != {self.n_images}:
            raise ProbeError(
                f"G's rows are misaligned: lengths {sorted(lengths)} for {self.n_images} "
                f"probed images"
            )
        batch_ids = np.concatenate(
            [np.full(len(ids), k, dtype=int) for k, ids in enumerate(self.image_ids)]
            or [np.empty(0, dtype=int)]
        )
        return np.stack(rows), batch_ids

    def magnitude_summary(self) -> dict[str, dict]:
        """Per-op mean/min/max of the magnitudes actually probed. For a constant mode,
        min == max is the check that it stayed constant."""
        summary = {}
        for name in self.names:
            chunks = self.magnitudes[name]
            if chunks and name not in self.unprobed:
                flat = np.concatenate(chunks)
                summary[name] = {
                    "mean": float(flat.mean()),
                    "min": float(flat.min()),
                    "max": float(flat.max()),
                }
        return summary


@contextlib.contextmanager
def frozen(model: torch.nn.Module):
    """eval() for the duration, then every module's own train/eval flag back as it was.
    Not a blanket ``model.train()``, which would unfreeze modules the caller had
    deliberately left in eval."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield model
    finally:
        # modules() yields parents before children, so a parent's recursive train()
        # is overridden by each child's own flag below it.
        for module, training in modes:
            module.train(training)


def sweep(
    adapter: ModelAdapter,
    probe_set: ProbeSet,
    dataset,
    ops: Mapping[str, Any],
    magnitudes: Magnitudes,
    *,
    checkpoint: Any = None,
    batch_size: int = 1,
    probe_seed: int = 0,
    fd_step: FdStep = DEFAULT_STEP,
    num_workers: int = 0,
    log_path: Optional[str] = None,
    meta: Optional[dict] = None,
) -> Sweep:
    """One frozen-frame pass of ``adapter``'s model over ``probe_set``. Returns one G.

    ``dataset`` is the map-style dataset ``probe_set`` was built from, yielding CLEAN
    items (no random augmentation). ``ops`` is an ordered vocab (``vocab.build_vocab``)
    and fixes G's row order. ``checkpoint`` and ``meta`` are labels copied into the
    result and the log; ``checkpoint`` must be JSON-serializable when logging. With
    ``log_path``, one JSONL record per batch is appended to it (Sensaug's
    ``aug_gradient_log.txt`` format, plus image ids), which is enough to recompute G
    offline.

    ``batch_size`` is also the bootstrap's cluster size. Sensaug defaulted to 1
    (singleton clusters). Larger is faster, but activation memory scales with it,
    since the backward reaches the full-resolution input.
    """
    if not ops:
        raise ValueError("no ops to probe")
    names = list(ops)
    result = Sweep(
        names=names,
        checkpoint=checkpoint,
        probe_set=probe_set.digest,
        magnitude_source=magnitudes.source,
        magnitude_mode=magnitudes.mode,
        probe_seed=probe_seed,
        methods={name: op_method(op) for name, op in ops.items()},
        meta=dict(meta or {}),
    )
    loader = probe_set.loader(
        dataset,
        batch_size,
        collate_fn=getattr(adapter, "collate", None),
        num_workers=num_workers,
    )
    batch_ids = getattr(adapter, "batch_ids", None)
    rng = np.random.default_rng(probe_seed)
    checked = False

    with contextlib.ExitStack() as stack:
        stack.enter_context(preserved_rng())
        stack.enter_context(frozen(adapter.model))
        log = stack.enter_context(open(log_path, "a")) if log_path else None

        for k, (ids, batch) in enumerate(zip(probe_set.chunks(batch_size), loader, strict=True)):
            # Drawn before probing, so a failed batch still consumes its draw and the
            # later batches' magnitudes don't depend on which batches failed.
            drawn = _draw(magnitudes, ops, len(ids), rng)
            if batch_ids is not None and tuple(batch_ids(batch)) != tuple(ids):
                raise ProbeError(
                    f"batch {k} holds {list(batch_ids(batch))[:3]}..., the probe set "
                    f"expects {list(ids)[:3]}... The loader is out of probe order."
                )
            try:
                rows = collect_grads(
                    adapter,
                    batch,
                    ops,
                    drawn,
                    probe_seed=probe_seed,
                    fd_step=fd_step,
                    check=not checked,
                )
            except ProbeError:
                raise
            except RuntimeError as error:
                result.failed_batches.append(
                    {"batch": k, "image_ids": list(ids), "error": repr(error)}
                )
                if result.n_batches == 0 and len(result.failed_batches) >= _MAX_LEADING_FAILURES:
                    raise RuntimeError(
                        f"the first {len(result.failed_batches)} probe batches all failed, "
                        f"which is systematic, not transient"
                    ) from error
                logger.warning("probe batch %d failed (checkpoint %s): %r", k, checkpoint, error)
                continue
            checked = True

            result.image_ids.append(tuple(ids))
            for name in names:
                result.grads[name].append(rows[name])
                result.magnitudes[name].append(drawn[name])

            if log is not None:
                log.write(json.dumps(_record(result, k, ids, rows, drawn)) + "\n")
                log.flush()

    if result.n_images == 0:
        raise RuntimeError(
            f"no probe batch succeeded at checkpoint {checkpoint}: "
            f"{[f['error'] for f in result.failed_batches][:3]}"
        )
    logger.info(
        "checkpoint %s: probed %d images in %d batches (%d failed), %d ops (%d unprobed), "
        "magnitudes %s/%s, probe set %s",
        checkpoint,
        result.n_images,
        result.n_batches,
        len(result.failed_batches),
        len(names),
        len(result.unprobed),
        magnitudes.source,
        magnitudes.mode,
        probe_set.digest[:12],
    )
    return result


def sweep_checkpoints(
    checkpoints: Sequence[Any],
    load_adapter: Callable[[Any], ModelAdapter],
    probe_set: ProbeSet,
    dataset,
    ops: Mapping[str, Any],
    magnitudes: Magnitudes,
    **sweep_kwargs,
) -> list[Sweep]:
    """One G per checkpoint, same probe set, same magnitudes, never pooled.

    ``load_adapter(checkpoint)`` returns an adapter holding that checkpoint's weights.
    Each result is labelled with ``str(checkpoint)``.
    """
    sweeps = []
    for checkpoint in checkpoints:
        adapter = load_adapter(checkpoint)
        sweeps.append(
            sweep(
                adapter,
                probe_set,
                dataset,
                ops,
                magnitudes,
                checkpoint=str(checkpoint),
                **sweep_kwargs,
            )
        )
        # Release this checkpoint's model before the next one loads, so two are
        # never resident at once.
        del adapter
    return sweeps


def _draw(magnitudes: Magnitudes, ops, batch_size, rng) -> dict[str, np.ndarray]:
    """One batch's magnitudes, validated, with NaN for the ops that have none."""
    drawn = magnitudes.draw(ops, batch_size, rng)
    out = {}
    for name, op in ops.items():
        if op_method(op) == UNPROBED:
            out[name] = np.full(batch_size, np.nan)
            continue
        if name not in drawn:
            raise ValueError(f"magnitude source {magnitudes.source!r} gave nothing for op {name!r}")
        value = np.asarray(drawn[name], dtype=np.float64)
        if value.shape != (batch_size,):
            raise ValueError(
                f"magnitude source {magnitudes.source!r} gave op {name!r} shape "
                f"{value.shape}, expected ({batch_size},)"
            )
        out[name] = value
    return out


def _jsonable(values: np.ndarray) -> list:
    """NaN -> None. json.dumps writes a bare ``NaN``, which jq and pandas reject."""
    return [float(v) if math.isfinite(v) else None for v in values.tolist()]


def _record(result: Sweep, k: int, ids, rows, drawn) -> dict:
    """One aug_gradient_log.txt line."""
    return {
        "checkpoint": result.checkpoint,
        "batch": k,
        "batch_size": len(ids),
        "image_ids": list(ids),
        "probe_set": result.probe_set,
        "magnitude_source": result.magnitude_source,
        "magnitude_mode": result.magnitude_mode,
        "probe_seed": result.probe_seed,
        "methods": result.methods,
        "meta": result.meta,
        "grads": {name: _jsonable(rows[name]) for name in result.names},
        "magnitudes": {name: _jsonable(drawn[name]) for name in result.names},
    }
