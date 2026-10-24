"""The two probe hooks of ``tasks/mmseg/hooks.py``, for a hand-written classification
training loop. Glue: they gate and call ``probe``, ``stats`` and ``selection``, and
compute nothing themselves.

    CollectGradientCallback  when the host's schedule fires: ``probe.sweep`` on the
                             live model -> one G, returned
    ApplyWeightsCallback     on that G: ``stats.analyze_sweep`` -> the logs ->
                             ``selection.decide`` -> ``host.set_weights(q)``

A loop calls them after each optimizer step, with the number of COMPLETED iterations
(mmengine's ``runner.iter + 1``), collect first::

    host = build_host("uniform", vocabulary=ops, probe_interval=5000)
    backend = ClsBackend(model, data=dict(type="imagefolder", root=val_root))
    collect = CollectGradientCallback(backend, host, magnitude=0.5, n_probe=2000)
    apply = ApplyWeightsCallback(host, selector=dict(type="kl_reweight", lam=0.25))
    for callback in (collect, apply):
        callback.before_run(work_dir, max_iters, resume=resume)
    for iteration in range(start + 1, max_iters + 1):
        ...  # one training step, augmentation drawn from host.sampling_probs()
        apply.after_train_iter(iteration, collect.after_train_iter(iteration))

``backend.model`` must be the live model, unwrapped from DDP. The sweep puts it in
eval() and restores every module's mode afterwards. Nothing here applies the host's
draws to training images; the loop does that. The files written are the mmseg hooks'
(``probe_set.json``, ``aug_gradient_log.txt`` and ``stats.logs``' three), so the
offline tools read either.

DDP, as in the mmseg hooks: every rank sweeps the whole probe set, rank 0 writes the
logs, and rank 0's q is broadcast so every rank installs the same one.
"""

import logging
import os
from collections.abc import Mapping
from typing import Optional, Union

import numpy as np
import torch.distributed as dist

import probe
import sa
import stats
from selection import build_selector, decide, measured_red, ramp_lambda, redundancy_of
from vocab import get_op

from ..magnitudes import probe_magnitudes

__all__ = ["ApplyWeightsCallback", "CollectGradientCallback"]

logger = logging.getLogger(__name__)

GRADIENT_LOG = "aug_gradient_log.txt"
PROBE_SET_FILE = "probe_set.json"


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _main() -> bool:
    return not _distributed() or dist.get_rank() == 0


class CollectGradientCallback:
    """Sweep the probe set on the frozen model whenever the host's schedule fires.

    magnitude    a float is one STRENGTH in [0, 1] for every op; a mapping gives each
                 op its own magnitude in its own units (as in the mmseg hook)
    magnitude_mode
                 ``"fixed"`` always probes at ``magnitude``. Any other mode probes a
                 host with magnitude bins of its own (randaugment, trivialaugment,
                 autoaugment) at them (``tasks.magnitudes``); there is no SA loop
                 here, so every other host uses ``magnitude``
    n_probe      images in the probe set; None sweeps the whole dataset
    probe_seed   picks the subset and pins each op's randomness
    batch_size   images per forward, and the bootstrap's cluster. Larger than the
                 mmseg default of 1: class labels always stack, and a classifier's
                 backward is cheap.
    fd_step      finite-difference step for ops autograd can't differentiate
    fade         the ops faded in instead (``probe.fade``): None, ``"non_differentiable"``
                 or a list of op names
    fade_at      the ``a`` a faded op's row is taken at (1 = the augmented image)
    probe_pruned also measure the ops the host prunes (``host.pruned()``); False leaves
                 them out with NaN rows labelled ``static_pruned`` (as in the mmseg hook)
    """

    def __init__(
        self,
        backend,
        host,
        magnitude: Union[float, Mapping] = 0.5,
        n_probe: Optional[int] = None,
        probe_seed: int = 0,
        batch_size: int = 16,
        fd_step=probe.DEFAULT_STEP,
        num_workers: int = 0,
        magnitude_mode: str = sa.MODE_MODAL,
        fade=None,
        fade_at: float = 1.0,
        probe_pruned: bool = False,
    ):
        if isinstance(magnitude, Mapping):
            fallback = probe.Magnitudes.fixed(dict(magnitude))
        else:
            fallback = probe.Magnitudes.relative(float(magnitude))
        self.backend = backend
        self.host = host
        self.ops = {name: get_op(name) for name in host.vocabulary()}
        self.magnitudes = probe_magnitudes(host, None, magnitude_mode, fallback)
        self.fade = fade
        self.fade_at = probe.check_fade_at(fade_at)
        # Draw once now, so a bad magnitude or fade fails at construction, not at the first probe.
        fallback.draw(self.ops, 1, np.random.default_rng(0))
        self.magnitudes.draw(self.ops, 1, np.random.default_rng(0))
        self.faded = probe.faded_ops(self.ops, fade)
        self.skip = () if probe_pruned else host.pruned()
        self.n_probe = n_probe
        self.probe_seed = probe_seed
        self.batch_size = batch_size
        self.fd_step = fd_step
        self.num_workers = num_workers

    def before_run(self, work_dir: str, max_iters: int, *, resume: bool = False) -> None:
        self.max_iters = int(max_iters)
        self.data = self.backend.probe_data()
        path = os.path.join(work_dir, PROBE_SET_FILE)
        if resume and os.path.exists(path):
            self.probe_set = probe.ProbeSet.load(path)
            self.probe_set.verify(self.data.ids)
        else:
            self.probe_set = probe.ProbeSet.build(
                self.data.ids, self.n_probe, seed=self.probe_seed, source=self.data.source
            )
            if _main():
                self.probe_set.save(path)

        self.log_path = os.path.join(work_dir, GRADIENT_LOG)
        if _main() and not resume and os.path.exists(self.log_path):
            os.remove(self.log_path)
        if not any(self.host.should_probe(i) for i in range(1, self.max_iters + 1)):
            logger.warning(
                "[probe] host %r never probes within %d iterations; give it probe_interval or probe_iters",
                self.host.name,
                self.max_iters,
            )
        logger.info(
            "[probe] %d ops (%d faded at a=%g, %d pruned and not measured), probe set %s (%d images of %s), "
            "magnitudes %s/%s",
            len(self.ops),
            len(self.faded),
            self.fade_at,
            len(self.skip),
            self.probe_set.digest[:12],
            len(self.probe_set),
            self.data.source,
            self.magnitudes.source,
            self.magnitudes.mode,
        )

    def after_train_iter(self, iteration: int) -> Optional[probe.Sweep]:
        """The sweep taken after ``iteration`` completed iterations, or None when the
        host's schedule doesn't fire there."""
        role = self.host.probe_role(iteration)
        if role is None:
            return None
        return probe.sweep(
            self.backend,
            self.probe_set,
            self.data.dataset,
            self.ops,
            self.magnitudes,
            checkpoint=iteration / self.max_iters if self.max_iters else 0.0,
            batch_size=self.batch_size,
            probe_seed=self.probe_seed,
            fd_step=self.fd_step,
            num_workers=self.num_workers,
            log_path=self.log_path if _main() else None,
            meta={"iter": iteration, "role": role},
            fade=self.fade,
            fade_at=self.fade_at,
            skip=self.skip,
        )


class ApplyWeightsCallback:
    """Turn a sweep into R, log it, and hand the host q.

    selector     ``dict(type=<selection name>, **kwargs)``; its ``lam`` is the target
                 strength, ramped by training progress. lam=0 (the default) measures
                 and logs R and never calls ``set_weights``: the control arm.
    lambda_ramp  "linear" or "constant"
    everything else
                 ``stats.analyze``'s keywords (n_min, normalize_per_image, bootstrap,
                 bootstrap_reps, bootstrap_seed, fdr_alpha, fdr_method, shrinkage)
    """

    def __init__(
        self,
        host,
        selector: Optional[dict] = None,
        lambda_ramp: str = "linear",
        mask_within_op: bool = True,
        **analysis_kwargs,
    ):
        selector = dict(selector or {"type": "kl_reweight", "lam": 0.0})
        self.host = host
        self.selector_name = selector.pop("type")
        if self.selector_name == "nms":
            raise ValueError(
                "selector nms ranks ops by loss increase and latches its prune; only the mmseg "
                "PruneHook runs it so far"
            )
        self.lam = float(selector.pop("lam", 0.0))
        self.selector_kwargs = selector
        # Fail at construction on an unknown selector, ramp or analysis keyword.
        build_selector(self.selector_name, lam=self.lam, **self.selector_kwargs)
        ramp_lambda(self.lam, 0.0, lambda_ramp)
        # A zero-image G is skipped below n_min, after analyze has checked every keyword.
        stats.analyze(np.zeros((2, 0)), np.zeros(0), ["a", "b"], **{**analysis_kwargs, "n_min": 1})
        self.lambda_ramp = lambda_ramp
        self.mask_within_op = mask_within_op
        self.analysis_kwargs = analysis_kwargs

    def before_run(self, work_dir: str, max_iters: int, *, resume: bool = False) -> None:
        self.work_dir = work_dir
        self.max_iters = int(max_iters)
        if _main() and not resume:
            stats.reset_logs(work_dir)

    def after_train_iter(self, iteration: int, sweep: Optional[probe.Sweep]):
        """The Selection taken on ``sweep`` (applied or declined), or None when there
        was no sweep at this iteration."""
        if sweep is None or sweep.meta.get("iter") != iteration:
            return None
        role = sweep.meta["role"]
        analysis = stats.analyze_sweep(sweep, static_pruned=self.host.pruned(), **self.analysis_kwargs)
        if _main():
            stats.write_analysis(self.work_dir, analysis)
            logger.info("[grad-corr] iter %d (%s): %s", iteration, role, analysis.summary())

        progress = iteration / self.max_iters if self.max_iters else 0.0
        lam = ramp_lambda(self.lam, progress, self.lambda_ramp)
        rel = dict(zip(self.host.vocabulary(), self.host.base_probs()))
        selector = build_selector(self.selector_name, lam=lam, **self.selector_kwargs)
        selected = decide(analysis, rel, selector, role=role, mask_within_op=self.mask_within_op)

        decision = [(selected.applied, selected.reason, selected.q)]
        if _distributed():
            dist.broadcast_object_list(decision, src=0)
        applied, reason, q = decision[0]
        if applied:
            try:
                self.host.set_weights(q)
            except (RuntimeError, ValueError) as error:
                # Declined, not raised: this runs mid-training on every rank.
                applied, reason = False, f"host refused q: {error}"

        if not _main():
            return selected
        if role != "active" or analysis.withheld is not None:
            logger.log(logging.INFO if role != "active" else logging.WARNING, "[grad-corr] q not taken: %s", reason)
            return selected
        score = measured_red(redundancy_of(analysis, self.mask_within_op), self.selector_kwargs.get("mode", "squared"))
        stats.append_jsonl(
            os.path.join(self.work_dir, stats.REDUNDANCY_LOG),
            stats.redundancy_record(
                score,
                analysis,
                self.mask_within_op,
                selector=self.selector_name,
                lam=lam,
                lam_target=self.lam,
                applied=applied,
                reason=reason,
                q=dict(zip(self.host.vocabulary(), np.asarray(q).tolist())) if applied else None,
                pruned=list(selected.pruned) if applied else [],
                covers=[list(pair) for pair in selected.covers] if applied else [],
            ),
        )
        if applied:
            logger.info(
                "[redundancy] iter %d: %s at lambda=%.3f (target %g, %s ramp), max/min q spread %.1fx%s",
                iteration,
                self.selector_name,
                lam,
                self.lam,
                self.lambda_ramp,
                selected.spread,
                (f", pruned {list(selected.pruned)}" if selected.pruned else "")
                + (f", covers {dict(selected.covers)}" if selected.covers else ""),
            )
        else:
            # Never silently: a q that quietly did nothing looks like a real arm.
            logger.log(
                logging.INFO if self.lam == 0 else logging.WARNING,
                "[redundancy] iter %d: q NOT applied (%s): %s",
                iteration,
                self.selector_name,
                reason,
            )
        return selected
