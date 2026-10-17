"""mmengine glue for the probe and the q it produces. Nothing is computed here.

    CollectGradientHook  when the host's schedule fires: ``probe.sweep`` on the live
                         model -> one G, stashed on the runner as ``probe_sweep``
    ApplyWeightsHook     on that G: ``stats.analyze_sweep`` -> the logs -> a
                         ``selection`` selector -> ``host.set_weights(q)``

Replaces Sensaug's ``CollectGradientHook`` (grad_hook.py) and
``PerturbationSensitivityAnalysisHookWithGradients`` (grad_sens_analysis.py) plus the
Lever 3 consumer in ``GradCorrValLoop``. Both hooks act in ``after_train_iter`` on the
same iteration, and ApplyWeightsHook must see the sweep CollectGradientHook just took,
so their priorities are load-bearing: NORMAL (50) runs before LOW (70).

With the SA loop (``loops.RobustValLoop``) the split is Sensaug's: the hooks publish,
and the loop applies. CollectGradientHook probes at the magnitudes the loop last
published (``runner.corr_magnitudes``; another host's are its own,
``tasks.magnitudes``), and ApplyWeightsHook keeps the latest R for
the loop to re-apply to every round's fresh pdf (``reweight``) at that round's lambda.
The val loop runs after ``after_train_iter`` on a shared iteration, so a probe that
fires one hook point before an SA round reaches that round's pdf.

The host is the one on the model's ``GpuAugSegDataPreProcessor`` (``.host``). It says
when to probe (``should_probe`` / ``probe_role``, evaluated at the completed-iteration
count ``runner.iter + 1``, identical on every rank), which ops to probe (its
vocabulary, looked up in ``vocab``), and it receives q. A ``"control"`` probe is
measured and logged but never becomes a q.

DDP. The probe's sharding and gather are not ported, so every rank sweeps the whole
probe set (same wall time, no rank idles into a collective timeout), and rank 0's q is
broadcast so every rank installs the identical q. Only rank 0 writes logs.

Config::

    custom_imports = dict(imports=["tasks.mmseg.register"], allow_failed_imports=False)
    model = dict(data_preprocessor=dict(
        type="GpuAugSegDataPreProcessor", host=dict(type="sensaug", max_iters=160000), ...))
    custom_hooks = [
        dict(type="CollectGradientHook", magnitude=0.5),
        dict(type="ApplyWeightsHook", selector=dict(type="kl_reweight", lam=0.25)),
    ]
"""

import os
from collections.abc import Mapping
from typing import Optional, Union

import numpy as np
from mmengine.dist import broadcast_object_list, is_main_process
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS

import probe
import sa
import stats
from selection import CLEAR, KEEP, Redundancy, build_selector, measured_red, publication, ramp_lambda
from vocab import get_op

from ..magnitudes import probe_magnitudes
from .backend import MMSegBackend, unwrap

__all__ = ["ApplyWeightsHook", "CollectGradientHook", "host_of"]

GRADIENT_LOG = "aug_gradient_log.txt"
PROBE_SET_FILE = "probe_set.json"


def host_of(runner):
    """The host on the model's data preprocessor."""
    host = getattr(unwrap(runner.model).data_preprocessor, "host", None)
    if host is None:
        raise TypeError(
            "the probe hooks read their schedule, vocabulary and weights from the host on "
            "the model's data preprocessor, and it has none. Set "
            "model.data_preprocessor.type='GpuAugSegDataPreProcessor' with a host= "
            "config (augment=False for a clean-trained control)."
        )
    return host


def _resuming(runner) -> bool:
    return bool(getattr(runner, "_resume", False))


def _progress(runner, iteration: int) -> float:
    max_iters = getattr(runner, "max_iters", 0) or 0
    return iteration / max_iters if max_iters else 0.0


@HOOKS.register_module()
class CollectGradientHook(Hook):
    """Sweep the probe set on the frozen model whenever the host's schedule fires.

    magnitude    a float is one STRENGTH in [0, 1] for every op (0 identity, 1 the far
                 end of its range; on the DIFF32 ops it is the magnitude itself, and
                 0.5 is Sensaug's fixed fallback). A mapping gives each op its own
                 magnitude in its own units. With an SA snapshot (below) this is the
                 fallback, for the ops the snapshot lacks and before the first one.
    magnitude_mode
                 where the magnitude comes from once RobustValLoop has published an SA
                 snapshot (``runner.corr_magnitudes``): ``"mode"`` (Sensaug's default,
                 each op at its modal SA level), ``"sampled_shared"``,
                 ``"sampled_independent"``, or ``"fixed"`` (always ``magnitude``).
                 See ``sa.magnitudes``. Read once per sweep. A host with magnitude bins
                 of its own (randaugment, trivialaugment, autoaugment) is probed at
                 them in every mode but ``"fixed"`` (``tasks.magnitudes``).
    n_probe      images in the probe set; None sweeps the whole split, as Sensaug did
    probe_seed   picks the subset and pins each op's randomness (``noise``)
    batch_size   images per forward, and the bootstrap's cluster. 1 is Sensaug's
                 default and the only size that stacks labels of mixed native sizes.
    fd_step      finite-difference step for ops autograd can't differentiate
    split        the dataloader config the probe reads ("val" or "test")
    fade         the ops faded in instead (``probe.fade``): None, ``"non_differentiable"``
                 or a list of op names. The host's ``fade:`` config block.
    fade_at      the ``a`` a faded op's row is taken at (1 = the augmented image)
    probe_pruned also measure the ops the host prunes (``host.pruned()``, the ``prune:``
                 block's ``probe``). False (the default, Sensaug's) leaves them out of
                 the sweep with NaN rows labelled ``static_pruned``. Either way the R
                 ApplyWeightsHook analyses sets them aside, so q is the same; True only
                 puts their rows in ``aug_gradient_log.txt``

    The probe set is saved to ``work_dir/probe_set.json`` and, on resume, reloaded and
    checked against the dataset, so every G of one run is on the same images.
    """

    priority = "NORMAL"

    def __init__(
        self,
        magnitude: Union[float, Mapping] = 0.5,
        magnitude_mode: str = sa.MODE_MODAL,
        n_probe: Optional[int] = None,
        probe_seed: int = 0,
        batch_size: int = 1,
        fd_step=probe.DEFAULT_STEP,
        num_workers: int = 0,
        split: str = "val",
        fade=None,
        fade_at: float = 1.0,
        probe_pruned: bool = False,
    ):
        if isinstance(magnitude, Mapping):
            self.magnitudes = probe.Magnitudes.fixed(dict(magnitude))
        else:
            self.magnitudes = probe.Magnitudes.relative(float(magnitude))
        if magnitude_mode not in sa.MAGNITUDE_MODES:
            raise ValueError(f"magnitude_mode {magnitude_mode!r} is not one of {sa.MAGNITUDE_MODES}")
        self.magnitude_mode = magnitude_mode
        self.n_probe = n_probe
        self.probe_seed = probe_seed
        self.batch_size = batch_size
        self.fd_step = fd_step
        self.num_workers = num_workers
        self.split = split
        self.fade = fade
        self.fade_at = probe.check_fade_at(fade_at)
        self.probe_pruned = bool(probe_pruned)

    def before_run(self, runner) -> None:
        self.host = host_of(runner)
        self.ops = {name: get_op(name) for name in self.host.vocabulary()}
        # Draw once now, so a bad magnitude or fade fails at startup, not at the first probe.
        rng = np.random.default_rng(0)
        self.magnitudes.draw(self.ops, 1, rng)
        probe_magnitudes(self.host, None, self.magnitude_mode, self.magnitudes).draw(self.ops, 1, rng)
        faded = probe.faded_ops(self.ops, self.fade)

        self.backend = MMSegBackend.from_runner(runner, split=self.split)
        self.data = self.backend.probe_data()

        path = os.path.join(runner.work_dir, PROBE_SET_FILE)
        if _resuming(runner) and os.path.exists(path):
            self.probe_set = probe.ProbeSet.load(path)
            self.probe_set.verify(self.data.ids)
        else:
            self.probe_set = probe.ProbeSet.build(
                self.data.ids, self.n_probe, seed=self.probe_seed, source=self.data.source
            )
            if is_main_process():
                self.probe_set.save(path)

        self.log_path = os.path.join(runner.work_dir, GRADIENT_LOG)
        if is_main_process() and not _resuming(runner) and os.path.exists(self.log_path):
            os.remove(self.log_path)
        runner.probe_sweep = None

        if not any(self.host.should_probe(i) for i in range(1, runner.max_iters + 1)):
            print_log(
                f"[probe] host {self.host.name!r} never probes within {runner.max_iters} "
                f"iterations; give it probe_interval or probe_iters",
                logger="current",
                level=30,
            )
        pruned = self.host.pruned()
        self.skip = () if self.probe_pruned else pruned
        if pruned:
            print_log(
                f"[probe] the run prunes {len(pruned)} op(s), "
                + ("still measured (probe_pruned): " if self.probe_pruned else "not measured, NaN rows: ")
                + ", ".join(pruned),
                logger="current",
            )
        print_log(
            f"[probe] {len(self.ops)} ops ({len(faded)} faded at a={self.fade_at:g}), probe set "
            f"{self.probe_set.digest[:12]} ({len(self.probe_set)} images of {self.data.source}), "
            f"magnitudes {self.magnitude_mode}: "
            + (
                f"the {self.host.name} host's own bins"
                if self.magnitude_mode != sa.MODE_FIXED and self.host.num_magnitude_bins is not None
                else f"from the SA snapshot once there is one, else "
                f"{self.magnitudes.source}/{self.magnitudes.mode}"
            ),
            logger="current",
        )

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        iteration = runner.iter + 1
        role = self.host.probe_role(iteration)
        if role is None:
            return
        # Whatever the SA loop published last, read once so the whole sweep is at one
        # set of magnitudes. Empty before the first SA round: the fallback.
        magnitudes = probe_magnitudes(
            self.host, getattr(runner, "corr_magnitudes", None), self.magnitude_mode, self.magnitudes
        )
        # A safe point: train_step already ran backward, step and zero_grad, and
        # torch.autograd.grad populates no parameter .grad anyway.
        runner.probe_sweep = probe.sweep(
            self.backend,
            self.probe_set,
            self.data.dataset,
            self.ops,
            magnitudes,
            checkpoint=_progress(runner, iteration),
            batch_size=self.batch_size,
            probe_seed=self.probe_seed,
            fd_step=self.fd_step,
            num_workers=self.num_workers,
            log_path=self.log_path if is_main_process() else None,
            meta={"iter": iteration, "role": role},
            fade=self.fade,
            fade_at=self.fade_at,
            skip=self.skip,
        )


@HOOKS.register_module()
class ApplyWeightsHook(Hook):
    """Turn the sweep CollectGradientHook just took into R, log it, and hand the host q.

    selector     ``dict(type=<selection name>, **kwargs)``; its ``lam`` is the target
                 strength, ramped by training progress each time. lam=0 (the default)
                 measures and logs R and never calls ``set_weights``: the control arm,
                 bit-identical to an un-probed run.
    lambda_ramp  "linear" (Sensaug's default: lam grows from 0 to the target over
                 training) or "constant"
    mask_within_op, n_min, normalize_per_image, bootstrap, bootstrap_reps,
    bootstrap_seed, fdr_alpha
                 as in Sensaug's analysis hook; see ``stats.analyze``
    """

    priority = "LOW"

    def __init__(
        self,
        selector: Optional[dict] = None,
        lambda_ramp: str = "linear",
        mask_within_op: bool = True,
        n_min: int = 100,
        normalize_per_image: bool = True,
        bootstrap: bool = True,
        bootstrap_reps: int = 1000,
        bootstrap_seed: int = 0,
        fdr_alpha: float = 0.05,
    ):
        selector = dict(selector or {"type": "kl_reweight", "lam": 0.0})
        self.selector_name = selector.pop("type")
        self.lam = float(selector.pop("lam", 0.0))
        self.selector_kwargs = selector
        # How red(a) is scored for the publication rule: the selector's own mode, as
        # Sensaug's --corr-red-mode set both. Selectors without one use Sensaug's default.
        self.mode = selector.get("mode", "squared")
        # Fail at startup on an unknown selector or a bad kwarg, not hours in.
        build_selector(self.selector_name, lam=self.lam, **self.selector_kwargs)
        ramp_lambda(self.lam, 0.0, lambda_ramp)
        self.lambda_ramp = lambda_ramp
        self.mask_within_op = mask_within_op
        self.analysis_kwargs = dict(
            n_min=n_min,
            normalize_per_image=normalize_per_image,
            bootstrap=bootstrap,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
            fdr_alpha=fdr_alpha,
        )

    def before_run(self, runner) -> None:
        self.host = host_of(runner)
        if is_main_process() and not _resuming(runner):
            stats.reset_logs(runner.work_dir)
        # Sensaug split Lever 3 the same way: its hooks published R, and
        # GradCorrValLoop applied it to every pdf the SA loop generated. With an SA
        # loop (loops.RobustValLoop) on an SA host, this hook only publishes, and the
        # loop calls reweight() after installing each round's pdf. Anything else (the
        # uniform host, a stock val loop) has no SA round, and q is set here.
        # runner._val_loop, not the val_loop property, which would build the loop. A
        # training run has built it by now; anything else leaves a config dict here.
        self.sa_driven = bool(getattr(runner._val_loop, "applies_lever3", False)) and hasattr(
            self.host, "set_sa_pdf"
        )
        # The R every SA round re-applies until an emission replaces or clears it
        # (selection.publication: a control or skipped R leaves it in place, a withheld
        # or unusable one clears it, as in Sensaug).
        self.published = None

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        sweep = getattr(runner, "probe_sweep", None)
        iteration = runner.iter + 1
        if sweep is None or sweep.meta.get("iter") != iteration:
            return
        runner.probe_sweep = None
        role = sweep.meta["role"]

        # The pruned ops are set aside even when the probe measured them (probe_pruned),
        # so the R a selector reads doesn't depend on that switch.
        analysis = stats.analyze_sweep(sweep, static_pruned=self.host.pruned(), **self.analysis_kwargs)
        if is_main_process():
            stats.write_analysis(runner.work_dir, analysis)
            print_log(f"[grad-corr] iter {iteration} ({role}): {analysis.summary()}", logger="current")

        # Sensaug's publication rule (selection.publication). Rank 0's decision, so every
        # rank holds the same R and reaches reweight()'s broadcast together: each rank
        # analysed its own sweep, and the alarm can fall either side of 0.9 in float noise.
        decision = [
            publication(analysis, role=role, mode=self.mode, mask_within_op=self.mask_within_op)
        ]
        broadcast_object_list(decision, src=0)
        action, reason = decision[0]
        if role != "active":
            print_log(
                f"[grad-corr] control probe at iter {iteration}: R logged as the baseline, "
                f"not turned into q",
                logger="current",
            )
            return
        if action == KEEP:
            print_log(
                f"[grad-corr] {reason}; the R published last stays in effect",
                logger="current",
                level=30,
            )
            return
        if action == CLEAR:
            if self.sa_driven:
                # Sensaug set runner.corr_redundancy = None here: every SA round trains
                # on its pdf as generated until a usable R is published.
                self.published = None
                message = "every SA round trains on its pdf as generated until a usable R"
            else:
                message = "host left as it was"
            print_log(f"[grad-corr] q withheld, {message}: {reason}", logger="current", level=30)
            return

        self.published = analysis
        if self.sa_driven:
            if is_main_process():
                lam = ramp_lambda(self.lam, _progress(runner, iteration), self.lambda_ramp)
                self._log_record(
                    runner, analysis, lam, applied=False,
                    reason="published; the SA loop applies it to each round's pdf",
                )
                print_log(
                    f"[grad-corr] iter {iteration}: R published; the SA round at this "
                    f"iteration applies it",
                    logger="current",
                )
            return
        self._apply(runner, analysis, iteration, f"iter {iteration}", record=True)

    def reweight(self, runner) -> None:
        """Lever 3 on the pdf the SA loop just installed, at the current lambda: the
        latest published R, re-selected against the host's new base. Sensaug's
        ``GradCorrValLoop._apply_redundancy_reweighting``. With no published R, or a
        selector that declines, the pdf trains as generated (the host's q is cleared,
        not carried over from the last round)."""
        if self.published is None:
            self.host.clear_weights()
            return
        # runner.iter + 1: Sensaug's training_progress, (runner.iter + 1) / max_iters,
        # which GradCorrValLoop called from the val loop, where runner.iter is already
        # the completed count. The lambda is one iteration's ramp past the round's.
        self._apply(
            runner,
            self.published,
            runner.iter + 1,
            f"SA round at iter {runner.iter}",
            record=False,
            clear_on_decline=True,
        )

    def _select(self, runner, analysis, iteration):
        lam = ramp_lambda(self.lam, _progress(runner, iteration), self.lambda_ramp)
        selected = build_selector(self.selector_name, lam=lam, **self.selector_kwargs).select(
            self._redundancy(analysis), dict(zip(self.host.vocabulary(), self.host.base_probs()))
        )
        return selected, lam

    def _redundancy(self, analysis) -> Redundancy:
        return Redundancy(
            analysis.names,
            analysis.primary,
            survives=analysis.survives,
            mask_within_op=self.mask_within_op,
            ci_lo=analysis.ci_lo,
        )

    def _apply(
        self, runner, analysis, iteration: int, when: str, *, record: bool, clear_on_decline: bool = False
    ) -> None:
        selected, lam = self._select(runner, analysis, iteration)
        # Every rank installs rank 0's decision, bit for bit.
        decision = [(selected.applied, selected.reason, selected.q)]
        broadcast_object_list(decision, src=0)
        applied, reason, q = decision[0]

        if applied:
            try:
                self.host.set_weights(q)
            except (RuntimeError, ValueError) as error:
                # Declined, not raised: this runs mid-training on every rank.
                applied, reason = False, f"host refused q: {error}"
        if not applied and clear_on_decline:
            self.host.clear_weights()

        if not is_main_process():
            return
        if record:
            self._log_record(
                runner, analysis, lam, applied=applied, reason=reason, q=q, pruned=selected.pruned,
                covers=selected.covers,
            )
        if applied:
            print_log(
                f"[redundancy] {when}: {self.selector_name} at lambda={lam:.3f} "
                f"(target {self.lam:g}, {self.lambda_ramp} ramp), max/min q spread "
                f"{selected.spread:.1f}x" + (f", pruned {list(selected.pruned)}" if selected.pruned else "")
                + (f", covers {dict(selected.covers)}" if selected.covers else ""),
                logger="current",
            )
        else:
            # Never silently: a q that quietly did nothing looks like a real arm.
            # At a target lam of 0 that is the control arm doing its job.
            print_log(
                f"[redundancy] {when}: q NOT applied ({self.selector_name}): {reason}",
                logger="current",
                level=20 if self.lam == 0 else 30,
            )

    def _log_record(self, runner, analysis, lam, *, applied, reason, q=None, pruned=(), covers=()) -> None:
        score = measured_red(self._redundancy(analysis), self.mode)
        stats.append_jsonl(
            os.path.join(runner.work_dir, stats.REDUNDANCY_LOG),
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
                pruned=list(pruned) if applied else [],
                covers=[list(pair) for pair in covers] if applied else [],
            ),
        )
