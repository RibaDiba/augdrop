"""Sensaug's mmengine loops, the SA loop among them. Glue: the math is ``sa``.

Ported from ``sensaug/loops/sensaug_loop.py`` and ``sensaug/loops/train_loops.py``:

    RobustValLoop             the SA loop (cfg.val_cfg). Every round after warmup it
                              scores each op at each SA-curve level on a subset of val,
                              turns that into the training pdf, and hands it to the
                              host. Every SA_CURVE_CADENCE rounds it first re-searches
                              the curve. Then the clean val as usual
    RobustBaselineValLoop     scores a static curve from a file every round and never
                              touches training
    SubsetTestLoop            a subset of val, plus KID (cfg.test_cfg). The SA curve
                              search scores through it
    SubsetValLoop             a subset of val
    RobustIterBasedTrainLoop  IterBasedTrainLoop plus ``init_sa`` (cfg.train_cfg): one
                              val round before training, on resume or with no warmup

The perturbation goes through the preprocessor's eval slot
(``GpuAugSegDataPreProcessor.set_eval``), as on Sensaug's diff32 path. The pdf goes to
``host.set_sa_pdf``. Lever 3 is ApplyWeightsHook's ``reweight``, called right after;
in Sensaug it was GradCorrValLoop's ``_apply_redundancy_reweighting``. Both arms use
this one loop: without an ApplyWeightsHook (plain sensaug) the pdf trains as
generated.

What Sensaug's train.py set for sensaug (its ``--aug-type ours``) / ``grad_corr``::

    train_cfg = dict(type="RobustIterBasedTrainLoop", max_iters=M,
                     val_interval=M // 20,   # the host's round_interval
                     init_sa=False)          # True on resume or --no-warmup
    val_cfg = dict(type="RobustValLoop", ratio=0.25, warmup_rounds=4,  # the host's
                   descending_MA=..., remove_H=..., uniform=..., weighted_augs=...,
                   pruned_augmentations=[...], hold_none_prob=...)
    test_cfg = dict(type="SubsetTestLoop", ratio=0.25)
    test_evaluator = val_evaluator
    # and model.data_preprocessor = GpuAugSegDataPreProcessor with host=dict(type="sensaug")

Kept from Sensaug, on purpose: ``descending_MA`` (see ``sa.pdf``).

Changed from Sensaug:

- KID sees the op. Sensaug's ``SubsetTestLoop`` fed KID ``data_batch["inputs"]``, the
  dataloader's images, but on the diff32 path (Sensaug ``cc1e403``, 2026-08-16, on) the
  op is applied later, in the preprocessor. So KID compared the clean subset with
  itself, and the objective's KID term was the ratio of two draws of the same subset
  noise. Here KID's fake side is the batch the model was scored on (the preprocessor's
  output, hooked), converted back to the form the real features were taken in: uint8,
  the dataloader's channel order and size.
- KID sees RGB (``kid_rgb``, ``sa.kid_rgb``, default true). Sensaug fed both sides in
  mmcv's loaded BGR, though Inception expects RGB; ``sa.kid_rgb=false`` reproduces that.
  Still Sensaug's, and open: the estimator is torchmetrics' with
  ``subset_size=4``, whose noise a check on stand-in images (not a run) found larger
  than a mild op's whole KID (``blur``, ``color_neg``). Both are open.
- The round counter is set at the first ``run()``, from the iteration count, not in
  ``__init__``. mmengine builds the val loop before ``load_or_resume``, so Sensaug's
  ``runner.iter // val_interval`` there was always 0, and a resumed run restarted its
  warmup.
- The curve is also searched when there is none yet (a resume between curve rounds),
  rather than training on the warmup policy until the next one.
- The eval slot is cleared before every clean evaluation. Sensaug's never was: its
  reset, ``apply_perturbations_dataloader(runner, train=False, perturb_levels={})``,
  passes no ``perturbation_set``, so it takes the CPU branch, rebuilds the val/test
  dataloaders, and leaves ``GpuAugSegDataPreProcessor``'s eval spec set. From the first
  SA round on, the round's "clean" val (the mIoU logged, and the one
  ``save_best="mIoU"`` picks the best checkpoint by) ran with the last op scored still
  applied: the last op of DIFF32 order the curve kept (``color_neg`` unless pruned) at
  its highest curve level. So did the curve search's ``miou_base`` at every curve round
  after the first (rounds 10 and 16 of 20), which moves those curves' levels.
- Not ported: ``perturbation_set`` (the host's vocabulary is the op set; only the GPU
  path exists), ``random_aug`` / ``geometric_only`` / ``photometric_only`` (the
  uniform host), ``corr_skip_pruned_eval`` (Sensaug's train.py rejected it), and the
  CPU-pipeline perturbation paths.
"""

import json
import math
import os
from pprint import pprint
from typing import Dict, List, Optional, Sequence

import torch
from mmengine.device import get_device
from mmengine.dist import is_main_process
from mmengine.logging import print_log
from mmengine.runner import IterBasedTrainLoop, TestLoop, ValLoop
from mmseg.registry import LOOPS

import sa
from hosts.sensaug import SA_CURVE_CADENCE

from .backend import _mean_std, unwrap
from .hooks import ApplyWeightsHook

__all__ = [
    "RobustBaselineValLoop",
    "RobustIterBasedTrainLoop",
    "RobustValLoop",
    "SubsetTestLoop",
    "SubsetValLoop",
]

SA_CURVE_LOG = "sa_curve_log.txt"
PERTURB_EVAL_LOG = "perturb_eval.txt"
CORR_MAGNITUDES_FILE = "corr_magnitudes.json"


def _eval_slot(runner):
    """The preprocessor whose eval slot applies the SA op."""
    dp = unwrap(runner.model).data_preprocessor
    if not hasattr(dp, "set_eval"):
        raise TypeError(
            "the SA loop applies its ops through the preprocessor's eval slot; set "
            "model.data_preprocessor.type='GpuAugSegDataPreProcessor'"
        )
    return dp


@LOOPS.register_module()
class RobustValLoop(ValLoop):
    """The SA loop. One ``run()`` per round (every ``val_interval`` iterations).

    ratio                 the fraction of the val dataloader each (op, level) is
                          scored on (Sensaug: 0.25)
    descending_MA         Sensaug's flag, Sensaug's behaviour: True gives each op's
                          lowest-mIoU level the most mass (``sa.generate_pdf_new``)
    remove_H              drop lighter_H / darker_H (``--no-inv-aug``)
    warmup_rounds         rounds before the first SA pdf; must equal the host's
    uniform               ``sa.generate_uniform_pdf`` instead (``--uniform``)
    weighted_augs         ``sa.generate_pdf_new_weighted_aug`` instead (``--weighted-augs``)
    pruned_augmentations  ops kept out of the curve and every pdf for the whole run (the
                          ``prune:`` block's ops; the builder sets it to the host's)
    hold_none_prob        count those ops in P(none)'s denominator (``--hold-none-prob``)
    num_levels, tolerance the curve search's (Sensaug hard-coded 5 and 0.05)
    sa_curve_path         unused here; RobustBaselineValLoop's static curve
    """

    #: ApplyWeightsHook reads this: with an SA host, Lever 3 waits for reweight().
    applies_lever3 = True

    def __init__(
        self,
        runner,
        dataloader,
        evaluator,
        ratio: float = 0.5,
        sa_curve_path: Optional[str] = None,
        uniform: bool = False,
        descending_MA: bool = False,
        remove_H: bool = False,
        warmup_rounds: int = 4,
        weighted_augs: bool = False,
        pruned_augmentations: Optional[Sequence[str]] = None,
        hold_none_prob: bool = False,
        num_levels: int = 5,
        tolerance: float = 0.05,
        fp16: bool = False,
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        self.ratio = ratio
        self.static_sa_curve_path = sa_curve_path
        self.uniform = uniform
        self.descending_MA = descending_MA
        self.remove_H = remove_H
        self.warmup_rounds = int(warmup_rounds)
        self.weighted_augs = weighted_augs
        self.pruned_augmentations = list(pruned_augmentations or [])
        self.hold_none_prob = hold_none_prob
        self.num_levels = num_levels
        self.tolerance = tolerance

        self.sa_curve: Optional[Dict[str, List[float]]] = None
        self.pdf_dict = None
        self.n_rounds: Optional[int] = None  # set at the first run(); see the module docstring
        self._sa_curve_just_recomputed = False
        self._remove_H_names = list(sa.REMOVE_H_NAMES)

        work_dir = runner.cfg.work_dir
        self.sa_log_path = os.path.join(work_dir, SA_CURVE_LOG)
        self.perturb_metrics_path = os.path.join(work_dir, PERTURB_EVAL_LOG)
        self.corr_magnitudes_path = os.path.join(work_dir, CORR_MAGNITUDES_FILE)

    # --- set up at the first round --------------------------------------------------

    def _start(self) -> None:
        runner = self.runner
        self.dp = _eval_slot(runner)
        host = self.dp.host
        self.host = host if hasattr(host, "set_sa_pdf") else None
        interval = runner.cfg.train_cfg.val_interval
        init_sa = bool(getattr(runner.train_loop, "init_sa", False))
        # Round r runs after (r + 1) * interval iterations, or after r * interval when
        # a pre-training round (init_sa) shifted the grid. Resumed runs set init_sa.
        self.n_rounds = max(runner.iter // interval - (0 if init_sa else 1), 0)

        if self.host is not None:
            mismatched = [
                f"{key}: loop {mine}, host {theirs}"
                for key, mine, theirs in (
                    ("warmup_rounds", self.warmup_rounds, self.host.warmup_rounds),
                    ("round interval", interval, self.host.round_interval),
                )
                if mine != theirs
            ]
            if mismatched:
                raise ValueError(
                    "the SA loop and the host's probe schedule disagree ("
                    + "; ".join(mismatched)
                    + "), so probes would not land on the SA rounds that read them"
                )
            vocabulary = set(self.host.vocabulary())
            unknown = sorted(set(self.pruned_augmentations) - vocabulary)
            if unknown:
                raise ValueError(f"pruned_augmentations {unknown} are not in the host's vocabulary")
            removed = set(self._remove_H_names) if self.remove_H else set()
            # What --hold-none-prob counts: ops the static prune removed that would
            # otherwise have reached the pdf.
            self._n_static_excluded = len(set(self.pruned_augmentations) - removed)

        if is_main_process():
            print(
                f"Starting from round {self.n_rounds}; Uniform sampling: {self.uniform}; "
                f"Descending MA: {self.descending_MA}; Remove color augs: {self.remove_H}; "
                f"Warmup rounds: {self.warmup_rounds}; SA host: {self.host is not None}"
            )

    # --- one round --------------------------------------------------------------------

    def run(self) -> dict:
        """Launch one round."""
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()
        if self.n_rounds is None:
            self._start()

        perturb_metrics = {}
        if self.host is not None and self.n_rounds >= self.warmup_rounds:
            self._sa_curve_just_recomputed = (
                self.n_rounds - self.warmup_rounds
            ) % SA_CURVE_CADENCE == 0 or self.sa_curve is None
            if self._sa_curve_just_recomputed:
                self.update_sa_curve()
            perturb_metrics = self.update_pdf()

        # full clean evaluation last
        self.dp.set_eval_none()
        metrics = self._run_eval(ratio=1.0)
        perturb_metrics.update(metrics)

        if is_main_process():
            print("Evaluation results with perturbations: ")
            pprint(perturb_metrics)
            with open(self.perturb_metrics_path, "a") as f:
                f.write(json.dumps(perturb_metrics) + "\n")

        # we log the clean evaluation officially to monitor progress
        self.runner.call_hook("after_val_epoch", metrics=perturb_metrics)
        self.runner.call_hook("after_val")
        self.n_rounds += 1
        return perturb_metrics

    def update_sa_curve(self) -> None:
        """Re-search the SA curve (Sensaug's ``adaptive_sensitivity_analysis_new``),
        over the host's vocabulary minus the static prune and ``remove_H``."""
        print_log("Running sensitivity analysis...", logger="current")
        excluded = set(self.pruned_augmentations) | (set(self._remove_H_names) if self.remove_H else set())
        ops = [op for op in self.host.vocabulary() if op not in excluded]

        self.dp.set_eval_none()
        print_log("Running baseline performance test", logger="current")
        miou_base = self._test_loop().run()["mIoU"]
        self._verify_effective(ops[0], 1.0)
        try:
            self.sa_curve = sa.sa_curve(
                self._evaluate,
                ops,
                miou_base,
                num_levels=self.num_levels,
                tolerance=self.tolerance,
                log=lambda message: print_log(message, logger="current"),
            )
        finally:
            self.dp.set_eval_none()

        if is_main_process():
            print("New SA curve computed")
            print(self.sa_curve)
            with open(self.sa_log_path, "a") as f:
                f.write(json.dumps(self.sa_curve) + "\n")

    def update_pdf(self) -> dict:
        """Score the curve, build this round's pdf, install it on the host, re-apply
        Lever 3, and publish the probe's magnitudes. Returns the per-op metrics."""
        miou_record, final_metrics = self.test_perturbed_new()
        if self.remove_H:
            sa.remove_H_perturbations(miou_record, self._remove_H_names)

        denom = sa.none_prob_denominator(len(miou_record), self._n_static_excluded, self.hold_none_prob)
        if self.uniform:
            pdf = sa.generate_uniform_pdf(miou_record, denom)
        elif self.weighted_augs:
            pdf = sa.generate_pdf_new_weighted_aug(miou_record, self.descending_MA)
        else:
            pdf = sa.generate_pdf_new(miou_record, self.descending_MA, denom)

        if is_main_process():
            print("Perturbation PDF computed:")
            print(json.dumps({str(k): v for k, v in pdf.items()}, indent=4, sort_keys=True))

        # This is what puts the pdf in front of training: the preprocessor samples
        # from host.pdf(). A q already set is carried over until reweight replaces it.
        self.host.set_sa_pdf(pdf)
        hook = self._weights_hook()
        if hook is not None:
            hook.reweight(self.runner)
        self.pdf_dict = self.host.pdf()
        self.publish_corr_magnitudes()
        return final_metrics

    def test_perturbed_new(self):
        """``({op: {level: mIoU}}, flat per-op metrics)`` for every op and level of the
        current curve, each on ``ratio`` of the val dataloader."""
        miou_record, metrics_record = {}, {}
        try:
            for p_type, levels in self.sa_curve.items():
                miou_record[p_type] = {}
                metrics_record[p_type] = []
                for level in levels:
                    self.dp.set_eval(p_type, level)
                    metrics = self._run_eval(ratio=self.ratio)
                    miou_record[p_type][level] = metrics["mIoU"]
                    metrics_record[p_type].append(metrics)
        finally:
            self.dp.set_eval_none()
        return miou_record, sa.level_metrics(metrics_record)

    def publish_corr_magnitudes(self) -> None:
        """Hand this round's per-op level distributions to the probe
        (``runner.corr_magnitudes``) and append them to ``corr_magnitudes.json``,
        the file a post-hoc probe of this run reads."""
        if not self.pdf_dict:
            return
        snapshot = sa.conditional_levels(self.pdf_dict, op_names=set(self.host.vocabulary()))
        self.runner.corr_magnitudes = snapshot
        if not is_main_process():
            return

        record = {
            "iter": int(self.runner.iter),
            "round": int(self.n_rounds),
            "magnitudes": {
                op: dict(entry, mode=sa.modal_magnitude(entry)) for op, entry in snapshot.items()
            },
        }
        records = []
        if os.path.exists(self.corr_magnitudes_path):
            with open(self.corr_magnitudes_path) as f:
                records = json.load(f)
        records.append(record)
        # Whole-file rewrite through a temp file, so an interrupted write can't take
        # the earlier rounds with it.
        tmp = self.corr_magnitudes_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(records, f, indent=2, allow_nan=False)
        os.replace(tmp, self.corr_magnitudes_path)
        print_log(
            f"[corr-magnitudes] round {self.n_rounds} (iter {self.runner.iter}): published "
            f"{len(snapshot)} ops, modal magnitudes "
            + ", ".join(f"{op}={sa.modal_magnitude(e):.3f}" for op, e in sorted(snapshot.items())),
            logger="current",
        )

    # --- evaluation -------------------------------------------------------------------

    def _evaluate(self, op: str, level: float):
        """Sensaug's ``calculate_miou_kid_new``: ``op`` at ``level`` on the eval slot,
        scored by SubsetTestLoop."""
        self.dp.set_eval(op, level)
        try:
            metrics = self._test_loop().run()
        finally:
            self.dp.set_eval_none()
        miou = metrics["mIoU"]
        kid_mean, kid_std = metrics.get("kid", (math.nan, math.nan))
        kid_mean, kid_std = float(kid_mean), float(kid_std)
        print_log(
            f"{op} {level}: {miou:.3f}, KID mean: {kid_mean:.3f}, KID std: {kid_std:.3f}",
            logger="current",
        )
        return miou, kid_mean

    def _test_loop(self) -> "SubsetTestLoop":
        loop = self.runner.test_loop
        if not isinstance(loop, SubsetTestLoop):
            raise TypeError(
                f"the SA curve search scores through SubsetTestLoop, but test_cfg built a "
                f"{type(loop).__name__}; set test_cfg=dict(type='SubsetTestLoop', ratio=...)"
            )
        return loop

    def _verify_effective(self, op: str, level: float) -> None:
        """At full magnitude an op must change pixels, or the curve would measure the
        clean model for it (Sensaug's ``verify_perturbation_effective``, GPU branch)."""
        probe = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            out, _ = self.dp.ops[op].apply(probe, torch.full((1,), float(level)), None)
        if out.shape == probe.shape and torch.allclose(probe, out):
            raise RuntimeError(
                f"Perturbation {op} at magnitude {level} left its input unchanged. "
                f"Refusing to produce a meaningless SA curve."
            )

    def _weights_hook(self) -> Optional[ApplyWeightsHook]:
        return next((h for h in self.runner.hooks if isinstance(h, ApplyWeightsHook)), None)

    def _run_eval(self, ratio: float) -> dict:
        """``ratio`` of the val dataloader through ``run_iter``, then the evaluator."""
        max_iter = int(len(self.dataloader) * ratio)
        n_samples = 0
        dataloader_iter = iter(self.dataloader)
        idx = 0
        while idx < max_iter:
            data_batch = next(dataloader_iter)
            self.run_iter(idx, data_batch)
            n_samples += len(data_batch["inputs"])
            idx += 1
        return self.evaluator.evaluate(n_samples)


@LOOPS.register_module()
class RobustBaselineValLoop(RobustValLoop):
    """Scores a static SA curve (``sa_curve_path``, ``{op: [levels]}``) every round,
    then the clean val. Never builds a pdf or touches training. The curve's ops must be
    in the host's vocabulary."""

    applies_lever3 = False

    def __init__(self, runner, dataloader, evaluator, ratio=0.5, sa_curve_path=None, fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, ratio=ratio, sa_curve_path=sa_curve_path, fp16=fp16)
        if sa_curve_path is None:
            raise ValueError("RobustBaselineValLoop scores a static curve; give sa_curve_path")
        self.load_sa_curve()

    def load_sa_curve(self) -> None:
        with open(self.static_sa_curve_path) as f:
            self.sa_curve = json.load(f)

    def run(self) -> dict:
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()
        if self.n_rounds is None:
            self.dp = _eval_slot(self.runner)
            self.n_rounds = 0

        _, perturb_metrics = self.test_perturbed_new()
        self.dp.set_eval_none()
        metrics = self._run_eval(ratio=1.0)
        perturb_metrics.update(metrics)

        if is_main_process():
            print("Evaluation results with perturbations: ")
            pprint(perturb_metrics)
            with open(self.perturb_metrics_path, "a") as f:
                f.write(json.dumps(perturb_metrics) + "\n")

        self.runner.call_hook("after_val_epoch", metrics=perturb_metrics)
        self.runner.call_hook("after_val")
        self.n_rounds += 1
        return perturb_metrics


@LOOPS.register_module()
class RobustIterBasedTrainLoop(IterBasedTrainLoop):
    """IterBasedTrainLoop that can run one val round before training (``init_sa``):
    Sensaug set it on resume and with --no-warmup, so an SA pdf exists from the start."""

    def __init__(
        self,
        runner,
        dataloader,
        max_iters: int,
        val_begin: int = 1,
        val_interval: int = 1000,
        init_sa: bool = False,
        dynamic_intervals=None,
    ) -> None:
        super().__init__(runner, dataloader, max_iters, val_begin, val_interval, dynamic_intervals)
        self.init_sa = init_sa

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook("before_train")
        # In iteration-based training the whole run is one big epoch.
        self.runner.call_hook("before_train_epoch")
        if self._iter > 0 and self._iter < self._max_iters:
            print_log(
                f"Advance dataloader {self._iter} steps to skip data that has already been trained",
                logger="current",
                level=30,
            )
            for _ in range(self._iter):
                next(self.dataloader_iterator)

        if self.init_sa:
            print_log("Initializing sensitivity analysis curve", logger="current")
            self.runner.val_loop.run()
        else:
            print_log(
                f"Not initializing sensitivity analysis curve. Training will resume at iteration {self._iter}.",
                logger="current",
            )

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()
            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (
                self.runner.val_loop is not None
                and self._iter >= self.val_begin
                and (self._iter % self.val_interval == 0 or self._iter == self._max_iters)
            ):
                self.runner.val_loop.run()

        self.runner.call_hook("after_train_epoch")
        self.runner.call_hook("after_train")
        return self.runner.model


@LOOPS.register_module()
class SubsetValLoop(ValLoop):
    """Val on the first ``ratio`` of the dataloader."""

    def __init__(self, runner, dataloader, evaluator, ratio: float = 0.5, fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        self.ratio = ratio

    def run(self) -> dict:
        self.runner.call_hook("before_val")
        self.runner.call_hook("before_val_epoch")
        self.runner.model.eval()
        max_iter = int(len(self.dataloader) * self.ratio)
        dataloader_iter = iter(self.dataloader)
        for idx in range(max_iter):
            self.run_iter(idx, next(dataloader_iter))
        metrics = self.evaluator.evaluate(int(max_iter * self.dataloader.batch_size))
        self.runner.call_hook("after_val_epoch", metrics=metrics)
        self.runner.call_hook("after_val")
        return metrics


@LOOPS.register_module()
class SubsetTestLoop(TestLoop):
    """Test on the first ``ratio`` of the dataloader, with KID against the same subset's
    clean images (torchmetrics' KernelInceptionDistance, features cached at build).

    The real side is ``data_batch["inputs"]``, as loaded. The fake side is what the
    model was scored on, the preprocessor's output (so the eval slot's op), put back in
    that form by ``_as_loaded``; see the module docstring."""

    def __init__(
        self, runner, dataloader, evaluator, ratio: float = 0.5, fp16: bool = False, kid_rgb: bool = True
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        self.ratio = ratio
        # KID's Inception (torch-fidelity's weights) expects RGB, but both sides are in the
        # loaded order, mmcv's BGR. kid_rgb flips both to RGB just before KID sees them;
        # False is Sensaug's (BGR into Inception).
        self.kid_rgb = kid_rgb
        self.kid = None
        self.update_kid()

    def _kid_view(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` (uint8, B×3×H×W, the loaded order) as KID should see them."""
        return images[:, [2, 1, 0]] if self.kid_rgb else images

    def update_kid(self) -> None:
        try:
            from torchmetrics.image.kid import KernelInceptionDistance
        except ImportError as error:  # torchmetrics, and torch-fidelity for the weights
            raise ImportError(
                "SubsetTestLoop computes KID with torchmetrics (and torch-fidelity's Inception "
                "weights); install both"
            ) from error

        max_iter = int(len(self.dataloader) * self.ratio)
        self.kid = KernelInceptionDistance(
            subsets=max_iter,
            subset_size=4,
            reset_real_features=False,
            normalize=False,
        ).to(get_device())

        dataloader_iter = iter(self.dataloader)
        for _ in range(max_iter):
            data_batch = next(dataloader_iter)
            self.kid.update(self._kid_view(torch.stack(data_batch["inputs"], dim=0)).to(get_device()), real=True)
        self.kid.reset()  # keeps the real features (reset_real_features=False)

    def run(self) -> dict:
        max_iter = int(len(self.dataloader) * self.ratio)
        self.runner.call_hook("before_test")
        self.runner.call_hook("before_test_epoch")
        self.runner.model.eval()
        self.kid.reset()

        # KID's fake side is the batch the model was scored on: the preprocessor's
        # output, with the eval slot's op applied. data_batch["inputs"] stays clean.
        dp = unwrap(self.runner.model).data_preprocessor
        scored = []
        handle = dp.register_forward_hook(lambda _module, _args, out: scored.append(out["inputs"]))
        try:
            dataloader_iter = iter(self.dataloader)
            for idx in range(max_iter):
                data_batch = next(dataloader_iter)
                scored.clear()
                self.run_iter(idx, data_batch)
                if len(scored) != 1:
                    raise RuntimeError(
                        f"the data preprocessor ran {len(scored)} times on one test batch; "
                        f"KID needs the one batch the model was scored on"
                    )
                fake = _as_loaded(dp, scored[0], data_batch["inputs"])
                self.kid.update(self._kid_view(fake).to(get_device()), real=False)
        finally:
            handle.remove()

        kid_result = self.kid.compute()
        metrics = self.evaluator.evaluate(int(max_iter * self.dataloader.batch_size))
        self.runner.call_hook("after_test_epoch", metrics=metrics)
        self.runner.call_hook("after_test")
        metrics["kid"] = kid_result
        return metrics


def _as_loaded(dp, inputs: torch.Tensor, loaded: Sequence[torch.Tensor]) -> torch.Tensor:
    """The preprocessor's output back in the form the dataloader loaded it, which is the
    form KID's real features were taken in: uint8, the loaded channel order (mmcv's BGR;
    the preprocessor's ``bgr_to_rgb`` is undone) and the loaded size (``stack_batch``'s
    test-time padding, at the bottom and right, is cropped). With a clean eval slot this
    gives back ``loaded`` exactly."""
    mean, std = _mean_std(dp)
    images = (inputs.float() * std + mean).round().clamp(0, 255).to(torch.uint8)
    if getattr(dp, "channel_conversion", False):
        images = images[:, [2, 1, 0]]
    h, w = loaded[0].shape[-2:]
    if images.shape[-2] < h or images.shape[-1] < w:
        raise RuntimeError(
            f"the preprocessor's output is {tuple(images.shape[-2:])}, smaller than the "
            f"loaded images ({h}, {w}); KID can't compare them"
        )
    return images[..., :h, :w].contiguous()
