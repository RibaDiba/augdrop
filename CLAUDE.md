# dropaug

Status: **WIP.** Add build/run commands here as they come into existence.

Tests: `pytest tests/` (the path to `src/` is set in `tests/conftest.py`). There is no
dropaug env yet; the conda env `sensAug` (torch 2.12, torchvision 0.27, kornia 0.8.2)
runs them. `tests/vocab/test_sensaug_parity.py` imports the originals from
`third_party/Sensaug` and skips when it is absent. That is a git submodule
(`RibaDiba/SensAug`, pinned to `1192a07` on `test/hard-prune-test`); run
`git submodule update --init` after cloning.

Numbers: every number a doc or paper reports as a finding must have a row in
`FACTS.md` (measured / inherited / derived, with its source). Add the row when you
cite the number. A measured row cites a `dropaug.io` artifact with its hashes.

## `src/vocab/`: what an augmentation is

Every op is an `OpSpec` (`op.py`) called as `spec.apply(images, magnitude, label)`
and returning `(images, label)`. Images are float `[0,1]` RGB BCHW, and the magnitude
is a scalar or `(B,)`. Geometric ops always move the label.
`registry.build_vocab(config)` returns an ordered `{name: OpSpec}` (the axes of R);
the default is Sensaug's 32 `DIFF32_OPS`, in its order. `audit.audit(vocab)` runs
every behavioural check (identity, per-image independence, gradient, label) and
returns a report.

| family | ops | implementation | magnitude units |
|---|---|---|---|
| `legacy_hsv` | 14 | `diff_ops.py`, ported from Sensaug | Sensaug's: unsigned `[0,1]`, direction in the name |
| `geometric` | 10 | same | same |
| `pil_autoaug` | 8 | same (kornia forms, not torchvision's; see TODO(calibration) in `diff_ops.py`) | same |
| `randaug_default` | 14 | torchvision v2 `RandAugment`'s own per-op dispatcher, on 8-bit images, `grad_mode="none"` | AutoAugment Table 6, which RandAugment inherits (degrees, factors, bits, px at 331×331) |
| `trivialaug_wide` | 14 | torchvision v2 `TrivialAugmentWide`'s dispatcher | TrivialAugment Table 8, wide ranges |
| `autoaugment` | 1 | `Invert`, the one AutoAugment op RandAugment dropped, via torchvision v2 `AutoAugment`'s dispatcher. AutoAugment's other 13 ops are `randaug_default`'s (same Table 6 ranges) | no magnitude |

Every op takes its magnitude at the images' dtype precision (the torchvision ops too),
so a python float and a float32 tensor holding it give identical output.
`tests/probe/test_vocab_ops.py` runs every vocab op and every host's vocabulary
through `collect_grads`, and pins the audit's "strongest magnitude" to the probe's
`strength_to_magnitude(op, 1.0)`.

## `src/hosts/`: one adapter per augmentation method

train.py picks a host with `build_host(name, **kwargs)`. A host is the method's
sampling policy only: pure numpy, no torch, and it does not apply ops or run the
probe. `AugHost` (`base.py`) provides `vocabulary()` (vocab op names, the axes of R
and q), `base_probs()` / `sampling_probs()`, `set_weights(q)`, `should_probe(it)` and
`probe_role(it)` (`"control"` R must not become a q).

| host | vocabulary | notes |
|---|---|---|
| `sensaug` | the 32 DIFF32 ops | the probe fires on Sensaug's round schedule (ported). The SA pdf arrives through `set_sa_pdf()` because the SA loop is not ported; `pdf()` returns it with q applied |
| `randaugment` | `randaug_default.*` minus Identity | torchvision defaults: N=2, M=9 of 31 bins |
| `trivialaugment` | `trivialaug_wide.*` minus Identity | |
| `autoaugment` | `randaug_default.*` ops used by the ImageNet policy, plus `autoaugment.Invert` | q tilts the 25 sub-policies; `sampling_probs()` is what the policy actually draws, not q |
| `uniform` | any; defaults to DIFF32 | Sensaug's `--random-aug` bank |

Rules every host keeps. The no-op draw (Identity, `("none", 0)`) stays out of the
vocabulary, and `set_weights` holds its `none_prob` fixed; AutoAugment is the one
exception, since it has no no-op draw. q replaces the previous q rather than
compounding it, so derive it from `base_probs()`. A non-finite q raises.
`tests/hosts/` pins the names against `vocab`, the torchvision op spaces and policy,
and Sensaug's `round_schedule` and `redundancy.reweight`.

## `src/probe/`: measuring G

Ported from Sensaug's `CollectGradientHook` and `compute_grad_corr.py`, with no
mmseg/mmengine and no knowledge of any op name. It stops at G (ops × probe images);
turning G into R is `src/stats/`. The package docstring
(`probe/__init__.py`) holds the two contracts the probe relies on:

- **ops** are a vocab (`build_vocab`), in G's row order. `grad_mode="autograd"` ops are
  differentiated directly, `"none"` ops by central finite differences
  (`finite_diff.py`), and ops with no `mag_range` get a NaN row labelled `unprobed`.
- **a model adapter** (`ModelAdapter`, not a host) exposes `.model`,
  `to_unit(batch) -> (images01, label)` and `loss(images01, label, batch) -> (B,)`.
  `tasks.mmseg.MMSegBackend` is one (see `src/tasks/`).

| file | what |
|---|---|
| `probe_set.py` | `ProbeSet.build(ids, n, seed)`: a subset chosen from the sorted ids, so it doesn't depend on listing order. Its sha256 `digest` goes on every G |
| `grad_collect.py` | `collect_grads(adapter, batch, ops, magnitudes)` → `{op: (B,)}`. Refuses train-mode modules, checks the label moved for geometric ops, pins each op's RNG by `(probe_seed, name)`. `check=True` (first batch of a sweep) also rejects a non-deterministic `to_unit` and any image coupling |
| `finite_diff.py` | the step is a fraction of each op's `mag_range` width (default 0.05). Quantized ops such as Posterize need a larger per-op `fd_step` |
| `frozen_sweep.py` | `sweep(...)` → one `Sweep` (one G, `.matrix()` → `(G, batch_ids)`), optionally appending `aug_gradient_log.txt`. `sweep_checkpoints` gives one G per checkpoint at identical magnitudes. `Magnitudes.fixed / .relative(strength) / custom draw` |
| `magnitude_sweep.py` | e3: one G per reference. A float is a strength in [0, 1] (`strength_to_magnitude`: identity → far end of `mag_range`); a mapping is absolute per-op magnitudes |

Differences from Sensaug to keep in mind when comparing numbers. The loss is per
image and summed; a scalar batch-mean loss is still accepted and gives the per-image
gradients divided by B. `noise` is seeded per op, not per batch. Only `RuntimeError`
(OOM) counts as a transient batch failure, and 3 leading failures stop the sweep.
Not ported: DDP sharding and the gather, the SA-snapshot magnitude modes
(`corr_magnitudes.resolve_magnitudes`, which plugs in as a `Magnitudes` draw), and
skip-pruned backfill.

## `src/stats/`: G → R

The pure-numpy functions of Sensaug's `grad_sens_analysis.py` and `compute_red` from
its `redundancy.py`, with the bodies unchanged and the names and signatures kept, so
`fdr.py` still calls a batch a "probe" (`probe_ids` is `Sweep.matrix()`'s
`batch_ids`). `tests/stats/` is Sensaug's tests for these functions, plus a parity
test of `compute_red` against the submodule. `analyze.py` is Sensaug's `_emit` chain
and `logs.py` its log writers (neither has tests yet).

| file | what |
|---|---|
| `corr.py` | `correlate` → `(R, dropped)`, `shared_image_factor`, `shared_factor_loadings`, `closure_null` |
| `dedup.py` | new: `find_aliases` / `merge_aliases`. Rows of G equal to an earlier row within `ALIAS_RTOL` (1e-6 of the row's scale) are one op under two names, e.g. `randaug_default.Solarize` / `trivialaug_wide.Solarize` at one magnitude. Proportional, constant and NaN rows are never aliases |
| `shrinkage.py` | new: `ledoit_wolf` (toward I; equals sklearn's intensity) and `schafer_strimmer(target=)` (target D, checked against brute force) → `Shrunk(r, dropped, intensity, target)` |
| `fdr.py` | `cluster_bootstrap_ci` (resamples whole batches; the p-value comes from the bootstrap SE), `fdr_correct(method="bh" \| "by")`, and new `fisher_z_se` / `fisher_z_ci` (the unclustered analytic reference; bootstrap SE / Fisher SE is the design effect) |
| `redundancy.py` | `compute_red` → `RedundancyScore` (red(a), squared / abs / signed), `within_op_pairs`, `summarise` |
| `diagnostics.py` | new: `diagnose(R, names, n_images)` → `Diagnostics`: effective rank, participation ratio, condition number, per-op VIF, Marchenko-Pastur edges and `n_signal` (eigenvalues above them). Reads only R's finite part |
| `analyze.py` | `analyze(G, batch_ids, names)` / `analyze_sweep(sweep)` → `Analysis`: R_raw, R_scalenorm, loadings, bootstrap, `survives` (A×A), `withheld` (below `n_min`, or the >0.9 shared-factor alarm), `aliases`, `diagnostics`, and with `shrinkage=` `R_shrunk` (which then becomes `primary`) |
| `logs.py` | `corr_matrix_log.json`, `corr_bootstrap_log.txt`, `corr_redundancy_log.txt` in Sensaug's record shapes (`stale` / `static_pruned` written empty; `unprobed`, `aliases`, `R_shrunk`, `shrinkage`, `diagnostics` and `fdr_method` added) |

Not ported: `merge_rank_buffers` (no DDP), `stack_probe_buffer` (`Sweep.matrix()` does
this) and the TensorBoard panels. The functions themselves keep a Sensaug issue: a NaN
row in G (an `unprobed` op, e.g. `autoaugment.Invert`) makes `shared_image_factor` NaN
in every column. `analyze` avoids it by setting all-NaN rows aside before anything is
computed and listing them in `unprobed`, not `dropped`; on a G without NaN rows it
equals Sensaug's chain. Exact aliases are set aside the same way, before the shared
factor, and listed in `aliases` (the DIFF32 ops have none). A hard selector sees an
alias as exempt, so map it through `aliases` when its fate should follow its
representative's. `dedup`, `shrinkage`, `diagnostics`, BY and Fisher z have no tests
yet; they were checked by hand (see the rows above).

## `src/selection/`: R → q

Named `selection`, not `select`: `select` is a stdlib module (pytest and
`subprocess` import it) and `src/` is first on `sys.path`. Every selector is
`select(R, rel, budget) -> Selection`. `R` is a `Redundancy` (names, the matrix,
the FDR `survives` mask scattered back to A×A, `mask_within_op`). `rel` is
`dict(zip(host.vocabulary(), host.base_probs()))`, which serves as both the prior
and mRMR's relevance, as the SA pdf did in Sensaug. `Selection.q` is aligned to
rel. When `applied` is False the selector declined (the reason is in `reason`) and
the host should be left alone. The rules are in the `base.py` docstring: q is over
ops only, the no-op mass stays with the host, decline rather than degrade, only
`hard` selectors zero an op, and an op with no usable row in R is exempt and kept.

| file | what |
|---|---|
| `base.py` | `Redundancy`, `Selection`, `Selector`, `HardSelector` (the shared pool: prior > 0 and a finite row in `pairwise(R)`, ranked in name order; exempt ops kept outside the budget), `pairwise` (Sensaug's `_pairwise_redundancy`), `budget_from_lambda` (`ceil(A/(1+λ))`), `ramp_lambda` |
| `kl_reweight.py` | soft, Sensaug's `redundancy.reweight` over ops. `KLReweight(lam=...)`, where lam is required |
| `mrmr.py` | Sensaug's `_downweight_mrmr` ranking, unchanged |
| `qpfs.py` | new: QPFS (Rodriguez-Lujan et al. 2010), solved by FISTA on the simplex, keeping the top-k by x |
| `dpp.py` | new: greedy MAP of `L = diag(rel) S diag(rel)` (Chen et al. 2018) |
| `random_k.py` | Sensaug's `null` draw (`draw_random_prune`), over the same pool as the ranked selectors. `null_band` runs it at several seeds |
| `relevance_only.py` | new: top-k by prior, ignoring R. Declines on a flat prior |
| `decide.py` | `decide(analysis, rel, selector, role=...)`: the training-loop policy in one place. A control probe or a `withheld` Analysis declines, and otherwise the selector runs on `redundancy_of(analysis)` (primary R + FDR survivors). Used by `tasks/cls/callbacks.py`; the mmseg `ApplyWeightsHook` predates it and inlines the same steps. Also `publication(analysis, role=, mode=, mask_within_op=)` → `PUBLISH` / `KEEP` / `CLEAR`: Sensaug's `prune_augmentations` rule for the R an SA loop holds (control, skipped or all-NaN R keeps the last one; the shared-factor alarm or an unusable `compute_red` clears it) |

A hard selector takes `budget` (measured ops kept) or is built with `lam`, which
derives the budget the way Sensaug did. Checked by hand against the submodule, with
no tests yet: `mrmr` → `host.pdf()` equals `_downweight_mrmr`, and `kl_reweight`
equals `reweight` to 1e-12. One deliberate difference: `measured_red` leaves an
all-NaN row out of red(a). `compute_red` scores such a row 0, which standardizes to
"least redundant", so the tilt would up-weight every unprobed op.

## `src/sa/`: SensAug's sensitivity analysis

The math of Sensaug's SA loop, with no training framework: every round the model is
scored under each op at a few levels, and those scores become the training pdf over
`(op, level)`. The bodies are Sensaug's, unchanged. Only the mmengine runner calls are
replaced, by an `evaluate(op, level) -> (miou, kid)` callable. The loop that drives it
is glue: `tasks/mmseg/loops.py`. No tests. The parity with the submodule was checked by
hand: the pdf generators were bit-identical in 72 cases, `sa_curve` gave the same levels
from the same sequence of evaluations, and `resolve_magnitudes` matched in all four modes.

| file | what |
|---|---|
| `curve.py` | `sa_curve(evaluate, ops, miou_base, num_levels=5, tolerance=0.05)`: Sensaug's `adaptive_sensitivity_analysis_new`. Per op, a PCHIP fit of (mIoU drop + 2·level − KID) is refined until its crossings of `linspace(0, 2, 6)[1:-1]` settle, giving 4 levels. `objective_function`, `pchip_interpolator` |
| `pdf.py` | `generate_pdf_new(miou_record, descending_MA, denom)`, `generate_pdf_new_weighted_aug`, `generate_uniform_pdf`, `none_prob_denominator` (`hold_none_prob`), `remove_H_perturbations`, `level_metrics` |
| `magnitudes.py` | Sensaug's `corr_magnitudes` (`conditional_levels`, `modal_magnitude`, `resolve_magnitudes`; modes `mode` / `sampled_shared` / `sampled_independent` / `fixed`) plus `probe_magnitudes`, which wraps it as a `probe.Magnitudes` |

`descending_MA` keeps Sensaug's behaviour: True gives each op's **lowest-mIoU** level
the most mass (`reverse = not descending_MA`, with betabinom(0.75, 1) decreasing in
rank). Sensaug's help text describes it the other way round. P(none) is not 1/(A+1),
because the betabinom is truncated (FACTS D6, D7).

## `src/dropaug/eval/`: scoring trained models

This package replaces Sensaug's `test_robust.py`, `scripts/eval_all_perturbations.py`
and the per-op table of `scripts/eval_all_augmentations.py`. Like the probe, it knows
no training framework. Its adapter is the probe's `to_unit`, plus
`predict(images01, batch)`, which returns scores (`(B, K)`, or `(B, K, H, W)` at the
label's resolution), and an optional `target(label, batch)` (classification returns the
class ids). It imports as `dropaug.eval`, while `vocab`/`probe`/`hosts` are still
top-level.

| file | what |
|---|---|
| `robustness.py` | A `Condition` is clean, one op at one level, or one corruption at one severity. `suite(vocab)` is test_robust's set: clean, every op at strengths 0.2 to 1.0, 7 ImageNet-C corruptions, and combination. `run(...)` appends one record per condition to a JSONL file and resumes. One file holds one checkpoint, and it refuses to reuse a condition measured differently. Also `SegMetric` (mmseg's mIoU/aAcc/mAcc), `ClsMetric`, `summarize`, `write_csv` |
| `corruptions.py` | Pre-rendered `ImageNetC`, `CifarC`, `Cityscapes`/`CityscapesC` readers; on-the-fly `corrupt` (the `imagecorruptions` package), Sensaug's `combination` (cv2), and `mce` with AlexNet's baseline |
| `compare.py` | `diff_table(control_runs, pruned_runs, pruned_ops=...)` → `condition_table` → `group_table` (clean / kept / pruned / unseen_op / corruption); `sensaug_prune_history` reads mRMR prune lines from Sensaug logs |

Strength levels go through `probe.strength_to_magnitude`, and every record carries the
op-unit magnitude. `both_directions=True` also evaluates the mirrored side of signed ops
(`randaug_default.Rotate`). Not ported: the adaptive SA level search (SA loop), the
mmseg glue (belongs in an mmseg adapter), and DDP. `imagecorruptions` and `cv2` are
optional and not in `sensAug`. imagecorruptions' `fog`, `glass_blur` and `gaussian_blur`
don't run on numpy≥2 / scikit-image≥0.19.

## `src/dropaug/io/`: artifacts

dropaug's own saved G, R, q and eval results. Every file has a header with `format`,
`schema_version`, `kind`, `names` (the axis order) and `vocab_hash`, and G and R also
carry `probe_set_hash`. It is `dropaug.io`, never a top-level `io`, because that would
shadow the stdlib (the same reason `select` became `selection`). Sensaug-format logs
stay in `stats.logs`. No tests yet; a round trip and every refusal were checked by
hand.

| file | what |
|---|---|
| `schema.py` | `SCHEMA_VERSION` (1), the keys each kind requires, `validate`. Any other version is refused, since there are no migrations. Bump the version when a field's meaning changes |
| `hashing.py` | `vocab_hash` (sha256 of each op's name, family, kind, source, mag_range, identity_at, grad_mode, warps_label, in order; not `fn`), `probe_set_hash` (= `ProbeSet.digest`), `file_hash` |
| `writers.py` | `write_G` (.npz: G, magnitudes, batch_ids, header), `write_R` / `write_q` (with `derived_from=` an R file) / `write_eval` (.json). Atomic. Refuses a vocab whose names or order differ from the object's |
| `readers.py` | `read_G` → `probe.Sweep`, `read_R` → `stats.Analysis` (diagnostics recomputed), `read_q` → `Selection`, `read_eval` → records. `vocab=` / `probe_set=` / `derived_from=` / `checkpoint=` must match or `MismatchError`; `check_registry` (default on) refuses a file whose ops the registry now defines differently. `sweeps_from_gradient_log` turns `aug_gradient_log.txt` back into Sweeps |

## `src/tasks/`: one backend per training framework

A `TaskBackend` (`base.py`) is a `probe.ModelAdapter` that can also `build_model()`,
`load_ckpt(path)` and name its clean probe data (`probe_data()` → `ProbeData(dataset,
ids)`; the probe builds the loader itself, in ProbeSet order). With `predict` it is
also `dropaug.eval`'s adapter. `build_backend(name)` imports a backend lazily, so
`import tasks` needs no framework. Everything next to a backend is glue: it gates and
calls `probe` / `stats` / `selection` / `hosts`, and computes nothing itself. No tests
yet. The mmseg pieces were checked against stand-ins for mmseg, never against mmseg
itself. The cls pieces were checked end to end on a torchvision resnet18 and a toy
ImageFolder, but never with timm (not in `sensAug`).

`tasks/mmseg/` (needs mmseg + mmengine; load it with
`custom_imports = dict(imports=["tasks.mmseg.register"])` and `src/` on `PYTHONPATH`):

| file | what |
|---|---|
| `backend.py` | `MMSegBackend(cfg, model=None, split="val")`, `.from_runner(runner)`. `to_unit` runs the model's preprocessor with `training=True` (mmseg pads the gt only there) with the train aug suspended, then de-normalizes. `loss` is `model.loss` per image, summed over its `loss*` keys, with the op's label swapped in and restored. `predict` is the segmentor's `inference` resized to the label. Refuses a random val pipeline, BGR input, and labels of mixed native sizes (use batch size 1). `load_ckpt` scopes the torch ≥2.6 `weights_only=False` patch to the one load |
| `hooks.py` | `CollectGradientHook` (NORMAL): on `host.probe_role(runner.iter + 1)`, `probe.sweep` → `runner.probe_sweep`; saves/verifies `probe_set.json`. Its `magnitude_mode` (default `"mode"`, Sensaug's) takes each op's magnitude from the SA loop's last snapshot (`runner.corr_magnitudes`), falling back to `magnitude`. `ApplyWeightsHook` (LOW): `stats.analyze_sweep` → logs → `build_selector(type, lam=ramp_lambda(...))` → `host.set_weights(q)`. With `RobustValLoop` on an SA host, it only publishes R (`selection.publication`, rank 0's call broadcast), and the loop calls `reweight()` on every round's fresh pdf at Sensaug's λ, `ramp(target, (iter + 1) / max_iters)` (`GradCorrValLoop`). With no R held, or a declined selection, `reweight()` calls `host.clear_weights()`, so the round trains on its pdf as generated, as in Sensaug; the last q is never carried over. Without an SA loop, control probes, lam=0, a withheld R and a declined selection leave the host alone. Every rank sweeps; rank 0's q is broadcast. Checked (a scratchpad harness, no test yet) against Sensaug's real `_downweight_mrmr` / `reweight` / `compute_red` round by round (mRMR and soft at λ 0, 0.25, 1; withheld, skipped and FDR-empty emissions; both pdf generators): equal to 1e-12, same ops pruned |
| `transforms.py` | `GpuAugSegDataPreProcessor(host=dict(type=...), augment=True)`: Sensaug's `GpuAugSegDataPreProcessor`, batched on GPU after collation, same arithmetic as the backend. It owns the host (`.host`) the hooks read. Sampling is Sensaug's: the SensAug pdf with q (+N(0, 0.1) level jitter), else `sampling_probs()` / `none_prob` at a uniform level. DIFF32 ops with the `sensaug` / `uniform` hosts only; other hosts raise unless `augment=False`. The eval slot (`set_eval(op, level)` / `set_eval_none()`, Sensaug's `set_eval_spec`) applies one op to every image on the eval path, for the SA loop |
| `loops.py` | Sensaug's loops. `RobustValLoop` (val_cfg) is the SA loop: after `warmup_rounds`, every round scores each op at each curve level on `ratio` of val through the eval slot, builds the pdf (`sa.generate_pdf_new` with `descending_MA`, or `uniform` / `weighted_augs`) and calls `host.set_sa_pdf`, then `ApplyWeightsHook.reweight`, then publishes `runner.corr_magnitudes`. It re-searches the curve (`sa.sa_curve`, via `SubsetTestLoop`) every `SA_CURVE_CADENCE` = 6 rounds. Writes `sa_curve_log.txt`, `perturb_eval.txt`, `corr_magnitudes.json`. Refuses `warmup_rounds` / val_interval that differ from the host's. Also `RobustBaselineValLoop`, `SubsetTestLoop` (test_cfg, with KID), `SubsetValLoop`, `RobustIterBasedTrainLoop` (train_cfg, `init_sa`). The module docstring has the config Sensaug's train.py set |
| `register.py` | imports `hooks`, `loops` and `transforms` so the registries see them |
| `config.py` | `build(cfg, work_dir)`: the mmengine config from the composed one (Sensaug's `build_config`). The task's .py is the model and optimizer; the dataset yaml gives the pipelines and the crop (also `data_preprocessor.size`, which pads training and probe batches, as every Sensaug config did). Sensaug's per-(backbone, dataset) file differences come in through `configs/recipe/<task>_<dataset>.yaml`, which `configs/train.yaml` selects automatically (`optional recipe: ${task}_${dataset}`). There is one for every pair Sensaug could train (7 backbones × cityscapes / ade20k / pascal_voc12 / loveda / potsdam / synapse / a2i2haze, 5 on acdc, vitsam on cityscapes: 55), and each was checked (a scratchpad diff, no test) against Sensaug's resolved config: model, optimizer, schedule, max_iters, dataloaders, auto_scale_lr, pad size. They differ only in num_classes. Sensaug's file pick changes the model on some pairs: pspnet and deeplabv3plus become R50 on ade20k (pspnet on pascal_voc12 too), and vit becomes DeiT-B/16 on every dataset except cityscapes. On fallback pairs (no `*<dataset>*` file) it trains 320k iterations, and convnext and deeplabv3plus there pad 512×512 crops to 512×1024. Sensaug's deeplabv3plus on pascal_voc12 kept 19 classes against 21 labels, so it cannot have trained. Sensaug could not build swin/mae on acdc or any backbone on idd (no `_base_/datasets/acdc.py` / `idd.py` for its fallback), so those have no recipe; idd stays eval-only. Two dataset properties travel with the dataset yaml rather than a recipe: synapse's `mmseg.val_evaluator` (mIoU + mDice) and ACDC's `sa_ratio: 1.0` (Sensaug's SA `eval_ratio`, read by `configs/sa/default.yaml`). Sensaug's LoveDA `cfg.optimizer.lr *= 0.1` is not ported: it never reached `optim_wrapper`, so Sensaug's LoveDA runs trained at the full lr |

`loops.py` keeps one Sensaug behaviour on purpose: `descending_MA` is unchanged. It
fixes three Sensaug bugs. KID: Sensaug's read `data_batch["inputs"]`, but the eval slot
applies the op later, in the preprocessor, so from Sensaug `cc1e403` (2026-08-16) on,
every diff32 run's KID compared the clean subset with itself and the curve's KID term
was noise. `SubsetTestLoop` now hooks the preprocessor's output (what the model was
scored on) and hands KID that, back in the loaded form (uint8, BGR, the loaded size).
Still Sensaug's, and open: KID sees mmcv's BGR (Inception expects RGB), and its
`subset_size=4` estimator, whose noise a stand-in check found larger than a mild op's
whole KID. The resume: Sensaug's round counter was set before
`load_or_resume`, so it was always 0. And the eval slot: Sensaug never cleared it
before a clean evaluation (its reset took the CPU branch), so from the first SA round
its "clean" val mIoU, the `save_best` checkpoint and the curve's `miou_base` at rounds
10 and 16 ran with the last op scored (`color_neg` at its top level) still applied.
Checked on stand-ins for mmengine/mmseg/torchmetrics only, not on mmseg itself.

Not ported: `pruned_ops` (exclude an op with the loop's `pruned_augmentations`, the
host's vocabulary or a zero in q). So `sa.pruned_augmentations` keeps an op out of the
curve and every pdf, as Sensaug's `--pruned-augmentations` did, but not out of the
warmup policy or the probe (Sensaug's gave it an all-NaN row in R). The
host's q is not saved in checkpoints, so a resumed run trains on the base distribution
until the next active probe (as in Sensaug). The torchvision geometric ops refuse a
label whose size differs from the image's, which mmseg val pipelines produce on
datasets whose val resize isn't the identity (not Cityscapes).

`tasks/cls/` (torch + torchvision; timm only to build a timm model):

| file | what |
|---|---|
| `backend.py` | `ClsBackend(model, data=, mean=, std=, label_smoothing=0)`. `model` is an nn.Module or `dict(library="timm" \| "torchvision", name=, **kwargs)`. The batch is `dropaug.eval`'s item collated (`image` in [0, 1], `target`, `id`); `to_unit` returns label None and refuses images outside [0, 1]; `loss` is per-image CE; mean/std default to timm's `pretrained_cfg`, else ImageNet's. `load_ckpt` takes `state_dict` / `model` / `state_dict_ema` or a bare state dict, strips `module.`, and loads strictly. `ImageFolderItems` (resize the short side + center crop) and `CifarItems` (never downloads) give ids that match `ImageNetC` / `CifarC` |
| `callbacks.py` | `CollectGradientCallback(backend, host, ...)` / `ApplyWeightsCallback(host, selector=..., **analyze kwargs)`: the mmseg hooks for a hand-written loop, `before_run(work_dir, max_iters, resume=)` then `apply.after_train_iter(it, collect.after_train_iter(it))`. Same files, same DDP handling (rank 0 writes and broadcasts q). The q policy is `selection.decide`. Applying the host's draws to training images is the loop's job; there is no cls train-time augmentation |

## `slurm/`: jobs on UMD Nexus (gamma)

Ported from Sensaug's `job_scripts/` (Della is no longer used). The checkout is assumed
to be at `/gammascratch/amodak/dropaug` (`REPO_DIR`). The `#SBATCH --output` lines
hard-code `slurm/logs/` under it, and that directory must exist before you submit.
Every script reads its knobs from env vars, which are documented in each script's header.
The entry points they call (`train.py`, `scripts/probe.py`, `scripts/eval.py`) **don't
exist yet**; each header lists the flags that entry point must accept.

| file | what |
|---|---|
| `train.sbatch` | `[host] [dataset] [selector] [seed]`, 4×A6000, torchrun. From `train_nexus_gamma.sbatch`. Selector `none` is the λ=0 control. The run dir is `experiments/<host>_<selector>-l<λ>_<dataset>_<backbone>_s<seed>`, and the script refuses to resume an existing run unless `ALLOW_RESUME=1`. With host `sensaug` it passes `--descending-MA --no-inv-aug`, as Sensaug's sbatch did on every run (`DESCENDING_MA=0` / `INV_AUG=1` turn them off). Afterwards it queues `eval.sbatch` as a separate 1-GPU job |
| `probe.sbatch` | `<exp_dir>`: a post-hoc probe of one checkpoint (`CKPT=best\|last\|path`) into `<exp_dir>/probe/<ckpt>[_<PROBE_TAG>]`. It refuses a directory that already holds a gradient log, because two Gs in one file would pool |
| `eval.sbatch` | `<exp_dir>`: `dropaug.eval` on one checkpoint → `<exp_dir>/eval/<ckpt>.jsonl`. Resumes on resubmit |
| `matrix.sbatch` | array over `HOSTS` × `DATASETS` × `SELECTORS` × `SEEDS`, one `train.sbatch` per task. `bash slurm/matrix.sbatch` lists the grid; `--submit` submits it |
| `common.sh` | sourced by all four: log grouping into `slurm/logs/<job-name>/`, conda/module setup, checkpoint resolution |

## What this repo is for

`../Sensaug` (GitHub `RibaDiba/SensAug`) built a **gradient-based sensitivity probe**
for data augmentations. It measures `d loss / d magnitude` per augmentation op per
image, then correlates the ops across images to get a redundancy matrix **R**. There,
the probe only works as one arm (`--aug-type=grad_corr`) of a single
MMSeg/MMEngine training pipeline with 32 fixed ops.

**dropaug's goal is to make that same probe easy to run on a general scale across
methods.** The probe should not be tied to SensAug's training loop, its SA pipeline
or its 32-op vocabulary. Since 2026-09-30 that includes recreating Sensaug itself,
SA loop and all, as a cleaner repo whose pieces are tools for other methods: each
method is a host, plus its own val loop where it needs one (as SensAug has
`RobustValLoop`). Port Sensaug's behaviour as it is, and flag its bugs rather than
silently fixing them. Which methods, models and datasets are in scope is still open.
Ask before committing to an architecture.

Read `../Sensaug/CLAUDE.md` for the full background. It is long, and parts of it are
stale (see `../Sensaug/RESULTS_SUMMARY.md`), so when the doc and the code disagree,
trust the code.

## Where the probe lives in Sensaug

Port from these files. Don't reinvent them.

| Sensaug file | What to take from it |
|---|---|
| `sensaug/hooks/grad_hook.py` | `CollectGradientHook`. `_sweep` runs the frozen-frame pass over the whole clean val set, `_probe_batch` sets up one shared clean batch for all ops, and `_grad_for_op` computes the per-image `torch.autograd.grad(loss, delta)`. The module docstring explains why the probe is designed this way. |
| `sensaug/hooks/grad_sens_analysis.py` | Turns the sweep into R: `stack_probe_buffer`, `correlate`, `shared_image_factor` / `shared_factor_loadings`, `closure_null`, `cluster_bootstrap_ci`, `fdr_correct`, `merge_rank_buffers`, and `_emit`, which assembles them and writes the logs. The functions are pure numpy; the class is mmengine glue. |
| `sensaug/corr_magnitudes.py` | Chooses the probe magnitude for each op (`resolve_magnitudes`, modes `mode` / `sampled_shared` / `sampled_independent` / `fixed`). Pure numpy. |
| `sensaug/redundancy.py` | `compute_red`: collapses R into a per-op redundancy score `red(a)`. Pure numpy. |
| `sensaug/hooks/corr_viz.py` | Heatmap rendering (`render_matrix`) and `markdown_report` for R. |
| `sensaug/dataset/differentiable_augmentations.py` | The 14 base autograd ops (RGB/HSV lighter/darker, `blur`, `noise`). The input contract is `[0,1]` RGB and the magnitude may be a per-image tensor. |
| `sensaug/dataset/differentiable_augmentations_aa.py` | The 18 AutoAugment-family ops, `DIFF32_OPS`, `GEOMETRIC_OP_KEYS`, and the label-safe pair `geometric_affine_matrix` / `warp_image_and_label` (+ `_matrix_in_frame`). |
| `scripts/compute_grad_corr.py` | **The closest template for dropaug.** It runs the probe post hoc on a trained checkpoint, with no training loop. It drives the hooks through `before_run` → `_sweep` → `_emit` on a `_PosthocRunner` proxy. |
| `scripts/calibrate_kid_magnitudes.py` | Picks a per-op magnitude by matching KID, for when there's no SA distribution to draw magnitudes from. |
| `sensaug/dataset/utils/label_safe_transforms.py` | Label-safe torchvision `AutoAugment` / `RandAugment` / `TrivialAugmentWide`. These are the non-differentiable reference methods. |
| `tests/test_grad_hook.py`, `tests/test_grad_sens_analysis.py`, `tests/test_corr_magnitudes.py`, `tests/test_redundancy.py` | Tests that pin the invariants below, including per-image independence. Port them along with the code. |

Past the probe, these are ported too: the SA loop (`sensaug/loops/sensaug_loop.py` +
`sensitivity_analysis.py` → `src/sa/` + `tasks/mmseg/loops.py`), the round schedule
(`round_schedule.py` → `hosts/sensaug.py`), Lever 3 and the mRMR arm
(`grad_corr_loop.py` → `selection/` + `ApplyWeightsHook`), the GPU training aug
(`gpu_augment.py` → `tasks/mmseg/transforms.py`) and the Nexus job scripts
(→ `slurm/`). Not yet: `train.py`'s config assembly (`build_config`), and `test.py`'s
domain-shift sets.

## How the probe works (the part that must generalize)

1. Freeze one model state and set `model.eval()`. Save the torch/CUDA RNG state and
   restore it afterwards.
2. For each batch of **clean** val images (training-time augmentation suspended), convert
   to `[0,1]` RGB. For every op, create a **length-B** magnitude tensor `delta`
   (`requires_grad=True`), apply the op, re-normalize, run `model.loss(...)`, and take
   `torch.autograd.grad(loss, delta)`. This gives one sensitivity per image per op.
3. Stack the results into a `(A ops × N images)` matrix. Divide out the per-image
   shared factor (mean |grad| over ops), then take the Pearson correlation across
   images. The result, R, answers "do these two ops hurt the *same* images?"
4. Attach per-cell cluster-bootstrap CIs (clusters = batches) and apply BH-FDR over the
   off-diagonal cells, testing against the closure null `-1/(A-1)`, not against 0.

## Invariants carried over from Sensaug (each one was a real bug)

- **eval() mode during the probe.** Train-mode BatchNorm couples the images in a batch
  through the batch statistics, and then `d loss / d delta[i]` absorbs the other
  images' deltas. No batch-level augmentation (Mixup/CutMix) in the preprocessor, for
  the same reason.
- **The per-image delta is a `(B,)` tensor.** A scalar delta gives one number per batch
  and cannot produce R. An op that collapses delta internally (e.g. `delta.mean()`)
  gives a gradient with the right shape but the wrong values. Only the per-image
  independence test catches that.
- **One frozen frame per R.** Sweep the whole val set on one model state. Never pool
  sweeps from different checkpoints into one R, because pooling mixes convergence drift
  (Simpson's paradox).
- **Geometric ops must warp the label too**, using `geometric_affine_matrix` +
  `warp_image_and_label` instead of the raw op. The matrix is expressed in the label's
  own pixel frame, because mmseg val pipelines resize the image but load the label at
  its native size. Without this, geometric rows measure misalignment instead of
  sensitivity. This contaminated Sensaug's R and mRMR choices until it was fixed.
- **Magnitudes are not commensurable across ops.** At a fixed 0.5, `blur` moves a pixel
  ≤78/255 and `noise` ≤251/255. Record the magnitude source and mode with every R;
  matrices taken at different magnitudes can't be compared. Magnitudes also don't
  transfer between different implementations of the "same" op (e.g. cv2 vs torch
  `blur`).
- **Fix the probe seed per batch** so `noise` draws the same sample every time. Probe
  randomness must never consume from the training RNG streams.
- **Missing data is NaN, never 0.** Keep zero-variance rows dropped as NaN, keep the
  matrix shape fixed, and keep `dropped` / `stale` / `static_pruned` as separate
  labels. A 0 reads as a real gradient.
- **Health checks log; they don't silently degrade.** Skip R below `n_min` images
  (default 100, SE ≈ 0.10). Warn when every op loads >0.9 on the shared image factor,
  because then R is ranking hard images, not redundant ops. A structural probe error
  (bad shape, non-finite gradient) raises `ProbeError` instead of being skipped.
- **Under DDP, gather every rank's buffers before correlating**, and keep batches as
  separate entries so the bootstrap clusters survive. Every rank must reach the same
  R.
- Observed effect sizes in Sensaug were small: mean |r| ≈ 0.11–0.22. Don't claim more
  than that supports.

## Sensaug environment (reference only; dropaug's is not set up)

- Conda env `sensaug`: Python 3.10, PyTorch ≥2.0, mmcv 2.0–2.1, mmengine, mmsegmentation,
  numpy <2.0, scipy, statsmodels, kornia, torch-fidelity. See `../Sensaug/requirements.txt`.
- Clusters: UMD Nexus (`/gammascratch/amodak/SensAug`, the one in active use) and
  Princeton Della (compute nodes have no internet, so pretrained checkpoints must be
  pre-staged).
- torch ≥2.6 defaults to `torch.load(weights_only=True)`, which can't unpickle mmengine
  checkpoints. `compute_grad_corr.py` patches `torch.load` to handle this.
- Sensaug tests: `cd ../Sensaug && pytest tests/`. Some are marked `requires_mmseg`.
- Conventions: paths go in config files, never in code; PascalCase classes;
  snake_case functions.

## Output format to stay compatible with

Sensaug's probe writes these logs to the work dir. Keeping the same shapes lets offline
tools (`scripts/calibrate_lambda.py`, `corr_viz`) read dropaug output.

- `aug_gradient_log.txt`: JSONL, one record per batch, with the per-image grads and
  magnitudes for every op. This is enough to recompute R offline.
- `corr_matrix_log.json`: a JSON array, one record per emission. Fields: `R_raw`,
  `R_scalenorm`, `names`, `n_images`, `n_probes`, `shared_factor_loadings`,
  `magnitude_source` / `magnitude_mode` / `magnitudes`, `dropped` / `stale` /
  `static_pruned`, `role`.
- `corr_bootstrap_log.txt`: JSONL with per-cell CIs and BH-FDR q-values.
- `corr_redundancy_log.txt`: JSONL with per-op `red(a)`.
