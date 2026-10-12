# Changelog

## 2026-10-07: first "ours" test run on Nexus

Goal: four full training runs of Sensaug's own method ("ours" = `--aug-type ours`:
host `sensaug`, the SA loop, selector `none`, so R is probed and logged but never
changes q) on segformer × {cityscapes, ade20k, pascal_voc12, loveda}, seed 0. Job names
are `ours-test-<dataset>-dropaug-train`. A short smoke test of the same four runs
(`ours-smoke-<dataset>`) ran first.

### Changes

- **Checkout path.** The slurm scripts and `configs/cluster/nexus.yaml` assumed
  `/gammascratch/amodak/dropaug`, which doesn't exist. The repo lives at
  `/fs/gamma-projects/pruneaug`, so that path is now `REPO_DIR`'s default, the
  `#SBATCH --output/--error` directory, `common.sh`'s `LOG_BASE` and the cluster's
  `work_root`. (`/gammascratch/amodak` also has only ~12 GB free, not enough for the
  checkpoints.) `data_root` still points to Sensaug's prepared datasets at
  `/gammascratch/amodak/SensAug/data`. Files: `slurm/{train,eval,probe,matrix}.sbatch`,
  `slurm/common.sh`, `configs/cluster/nexus.yaml`, `CLAUDE.md`.
- **`train.sbatch` default selector.** The hotfix `a1473a7` set it to `sensaug`, which
  is not a `configs/selector` option, so `train.py --dry-run` would have rejected every
  run that relied on the default. It is now `none` (the λ=0 control), which is what
  "ours" means with the sensaug host.
- **`train.sbatch` preflight/launch mismatch (found by the smoke test).** The preflight
  `train.py --dry-run` ran without `--launcher pytorch`, so it saved `launcher: none` in
  `<run>/dropaug.yaml`. The torchrun launch then composed `launcher: pytorch`, and
  train.py exited 2 ("dropaug.yaml exists and differs"). Every `train.sbatch` run
  failed this way within a minute. `--launcher pytorch` is now in `TRAIN_FLAGS`, so both
  steps compose the same config. `scripts/eval.py` / `probe.py` don't read `launcher`
  (`tasks.mmseg.config.build` defaults to `none`), so a saved `pytorch` doesn't affect them.
- **`train.py` never registered the mmseg glue (found by the smoke test).** The mmengine
  config is built in memory (`tasks.mmseg.config.build`), and mmengine only runs
  `custom_imports` when it parses a config *file*. So `tasks.mmseg.register` was never
  imported, and `Runner.from_cfg` died with "GpuAugSegDataPreProcessor is not in the
  mmseg::model registry". `train_mmseg` now calls
  `import_modules_from_strings(**mm_cfg.custom_imports)` before building the Runner, as
  `scripts/probe.py` and `scripts/eval.py` already did. Checked locally (CPU): the Runner,
  its val loop and its test loop build for all four datasets.
- **`.gitignore`** gains `experiments/` (run dirs now sit inside the checkout).
- **Env.** `hydra-core 1.3.7` and `omegaconf 2.3.1` were pip-installed into the conda
  env `sensaug`. `train.py` / `scripts/*.py` compose with Hydra, and `common.sh`'s
  `check_hydra` would have stopped every job without them. Nothing else in the env
  changed (`pip check` is clean).
- **Env: `imagecorruptions 1.1.2`** (`--no-deps`; its cv2 / scikit-image / scipy / Pillow
  were already in the env). Without it, the eval that `train.sbatch` queues after every
  run exits at once ("needs the optional dependency 'imagecorruptions'"), because the
  default suite includes 7 ImageNet-C corruptions. All 7 (`motion_blur zoom_blur pixelate
  jpeg_compression snow frost fog`) ran at severity 3 here, `fog` included, on
  scikit-image 0.25 / numpy 1.26.

### Smoke test

`task.max_iters=200 sa.ratio=0.02 sa.test_ratio=0.02 probe.n_probe=16
probe.analysis.n_min=8 probe.analysis.bootstrap_reps=50`, `EVAL=0`, under
`experiments/smoke2gpu/`. It ran on 2 GPUs (`--gres=gpu:2 --qos=medium`) because 4×A6000 had
a multi-hour queue. That still exercises DDP, the SA loop, the probe and R.

- Attempt 1 (7786330–33): all four failed on the launcher mismatch above.
- Attempt 2 (7786374–77): all four failed on the registry error above.
- Attempt 3 (7786443–46): pascal_voc12 completed (200/200) in ~16 min. The warmup
  rounds ran, then a control probe at iter 40 (32 ops × 16 images, 32/496 FDR survivors)
  and active probes after it, with magnitudes from the SA snapshot (`sa_snapshot/mode`)
  once one existed. The SA curve search and per-op scoring ran too. The run dir holds
  checkpoints, `aug_gradient_log.txt`, `corr_{matrix,bootstrap,redundancy}_log`,
  `sa_curve_log.txt`, `perturb_eval.txt`, `corr_magnitudes.json` and `probe_set.json`.
  ade20k and loveda also completed, and cityscapes did too in 45 min (the largest images).
  All four went through the same SA + probe sequence with no errors.
- Eval smoke on the pascal checkpoint (`eval.strengths=[1.0]`): 7786732 failed on the
  missing `imagecorruptions` (installed above). The rerun, 7786747, wrote clean, all 32
  ops and the first corruption records (36/74 conditions) and was then cancelled. The
  last 6 conditions (Sensaug's cv2 `combination`) were checked locally instead: all 6
  levels run.

### Launched

The first submission (7787172–75, 4×A6000) was cancelled while still queued, because its
estimated start was 00:44–03:25. It was resubmitted on 4×A5000 with `--mem=64gb` (the smoke
runs peaked at 10–14 GB RSS on 2 GPUs). The recipe is unchanged: 4 GPUs at the same
per-GPU batch.

    EXP_NAME=ours-test-A5000-<dataset> SBATCH_ARGS="--gres=gpu:rtxa5000:4 --mem=64gb" \
        bash slurm/submit.sh ours-test-A5000-<dataset> slurm/train.sbatch sensaug <dataset> none 0

Jobs 7787925 (cityscapes), 7787926 (ade20k), 7787927 (pascal_voc12) and 7787928 (loveda)
are named `ours-test-A5000-<dataset>-dropaug-train`. All four started at 22:29 and were
training without errors 10 min later. mmengine's training-only ETA was about 5 h for
cityscapes (160k iterations), 4.5 h for ade20k (160k), 20 min for pascal_voc12 (20k) and
1.5 h for loveda (80k); the SA rounds and probes add to that. Run dirs are
`experiments/ours-test-A5000-<dataset>/`. Each run queues `dropaug-eval` on its best
checkpoint when it finishes.

### Known issues (not fixed)

- `pytest tests/` in `sensaug` (torch 2.0, torchvision 0.15): 347 passed, 49 failed,
  16 skipped. Every failure comes from versions. The torchvision-family vocab ops
  (`randaug_default`, `trivialaug_wide`, `autoaugment`) use `RandAugment._fill` etc.,
  which torchvision 0.15 doesn't have, and one test raises `torch.OutOfMemoryError`
  (torch ≥ 2.1). The DIFF32 ops that "ours" trains and probes with are unaffected.
  Before any non-DIFF32 host runs on Nexus, those ops need a version shim or a newer env.

### Outcome (2026-10-08)

All four training jobs completed with exit 0 (wall time: cityscapes 12h57, ade20k 9h41,
pascal_voc12 3h25, loveda 10h53), each with 20 SA rounds and every probe. Each queued its
eval. Open items, not yet investigated:

- The ade20k eval runs ~17 min per condition (2000 images, batch size 1). 202 conditions
  is about 57 h, so it will hit eval.sbatch's 16 h limit about 3 times. It resumes from
  the JSONL on resubmit (`sbatch slurm/eval.sbatch experiments/ours-test-A5000-ade20k`).
- dropaug.eval's clean mIoU on the best checkpoint is well above the training loop's val
  mIoU for that checkpoint on ade20k and loveda, the two datasets with `reduce_zero_label`.
  It roughly matches on cityscapes and pascal_voc12. Suspect label or ignore-index handling
  in `SegMetric` / the eval data path. Don't compare eval numbers across datasets until
  this is resolved.

## 2026-10-08: KID gets RGB

- **Fix: KID fed Inception BGR.** `SubsetTestLoop` gave torchmetrics' KID (torch-fidelity
  Inception weights, which expect RGB) both sides in mmcv's loaded order, BGR. The
  preprocessor's `bgr_to_rgb` is downstream of the real side, and `_as_loaded` undoes it
  on the fake side. Inherited from Sensaug. Now `SubsetTestLoop(kid_rgb=True)` flips both
  sides to RGB just before `kid.update` (`_kid_view`). It is wired from the new
  `sa.kid_rgb` (configs/sa/default.yaml, default `true`), and `sa.kid_rgb=false`
  reproduces Sensaug. Files: `src/tasks/mmseg/loops.py`, `src/tasks/mmseg/config.py`,
  `configs/sa/default.yaml`. Checked locally on pascal_voc12 val (8 images): the loaded
  batch differs from the image read as RGB from disk by up to 54.8 in a per-channel mean
  (it is BGR). KID's input with the fix differs by ≤0.13 (resize only), and with
  `sa.kid_rgb=false` by 54.8, as before.
- **Correction to the outcome note above.** The eval-vs-training-val mIoU gap on ade20k
  and loveda is not a dropaug.eval bug. Sensaug's own `experiments_final_eval` shows the
  same gap (ade20k 35.3 vs 30.1, loveda 51.2 vs 44.0, seed means).
- **Launched:** `ours-kidrgb-A5000-<dataset>` (7796835 cityscapes, 7796836 ade20k,
  7796837 pascal_voc12, 7796838 loveda). Same setup as `ours-test-A5000-*` (sensaug host,
  selector none, seed 0, 4×A5000, 64 GB) plus the KID fix. The time limit was lowered to
  20 h (`scontrol update`) so the jobs start before the monthly maintenance window
  (2026-10-09 16:45). The eval jobs they queue have eval.sbatch's 16 h limit and may wait
  until after the maintenance.

### Outcome of the KID-RGB runs (2026-10-09)

All four `ours-kidrgb-A5000-*` trainings completed. The pascal_voc12 eval is done. The
cityscapes / ade20k / loveda evals (7800842, 7801089, 7801428) are waiting on the
maintenance reservation. The BGR ade20k eval (7792696) timed out at 16 h with 56 of 160
op conditions done and has not been resubmitted.

The flip did land: KID at level 1.0 for the R and B channel ops swaps between the arms.
For example, cityscapes lighter_R / lighter_B go from 0.052 / 0.161 (BGR) to 0.171 /
0.058 (RGB). But it does not explain color_neg. Its KID is ≈ 0 (|KID| ≤ 0.004) in both
arms on every dataset, and so is color_pos's. Its instability rank across the 3 curve
searches is no better (mean rank 9.2 RGB vs 8.8 BGR, out of 30), and training val mIoU is
within ±0.5 on all four datasets. The working explanation is still that the model barely
reacts to color_neg (mIoU drop ≈ 0 and KID ≈ 0), so both normalised objective terms are
noise divided by noise. Comparison script: a scratchpad, not checked in.

## 2026-10-09: KID noise check (no training)

`scripts/kid_noise_check.py` (new) rebuilds a run's SA images exactly as
`SubsetTestLoop` sees them (eval slot → `_as_loaded` → RGB) and compares two KID
estimators per (op, level): SubsetTestLoop's `subset_size=4` (30 redraws) and KID on all
images at once. Each is tested against its own permutation null (200 perms). It needs no
model. It ran on the login node (CPU, `nice`) while the cluster was in maintenance, on
`ours-kidrgb-A5000-{cityscapes,pascal_voc12,loveda}` (levels 0.5 and 1.0, all 32 ops).
Output is in `experiments/kid_check/<dataset>.{json,log}`. ade20k was not run.

Findings (log-level numbers, not dropaug.io artifacts):
- subset4's clean-vs-clean spread reproduces the null taken from Sensaug's logs
  (cityscapes 95% [−0.0145, +0.0023] here vs [−0.016, +0.007] from the logs).
- KID on all images has a 5–10× tighter null (97.5th pct: cityscapes 0.0024 vs 0.0124,
  pascal 0.0008 vs 0.0065, loveda 0.0008 vs 0.0082).
- Ops detected (cityscapes / pascal / loveda): at level 1.0, subset4 reliably 21 / 16 / 25
  of 32 and full 27 / 25 / 29; at level 0.5, subset4 14 / 5 / 20 and full 23 / 17 / 26.
  The gain is mostly at mid levels (shear/translate, hue, brightness, channel ops).
- One subset4 draw ranks KID(0.5) above KID(1.0) 6–16% of the time; full KID does so for
  0–3% of ops.
- No estimator sees sharpness_pos/neg or blur at full strength, nor color_neg/color_pos
  on cityscapes and pascal. These ops barely change the image: at level 1.0 on a
  cityscapes image the mean |pixel change| is 0.5–1.6/255 (sharpness, blur) and 4/255
  (color_neg), vs 55–96/255 for lighter_R or noise. So color_neg's level instability
  comes from the op's weak magnitude range, not from the KID estimator.
