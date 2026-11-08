# FACTS: the claims ledger

Every number that a paper, slide, README or docstring states as a finding traces to a
row here. A row names the claim, its value, where the number comes from (an artifact,
a log, a job, or a formula you can recompute) and its status. If a number has no row,
don't cite it.

**Status**

- `measured`: re-derived by dropaug from an artifact. The row cites the `dropaug.io`
  file and its `vocab_hash` / `probe_set_hash` (or the eval file and its checkpoint).
- `inherited`: stated in Sensaug's docs or code and not yet re-derived here. Parts of
  Sensaug's docs are stale (`third_party/Sensaug/RESULTS_SUMMARY.md`), so an
  inherited number is a lead, not a result.
- `derived`: follows from a formula, with no measurement behind it.

**Rules**

- Measured rows cite an artifact written by `dropaug.io`, never a number copied off a
  log line. The artifact's hashes are what make two numbers comparable: an R over
  other ops, other images or other op definitions is a different measurement.
- A number measured at one magnitude setting holds at that setting only. Record the
  `magnitude_source` / `magnitude_mode` with it.
- Never pool: one R is one checkpoint (see CLAUDE.md, "One frozen frame per R").
- When a number is superseded, strike it through and add the new row below it. Don't
  delete it.
- Sensaug paths are relative to `third_party/Sensaug/` (pinned at `1192a07`).

## Measured (dropaug)

None yet. No probe, R or eval has been run on real data through dropaug.

| # | claim | value | artifact (path, vocab_hash, probe_set_hash / checkpoint) | date |
|---|---|---|---|---|

## Inherited (Sensaug, not re-derived)

| # | claim | value | source | caveat |
|---|---|---|---|---|
| I1 | Mean \|r\| over R's off-diagonal cells | 0.11–0.22 | `CLAUDE.md:508`, `sensaug/redundancy.py:31`, `sensaug/hooks/grad_sens_analysis.py:1029` | The runs and matrix (R_raw or R_scalenorm) aren't named. The geometric rows were contaminated by label misalignment until `_matrix_in_frame` landed. |
| I2 | Raw red(a) row sums across logged matrices: spread of means / of SDs | means −0.71 to +4.08, SDs 0.24 to 3.69 | `sensaug/redundancy.py:21-23` | "the four logged corr_matrix_log.json files", which aren't identified |
| I3 | Max/min q spread of the KL tilt at λ=0.25 | 2.3–3.1× at every checkpoint | `sensaug/redundancy.py:25-27` | same four files, 14- and 32-op vocabularies |
| I4 | Largest pixel change at magnitude 0.5 | `blur` 78/255, `noise` 251/255 | `sensaug/hooks/grad_hook.py:16`, `:274` | measured on "a random image"; Sensaug's implementations |
| I5 | SegFormer mit-b0 / Cityscapes / 160k, clean mIoU | random 75.47; grad_corr (none) 75.26; sensaug 74.89; grad_corr (soft, λ=0) 74.67 | `RESULTS_SUMMARY.md` (2026-08-24), where sensaug is labelled `ours` (`--aug-type ours`) | One seed each. Every grad_corr run had λ=0, so Lever 3 never fired. The random baseline went through a plain `ValLoop` (aAcc 95.81 vs ~94.7), so it isn't comparable as-is. |
| I6 | Same runs, mean perturbed mIoU over the 32 ops | grad_corr (none) 70.31; sensaug 68.96; grad_corr (soft, λ=0) 69.11; random n/a | `RESULTS_SUMMARY.md` | as I5. The sensaug run's corruption sweep never finished. |

## Derived (formulas)

| # | claim | value | formula / where |
|---|---|---|---|
| D1 | Closure null of a per-image-normalized R | −0.077 at A=14, −0.032 at A=32 | −1/(A−1); `stats.closure_null` |
| D2 | Standard error of r at the `n_min` floor | ≈ 0.10 at n=100 | 1/√(n−3); `stats.fisher_z_se` |
| D3 | Share of measured ops a hard selector keeps | λ=0.25 → ~80%, 0.5 → ~67%, 1 → 50%, 2 → ~33% | ⌈A/(1+λ)⌉; `selection.budget_from_lambda` |
| D4 | Benjamini-Yekutieli's extra divisor over BH | 5.1 at 91 cells (A=14), 6.8 at 496 (A=32) | Σ_{i≤m} 1/i; `stats.fdr_correct(method="by")` |
| D5 | Eigenvalue range of R for independent ops from sampling noise alone | 0.47–1.73 at A/N = 0.1 | (1 ± √(A/N))²; `stats.marchenko_pastur_edges` |
| D6 | P(none) of the SA pdf (`generate_pdf_new`), 4 levels per op | 0.185 at 30 ops, 0.234 at 10 (not 1/(A+1)); 15.8% of each op's allotment leaks to none | 1 − A/(A+1) · Σ_{i<4} betabinom.pmf(i; 4, 0.75, 1); `sa.pdf` |
| D7 | P(none) of `generate_pdf_new_weighted_aug`, 4 levels per op | 0.0559 at 30 ops, 0.0675 at 10 | 1 − 0.95 · Σ_{i<4A} betabinom.pmf(i; 4A, 0.75, 1); `sa.pdf` |
| D8 | Chance ceiling of the largest off-diagonal cell of a Spearman R (95%, one-sided union bound) | 0.37 at A=32, N=100; 0.17 at A=32, N=500. τ = 0.5 clears it from N ≥ 57 at A=32 (N ≥ 44 at A=14) | z_{1−0.05/m}/√(N−1), m = A(A−1)/2: Spearman's null variance 1/(N−1) and its normal limit; `stats.null_max_bound` |
| D9 | MADAug's policy network, linear on SegFormer-B0's pooled 256-channel feature: trainable / frozen parameters with k ops pruned | 8,738 / 0 at k = 0; 6,682 / 1,028 at k = 4 | (F+1)·2(17−k) trainable, (F+1)·k frozen (the pruned weight logits), F = 256; `madaug.Projection.param_counts` (checked on CPU) |
| D10 | Faster AutoAugment's search policy, L = 10 sub-policies × K = 2 stages: parameters over the ops searched | 1,480 over all 26; 1,200 with 5 left out (4 with a magnitude, 1 without) | L·K·(2·A + A_μ), A ops searched, A_μ of them with a magnitude (22 of 26); `Policy.param_count` (checked on CPU) |
| D11 | Deep AutoAugment's searched policy (K layers over the kept transforms) and its search's passes | 695 logits at K = 5 over all 139; 570 with 25 transforms left out (Rotate_pos, Solarize, Contrast_neg, Cutout). At DeepAA's released settings: 43,958,272 image backward passes and 148,029,440 JVPs over all 139 | K·\|kept\| logits; backward S·n·(V + \|kept\|) + (K−1)·S·n·(V + 4\|kept\|), JVP S·n·\|kept\|·(1 + 16(K−1)), S = 1024 steps a layer, n = 16 images, V = 64; `deepaa.Search.param_count`, `deepaa.pass_counts` (checked on CPU) |
| D12 | Adversarial AutoAugment's controller (hidden 100, embedding 32): trainable parameters with k types removed | 57,090 at k = 0; 56,691 at k = 3 | 57,090 − 133·k (a type's head row, 100 + 1, and embedding row, 32); `advaa.Controller.param_count` (checked on CPU) |
| D13 | Greedy AutoAugment's search: trials over the (op, level) pairs a prune leaves | 250 a level over all 25 search ops (500 at k = 2); 225 a level with Rotate_pos, Color_neg, Sharpness_pos and Sharpness_neg left out | 10 levels a search op, an enhance direction 5 (levels 0–4 or 5–9), × k levels (Eq. 1); `hosts.greedy_autoaugment.candidates`, `greedyaa.trial_counts`, `gaa_search.py --dry-run` (checked on CPU) |
| D14 | AdaAug's policy h, linear on the search segmentor's pooled feature (SegFormer-B0: F = 256), over the outputs a prune leaves, and the search's cost at its defaults on Cityscapes | 8,738 parameters and 17 explore copies per held-out image over all 17 outputs; 7,710 and 15 with ShearX_pos, ShearX_neg, Rotate_pos, Solarize and Contrast_neg left out (ShearX and Solarize lose their outputs; Rotate and Contrast keep theirs, the other direction). At 2975 training images, batch 8, 200 epochs, an explore of 16 every 3 steps: 74,200 exploit steps, 24,800 explores, 6,745,600 explore copies over all 17 (5,952,000 at 15) | (F+1)·2n parameters, n outputs (`hosts.adaaug.searched_outputs`); explores ⌈⌊N/b⌋/3⌉·E, copies explores·16·n; `adaaug.policy_params`, `adaaug.pass_counts`, `adaaug_search.py --dry-run` (checked on CPU) |
