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
| I5 | SegFormer mit-b0 / Cityscapes / 160k, clean mIoU | random 75.47; grad_corr (none) 75.26; ours (SensAug) 74.89; grad_corr (soft, λ=0) 74.67 | `RESULTS_SUMMARY.md` (2026-08-24) | One seed each. Every grad_corr run had λ=0, so Lever 3 never fired. The random baseline went through a plain `ValLoop` (aAcc 95.81 vs ~94.7), so it isn't comparable as-is. |
| I6 | Same runs, mean perturbed mIoU over the 32 ops | grad_corr (none) 70.31; ours 68.96; grad_corr (soft, λ=0) 69.11; random n/a | `RESULTS_SUMMARY.md` | as I5. Ours' corruption sweep never finished. |

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
