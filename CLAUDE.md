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

**HARD RULE: the Slurm job cap.** Everything this repo sends to Slurm (train, probe,
matrix) is **at most ONE job on A6000s (`rtxa6000`) OR TWO jobs on A5000s (`rtxa5000`) in
flight at a time. Never a mix, never more.** A training job keeps its 4 GPUs (probe has
1); the cap is on how many jobs run, not on GPUs per job. A job submitted past the cap
waits for the ones ahead (`--dependency=afterany`), like a queue. `eval.sbatch` is the
one exemption. This is the owner's most important standing instruction: never submit with
a bare `sbatch`, never pass `--gres` / `-G` / `--gpus*` / `--dependency` to get around
it, never raise a matrix throttle, and never write a new sbatch script that skips the
gate. Submit with `bash slurm/submit.sh <name> slurm/train.sbatch ...` (`GPU=a5000` for
the A5000 arm; default a6000) or `bash slurm/matrix.sbatch --submit`. See
`slurm/gate.sh` and the `slurm/` section.

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
| `randaug_default` | 14 | torchvision v2 `RandAugment`'s own per-op dispatcher, on 8-bit images, `grad_mode="none"`. The 4 enhance ops also carry an exact `tangent` (below) | AutoAugment Table 6, which RandAugment inherits (degrees, factors, bits, px at 331×331) |
| `trivialaug_wide` | 14 | torchvision v2 `TrivialAugmentWide`'s dispatcher | TrivialAugment Table 8, wide ranges |
| `autoaugment` | 1 | `Invert`, the one AutoAugment op RandAugment dropped, via torchvision v2 `AutoAugment`'s dispatcher. AutoAugment's other 13 ops are `randaug_default`'s (same Table 6 ranges) | no magnitude |
| `trivialaug_diff` | 22 | `trivialaug_diff.py`: `trivialaug_wide`'s 13 ops rewritten so autograd differentiates the magnitude, each of the 9 signed ops split into `_pos` / `_neg` (no Identity). The geometric ops are torchvision's affine matrix (`_get_inverse_affine_matrix`) built in torch and warped by kornia (bilinear, translate not truncated), label moved as in `geometric`; the enhance ops are torchvision's float blend with no uint8 round trip. Posterize, Solarize, AutoContrast and Equalize are `trivialaug_wide`'s own ops. `tests/vocab/test_trivialaug_diff.py` pins them against torchvision (enhance within one 8-bit level, geometric equal at NEAREST), autograd against central differences, and `BINS` against TrivialAugmentWide's. Its `split_specs` / `split_table` / `build_split_family` build the other split families | TrivialAugment Table 8, unsigned; the direction is in the name |
| `randaug_diff` | 22 | `randaug_diff.py`: `randaug_default`'s 13 ops split as `trivialaug_diff` splits its own (`build_split_family`, autograd in the magnitude). Posterize, Solarize, AutoContrast and Equalize are `randaug_default`'s own ops. `BINS` is torchvision RandAugment's table up to 31 bins (level M is bin 3M); past 31 it continues the grid linearly (the paper's levels above 10, its footnote 2), each op stopped where it stops meaning anything (threshold 0, 0 bits, an enhance `_neg` at factor 0). Those bins can lie outside `mag_range` (see the `randaugment_diff` host). The `autoaugment_diff` host reads it at AutoAugment's 10 bins: torchvision's AutoAugment table has RandAugment's formulas (checked against torchvision 0.27's source, no test). `tests/vocab/test_randaug_augmix_diff.py` | AutoAugment Table 6, unsigned; the direction is in the name |
| `augmix_diff` | 22 | `augmix_diff.py`: torchvision `AugMix(all_ops=True)`'s 13 ops, as Sensaug's AugMix arm ran it (the paper's default leaves out the 4 enhance ops), split the same way. Posterize, Solarize, AutoContrast and Equalize are torchvision's own, through AugMix's dispatcher (Posterize runs down to 0 bits). `BINS` is AugMix's table at its `_PARAMETER_MAX` = 10 bins. The geometric ops move a label like every geometric vocab op (the probe needs it); AugMix training never hands them one. `tests/vocab/test_randaug_augmix_diff.py` | AugMix's ranges, unsigned; translate a fraction of W/H (≤ 1/3) |
| `smartaug_diff` | 21 | `smartaug_diff.py`: SmartAugment's Table 2 (Negassi, Wagner & Reiterer 2022), its 9 signed ops split via `split_specs`; AutoContrast, Equalize and Solarize are `randaug_default`'s. No code was released, so `BINS` (levels 0 to 30) is a reconstruction: geometric `hi·b/30`; enhance factor `0.1 + 1.8·b/30`, `_neg` below level 15 and `_pos` from it (the direction follows the level, not a random sign); Solarize threshold `256·b/30` (RandAugment's code; AutoAugment's and AugMix's run the other way). No test of the bins | Table 2's ranges, in `augmix_diff`'s units |
| `madaug` | 2 | `madaug.py`: the two MADAug ops no other family has, in torch, `grad_mode="none"`. `Cutout` is MADAug/AdaAug's `CutoutAbs`: a square of side v·width, fill (125, 123, 114), PIL's inclusive corners, its centre uniform on (1, w] as `np.random.uniform(w)` draws it (ported as is), photometric (the label stays). `Flip` is `PIL.ImageOps.mirror`, geometric, no magnitude. MADAug's other 14 ops are `randaug_default`'s 13 and `autoaugment.Invert` (`configs/vocab/madaug.yaml`). Checked by hand against MADAug's own PIL `operation.py` at levels 0.1 to 0.9 (scratchpad, no test): Cutout, Flip, Shear, Solarize, Posterize, Brightness, Sharpness, Equalize and Invert are pixel-identical; Rotate differs on < 0.05% of pixels (the frame's edge); Contrast and Color by one 8-bit level on 19-62% of pixels, AutoContrast on 7%; Translate by 1 px when the shift's fraction is ≥ 0.5 (torchvision truncates, PIL rounds) | Cutout: a fraction of the width, [0, 0.2] |
| `faster_aa` | 26 | `faster_aa.py`: Faster AutoAugment's 17 ops (the paper's Table 2), dda's `functional.py` bodies (github.com/moskomule/dda) in dda's units, ported from kornia 0.3.1 to 0.8.2. The 9 signed ops are split into `_pos` / `_neg` (in the policy search too, a departure from the paper's random sign). Geometric: dda's kornia matrices (shear about the top-left pixel, translate a fraction of W/H, rotate about the center), label moved as in `geometric`; HorizontalFlip is `madaug.flip`. Enhance ops: dda's blends, factor 1 ± m. Solarize, Posterize and Cutout are `grad_mode="none"`; Cutout is Fast AutoAugment's (`madaug.cutout`, the search space the paper says it shares). `SamplePairing` (dda's, partner from a `randperm` of the batch, possibly itself) has `couples_images=True`. Flagged against dda: TranslateX added (dda lists TranslateY twice); Posterize fixed (dda shifts left then right, an identity); Solarize's range runs to 256/255 for an exact identity (dda's μ = 1 still inverts pure white; the search keeps dda's [0, 1]); kept: gray weights 0.299/0.587/0.110. Checked by hand (scratchpad, no test) against dda's formulas and kornia's shear/translate/rotate: equal (shear at `align_corners=True`; kornia 0.8.2's `shear` alone defaults to False) | dda's: shear factor ≤ 0.3, translate fraction ≤ 0.45, degrees ≤ 30, factor 1 ± m ≤ 1, Solarize threshold, Posterize μ (8 − ⌊8(1 − μ)⌋ bits), Cutout ≤ 0.2 of W, SamplePairing weight ≤ 1 |

Every op takes its magnitude at the images' dtype precision (the torchvision ops too),
so a python float and a float32 tensor holding it give identical output. The torchvision
dispatchers take their fill from `method._fill` (torchvision 0.27) or `method.fill` (0.15,
the `sensaug` env on Nexus), through `randaug_default._fill_of`.
`magnitude_bins(name, num_bins)` → `(magnitudes, signed)`: the bins a torchvision method
draws a `randaug_default` / `trivialaug_wide` / `trivialaug_diff` / `randaug_diff` /
`augmix_diff` op's magnitude from, in the op's units (the split families' are unsigned: the
sign is the `_pos` / `_neg` choice). It is read off torchvision's own `_AUGMENTATION_SPACE`
(the families' `BINS`), not re-derived, except `randaug_diff` past 31 bins (the grid
continued) and `smartaug_diff` (SmartAugment's levels, reconstructed; no torchvision method).
torchvision's AutoAugment shares RandAugment's table, and float32 round-off at an end bin
is clipped to `mag_range`. Checked by hand against `_AUGMENTATION_SPACE` element by element
(RandAugment and TrivialAugment at 31 bins, AutoAugment at 10); no test yet.
`OpSpec.couples_images` marks an op whose output for one image depends on the batch's other
images (`faster_aa.SamplePairing`): the probe gives it an `unprobed` NaN row (faded or not),
the audit skips its per-image checks, and `tasks.augment.apply_draws` runs it over the whole
batch, every image that didn't draw it at the identity.
`OpSpec.tangent` (optional, photometric ops only) is `(images, magnitude) -> (B, C, H, W)`,
the exact derivative of the op's output in its magnitude, for an op autograd can't
differentiate. Only the torchvision enhance ops have one (`randaug_default.enhance_tangent`,
shared by `trivialaug_wide`). torchvision computes each as `clamp(f·x + (1−f)·d)` cast to
uint8, so the tangent is `x − d`, and 0 where the clamp holds. `d` is torchvision's own
image at factor 0, except Contrast's, which is the float mean of `rgb_to_grayscale` (its
factor-0 output truncates that mean). Checked by hand on all four ops at factors 0.01
to 2.0: rebuilding the blend from `d` gives torchvision's output pixel for pixel. On a
smooth float64 model, the probe's tangent row is within 0.6% of `d/df` of the float blend.
`tests/probe/test_vocab_ops.py` runs every vocab op and every host's vocabulary
through `collect_grads`, and pins the audit's "strongest magnitude" to the probe's
`strength_to_magnitude(op, 1.0)`.

## `src/hosts/`: one adapter per augmentation method

train.py picks a host with `build_host(name, **kwargs)`. A host is the method's
sampling policy only: pure numpy, no torch, and it does not apply ops or run the
probe. The one exception is `hosts/faster_autoaugment/`, whose policy search (torch) sits
next to its host (the user's choice); `hosts/__init__` imports only its numpy `host.py`. `AugHost` (`base.py`) provides `vocabulary()` (vocab op names, the axes of R
and q), `base_probs()` / `sampling_probs()`, `set_weights(q)`, `should_probe(it)` and
`probe_role(it)` (`"control"` R must not become a q). A method with a magnitude policy of
its own also has `num_magnitude_bins` and `draw_bins(B, rng)`, which give per image the bin
it would apply each op at and whether a signed op is negated (p = 0.5). That is the
probe's intensity for it (`tasks/magnitudes.py`). It uses the method's distribution, never
the q-tilted one. `sensaug`, `uniform`, `madaug` and `faster_autoaugment` return None (the
last two give the probe its magnitudes another way: `method_magnitudes`, `draw_magnitudes`).

| host | vocabulary | notes |
|---|---|---|
| `sensaug` | the 32 DIFF32 ops | the probe fires on Sensaug's round schedule (ported). The SA pdf arrives through `set_sa_pdf()` because the SA loop is not ported; `pdf()` returns it with q applied |
| `randaugment` | `randaug_default.*` minus Identity | torchvision defaults: N=2, M=9 of 31 bins. `draw_bins`: bin M for every image |
| `trivialaugment` | `trivialaug_wide.*` minus Identity | `draw_bins`: one uniform bin per image, shared across ops |
| `trivialaugment_diff` | `trivialaug_diff.*` (`split_ops`) | TrivialAugmentWide's policy with the random sign made explicit: a signed op's 1/13 is split evenly between `_pos` and `_neg` (1/26 each), `none_prob` 1/14, so uniform q is TrivialAugmentWide exactly, and q can weight the two directions differently. `draw_bins` is `trivialaugment`'s; its flips are ignored. Trains (`tasks.augment.BIN_HOSTS`) |
| `randaugment_diff` | `randaug_diff.*` (`split_ops`) | RandAugment at the paper's published (N, M) for `dataset` (`PUBLISHED` / `STAND_INS` in `hosts/randaugment.py`, the paper's numbers, not a config): every segmentation set takes ImageNet ResNet-50's (2, 9), cifar10 (3, 5), cifar100 (2, 14). N draws per image, each uniform over the 13 ops plus Identity at level M (bin 3M), the signed ops' shares split as `trivialaugment_diff`'s, `none_prob` 1/14. `draw_bins`: bin 3M for every image (flips ignored). Trains (`BIN_HOSTS`, `num_ops` rounds). Above level 10 (cifar100's 14) the bins pass `mag_range` (Rotate_pos 42 on [0, 30], Posterize 2 bits on [4, 8]): training applies them, but the probe refuses them (`ProbeError` in `_batch_magnitude`) |
| `augmix_diff` | `augmix_diff.*` (`split_ops`) | AugMix as Sensaug ran it (`hosts/augmix.py`, Sensaug's settings, not a config): severity 3, width 3, depth 1 to 3, α = 1, all 13 ops. Per image `m0·x + Σ wᵢ·chainᵢ(x)`, weights from `draw_mix` (Dirichlet), each chain step one op from `sampling_probs()` at a bin uniform on the first 3 of 10. `none_prob` 0 (no Identity). Trains (`BIN_HOSTS` / `MIX_HOSTS`); the mask is never moved, as in Sensaug, so a geometric step misaligns the mix with its label, while the probe moves the label with the op. `draw_bins`: one bin per image on 0..2, shared across ops; bin 0 is every split op's identity, so about a third of the probe images are measured at m = 0 |
| `smartsampling_diff` | `configs/vocab/smartsampling_diff.yaml`: 13 `smartaug_diff` ops (Sharpness, AutoContrast and the 5 geometric ops, the 6 signed ones split) | SmartSamplingAugment (Negassi et al. 2022, Algorithm 2), reconstructed (no code released; `hosts/smartsampling.py`). Per image: augmented with P = min(1, step / max_iters) (`set_step`, from `HostProgressHook`), then 2 distinct ops drawn by W at one level uniform on 5..30. W is RandAugment's Table 5 positive gains (Identity out), normalized; a geometric op's sign at p = 0.5, Sharpness's direction by the level (`_neg` 10/26 of its W, `_pos` 16/26). `base_probs()` is that, not uniform. q tilts W by q / base inside the draw; `none_prob` is 1 − P, never moved by q; `prune(hold_none_prob=False)` raises. Trains (`BIN_HOSTS` / `SLOT_HOSTS`: `draw_slots`). `draw_bins`: one u per image mapped onto each op's own levels, so Sharpness `_neg` is strongest at u = 0, where every other op is weakest |
| `autoaugment` | `randaug_default.*` ops used by the ImageNet policy, plus `autoaugment.Invert` | q tilts the 25 sub-policies; `sampling_probs()` is what the policy actually draws, not q. `draw_bins`: per op, a bin from the policy steps that apply it, weighted by step p. Probe-only; trains through `autoaugment_diff` |
| `autoaugment_diff` | `configs/vocab/autoaugment_diff_imagenet.yaml`: 15 ops, the `randaug_diff` ops the ImageNet policy applies (Rotate, Color, Contrast, Sharpness and ShearX split) plus `autoaugment.Invert` | `autoaugment`'s policy and tilt over `randaug_diff` at AutoAugment's 10 bins (torchvision's AutoAugment and RandAugment share their `_AUGMENTATION_SPACE` formulas), published, not searched. A step on a signed op chooses between its `_pos` / `_neg` columns, half each. q tilts per step: the sub-policy weight is Π (Σ share·q/base)^p over its steps and the step picks a direction ∝ share·q/base, so q = 0 on one direction forces the step to the other and removes no sub-policy; 0 on both (or on an unsigned op) removes its sub-policies as in `autoaugment`. A prune is the same, the directions renormalized for the run. With one column per op this is `autoaugment`'s tilt exactly (`AutoAugmentHost._columns`). `none_prob` 0. Trains (`BIN_HOSTS` / `SLOT_HOSTS`: `draw_slots`, a sub-policy per image, each step with its p). `draw_bins`: `autoaugment`'s, both directions of an op sharing its bins (flips ignored); Color is applied at bin 0 (its identity) in 0.8 of its 3.8 step-weight, so about a fifth of its probe images are measured at m = 0, where `Color_pos` and `Color_neg` are exact negatives (a within-op pair). No test of its own beyond the host-contract tests |
| `madaug` | `configs/vocab/madaug.yaml`: 16 ops (`randaug_default`'s 13, `autoaugment.Invert`, `madaug.Cutout` / `Flip`) | MADAug's sampling (`hosts/madaug.py`); its policy network is `src/madaug/`, trained by `MADAugPolicyHook`. Per image: augmented with p(t) = tanh(t / τ), τ = 40/240 of training (`set_step`); then `k_ops` = 2 entries drawn from the policy's p without replacement (Identity applies nothing), each level δ-perturbed (0.3), the 5 geometric ops signed at random, mapped through MADAug's ranges into the vocab op's units (`level_to_magnitude`). `base_probs()` is a running mean of the policy's p (`observe`); q is held as q / base and tilts each image's p over the ops with p(Identity) held; `none_prob` is 1 - p(t). `draw_bins` is None: the probe's magnitude is the policy's own level on each probe image (`method_magnitudes`). `prune(hold_none_prob=False)` raises. Trains (`tasks.augment.POLICY_HOSTS`), on mmseg only |
| `faster_autoaugment` | `faster_aa.*` (26 ops) | Faster AutoAugment's searched policy (`hosts/faster_autoaugment/host.py`), read from `policy_file` (the search's `final.npz`): L = 10 sub-policies × K = 2 stages, each stage a weight w, probability p and magnitude μ per op. Per image: a sub-policy uniformly, then per stage op a with probability w·p at μ, or nothing (dda samples per chunk of a 16-way split; at ≤ 16 images a GPU that is per image). `base_probs()` is each op's share of Σ w·p; q tilts every stage by q / base with each stage's no-op mass held; `sampling_probs()` is the realized share; `none_prob` the mean stage no-op. The prune zeroes the ops in every stage (`hold_none_prob=False` lets stage masses shrink). `draw_stages` for training (`tasks.augment.STAGE_HOSTS`); `draw_magnitudes` for the probe (per op, its searched μ's weighted by w·p, untilted). `policy_file: null` is dda's starting point (all equal, μ = p = 0.5), probe-only: `check_trainable` refuses it |
| `uniform` | any; defaults to DIFF32 | Sensaug's `--random-aug` bank |

Rules every host keeps. The no-op draw (Identity, `("none", 0)`) stays out of the
vocabulary, and `set_weights` holds its `none_prob` fixed; AutoAugment is the one
exception, since it has no no-op draw. q replaces the previous q rather than
compounding it, so derive it from `base_probs()`. A non-finite q raises.
`tests/hosts/` pins the names against `vocab`, the torchvision op spaces and policy,
and Sensaug's `round_schedule` and `redundancy.reweight`.

**The static prune** (`hosts/prune.py`, the top-level `prune:` block of
`configs/train.yaml`) is Sensaug's `--pruned-augmentations` plus its `null` random-prune
arm, for every host and every selector. `prune.ops` names ops; `prune.random: K` draws K
more before training with Sensaug's `draw_random_prune` (`RandomState(prune.seed)`, by
default the run's seed), from the vocabulary minus `prune.ops` and, on the sensaug host
under `sa.remove_H`, minus lighter_H / darker_H. One op name is one unit (`X_pos` and
`X_neg` are drawn independently). `dropaug.config.prune_of(cfg)` → `Prune` is a pure
function of the config, so every rank, the `--dry-run` preflight, a resume and the
post-hoc scripts get the same set; Sensaug instead derived its seed from `SLURM_JOB_ID`.
`AugHost.prune(names, hold_none_prob=)` (applied by `build_host(pruned=...)`, which
`host_kwargs` feeds) keeps the ops in the vocabulary at base probability 0, renormalizes
the rest, and makes `set_weights` refuse a q that revives one. `hold_none_prob: true` (the
default) keeps `none_prob`; false is Sensaug's default, the no-op draw renormalized with
the survivors (1/(A−k+1) on a uniform host). sensaug: the warmup policy is pruned, the SA
loop gets the set as `pruned_augmentations` and `prune.hold_none_prob` as its
`hold_none_prob`, and `set_sa_pdf` refuses a pdf that samples a pruned op. autoaugment:
every sub-policy that applies a pruned op is dropped, and an op left with none is pruned
by implication (`prune()` returns it, `check()` warns). `prune.probe: false` (default)
leaves the pruned ops out of the in-training probe (NaN rows, method `static_pruned`);
true measures them, and either way the R the selector reads sets them aside, so q is the
same. train.py writes `<run>/prune.json` (`Prune.record`) and refuses a run dir whose
record disagrees; the run name gets `_prune-o<N>.<sha1 of the sorted names, 6 hex>-r<K>[s<seed>]`
(`prune_tag`, spelled identically in `train.sbatch`); `scripts/eval.py` records
`meta.pruned_ops`. Both that and prune.json's `kept` follow `host.pruned()` (what the
host realizes, so AutoAugment's implied ops are in), while prune.json's `pruned` stays the
configured set; a prune.json written before this, on an autoaugment run with implied ops,
no longer matches the config and a resume refuses it. `sa.pruned_augmentations` / `sa.hold_none_prob` moved here, and
`prune_of` refuses them. `tests/hosts/test_prune.py`, `tests/config/test_prune_config.py`,
`tests/probe/test_static_pruned.py`.

Each `configs/host/<method>.yaml` also has a `fade:` block (`ops`, `at`), which sets the
ops the probe fades in (see `src/probe/`). `host_kwargs` pops it, so it never reaches the
host. `dropaug.config.fade_of(cfg)` reads it, and `check()` validates it against the vocab.
randaugment and autoaugment have `ops: non_differentiable` (every op). trivialaugment
fades only the four ops with no derivative in their magnitude (Solarize, Posterize,
AutoContrast, Equalize). Its enhance ops go through their tangent, and its geometric
ops through finite differences, which move the label with the image.
trivialaugment_diff fades the same four (its other 18 ops go through autograd), and so do
randaugment_diff and augmix_diff. autoaugment_diff fades those four and Invert (its other
10 go through autograd). smartsampling_diff fades only AutoContrast (its other 12
go through autograd). faster_autoaugment fades seven: HorizontalFlip, Solarize, Posterize,
Invert, AutoContrast, Equalize and Cutout; SamplePairing stays an `unprobed` NaN row
(`couples_images`).
madaug fades seven: those four, Invert, Cutout (its hard edge steps) and Flip; its enhance
ops go through their tangent and its geometric ops through finite differences, as
trivialaugment's. madaug's yaml also has a `policy:` block (MADAugPolicyHook's keys),
which `host_kwargs` pops too and `dropaug.config.policy_of(cfg)` reads.
sensaug and uniform have `ops: null`. A run saved before the block existed fades nothing.

## `src/probe/`: measuring G

Ported from Sensaug's `CollectGradientHook` and `compute_grad_corr.py`, with no
mmseg/mmengine and no knowledge of any op name. It stops at G (ops × probe images);
turning G into R is `src/stats/`. The package docstring
(`probe/__init__.py`) holds the two contracts the probe relies on:

- **ops** are a vocab (`build_vocab`), in G's row order. `grad_mode="autograd"` ops are
  differentiated directly. `"none"` ops go through their exact `tangent` when they have
  one (method `"tangent"`: the loss at `op(x, m) + δ·tangent`, differentiated in δ at 0),
  and by central finite differences otherwise (`finite_diff.py`). Ops with no
  `mag_range`, and ops that `couples_images`, get a NaN row labelled `unprobed`, and the ops in a sweep's `skip` (the
  run's static prune) one labelled `static_pruned`. Every row that isn't faded is
  `d loss / d strength`: the derivative in m times `strength_sign` (+1 above
  `identity_at`, −1 below). Without that sign, an image on which a method negated a
  signed op would enter R with the opposite sign. It is +1 on every DIFF32 op, so their
  rows are unchanged.
  A sweep's `fade` (a rule, `"non_differentiable"`, or a list of op names) overrides
  the last three for the ops it selects. Each of those is faded in,
  `x + a·(op(x, m) − x)`, with `op(x, m)` computed under no_grad at its probe magnitude
  `m`. Its row is `d loss / d a` at `a = fade_at` (default 1, the augmented image), and
  its method is `"fade"` (`fade.py`). An op with no magnitude gets a row this way too.
  `Sweep.methods` says which ops were faded, and `meta["fade_at"]` holds `a`. A faded row
  is a derivative along the fade, not `d loss / d m`. For a geometric op at 0 < a < 1 the
  image is a ghost of two frames, and the warped label matches only one of them.
- **a model adapter** (`ModelAdapter`, not a host) exposes `.model`,
  `to_unit(batch) -> (images01, label)` and `loss(images01, label, batch) -> (B,)`.
  `tasks.mmseg.MMSegBackend` is one (see `src/tasks/`).

| file | what |
|---|---|
| `probe_set.py` | `ProbeSet.build(ids, n, seed)`: a subset chosen from the sorted ids, so it doesn't depend on listing order. Its sha256 `digest` goes on every G |
| `grad_collect.py` | `collect_grads(adapter, batch, ops, magnitudes)` → `{op: (B,)}`. Refuses train-mode modules, checks the label moved for geometric ops, pins each op's RNG by `(probe_seed, name)`. `check=True` (first batch of a sweep) also rejects a non-deterministic `to_unit` and any image coupling. `op_method` (autograd / tangent / finite_diff / unprobed), `far_end` (the direction an op strengthens in, which `strength_to_magnitude` uses too) and `strength_sign`. The sign was checked by hand: a finite-difference row of Rotate at ±40° equals the sign times the raw central difference. `probe_batch` (what the sweep calls) adds each image's loss under every op at its magnitude and on the clean image, `ProbeBatch(grads, losses, clean)`: mostly forwards the rows already ran, plus one per batch (clean), one per finite-difference op and one per faded op at `fade_at` ≠ 1. A batch-mean loss at B > 1 gives NaN losses. Checked in a scratchpad harness (no test): G bit-identical to the code before it on DIFF32 and trivialaugment (fade at a = 0.5), and every loss equal to an independent no-grad forward of `op(x, m)` |
| `finite_diff.py` | the step is a fraction of each op's `mag_range` width (default 0.05). Quantized ops such as Posterize need a larger per-op `fd_step` |
| `frozen_sweep.py` | `sweep(...)` → one `Sweep` (one G, `.matrix()` → `(G, batch_ids)`; `.loss_matrix()` → Δ, each image's loss under each op minus its clean loss), optionally appending `aug_gradient_log.txt` (with `losses` / `clean_loss`, which `sweeps_from_gradient_log` reads back; older logs give NaN Δ). `skip=` ops get NaN rows, magnitudes and losses (`static_pruned`), and every other row is bit-identical to the sweep without it. `sweep_checkpoints` gives one G per checkpoint at identical magnitudes. `Magnitudes.fixed / .relative(strength) / custom draw` |
| `fade.py` | `FADE`, `FADE_RULES`, `faded_ops(ops, fade)`, `check_fade_at`, `fade_images`. The row itself is `grad_collect._fade_row` (`resolve_methods(ops, fade)` picks the method per op). Checked by hand: on a smooth float64 model, the rows equal central differences in `a` to ~1e-6 at a = 1 and 0.5 for every torchvision op kind. Before tangents and `strength_sign` (2026-10-12), `fade=None` with fixed magnitudes gave G bit-identical to the pre-fade code on all three torchvision hosts. Now the enhance ops' rows differ (tangent instead of finite differences), and so does any row whose magnitude is below `identity_at` (sign). No test yet |
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
| `residual.py` | new, the NMS arm's R: `residualize` (each op over its median \|g\|, then the OLS residual on the other ops' mean \|Z\| per image, leaving the op out of its own factor), `residual_rank_corr` (Spearman of those residuals), `null_max_bound(A, N)` (the 95% union-bound chance ceiling of R's largest cell, FACTS D8). Leaving each op out does not remove the closure pull: on synthetic independent ops the mean cell sits about where R_scalenorm's closure null does |
| `diagnostics.py` | new: `diagnose(R, names, n_images)` → `Diagnostics`: effective rank, participation ratio, condition number, per-op VIF, Marchenko-Pastur edges and `n_signal` (eigenvalues above them). Reads only R's finite part |
| `analyze.py` | `analyze(G, batch_ids, names)` / `analyze_sweep(sweep)` → `Analysis`: R_raw, R_scalenorm, loadings, bootstrap, `survives` (A×A), `withheld` (below `n_min`, or the >0.9 shared-factor alarm), `aliases`, `static_pruned` (the run's pruned ops, set aside like unprobed ones even when measured, their Δ too; `analyze_sweep` reads them off the sweep), `diagnostics`, `ci_lo` (the bootstrap CIs' lower bounds scattered back to A×A, NaN where untested; `Redundancy.ci_lo` carries it to the selectors), and with `shrinkage=` `R_shrunk` (which then becomes `primary`). With `residual_rank=` (needs `bootstrap=False`, no shrinkage) `R_residual` and `residual_slopes`, and `R_residual` becomes `primary`. `analyze_sweep` passes the sweep's Δ as `d_loss`, so every Analysis has `loss_increase` (per-op mean Δ). `mean_offdiag` is the mean off-diagonal cell of `primary` |
| `logs.py` | `corr_matrix_log.json`, `corr_bootstrap_log.txt`, `corr_redundancy_log.txt` in Sensaug's record shapes (`stale` written empty, `static_pruned` the run's pruned ops; `unprobed`, `aliases`, `R_shrunk`, `shrinkage`, `R_residual`, `residual_slopes`, `loss_increase`, `mean_offdiag`, `diagnostics` and `fdr_method` added) |

Not ported: `merge_rank_buffers` (no DDP), `stack_probe_buffer` (`Sweep.matrix()` does
this) and the TensorBoard panels. The functions themselves keep a Sensaug issue: a NaN
row in G (an `unprobed` op, e.g. `autoaugment.Invert`) makes `shared_image_factor` NaN
in every column. `analyze` avoids it by setting all-NaN rows aside before anything is
computed and listing them in `unprobed`, not `dropped`; on a G without NaN rows it
equals Sensaug's chain. Exact aliases are set aside the same way, before the shared
factor, and listed in `aliases` (the DIFF32 ops have none). A hard selector sees an
alias as exempt, so map it through `aliases` when its fate should follow its
representative's. `dedup`, `shrinkage`, `diagnostics`, BY and Fisher z have no tests
yet; they were checked by hand (see the rows above). `residual` too, on a synthetic G with a
planted triple: the triple's cells stand out, every other cell sits below `null_max_bound`,
and a median-zero or constant row comes back NaN in `dropped`.

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
| `relevance_only.py` | new: top-k by prior, ignoring R. Declines on a flat prior. The split hosts' prior (trivialaugment_diff, randaugment_diff, augmix_diff) is not flat, since an unsplit op has twice a direction's share, so there it keeps Posterize, Solarize, AutoContrast and Equalize first and fills the rest by name; `config.check`'s "flat prior" warning names every host but sensaug and smartsampling_diff |
| `cover.py` | new, naive, two arms of one experiment: `cover_single` and `cover_even`. A cross-op pair whose bootstrap CI lower bound (`Redundancy.ci_lo`, from `Analysis.ci_lo`) is above `tau` (0.3) moves the smaller-prior op, which ends at exactly `(1 - lam) * prior` in both. `cover_single` gives the mass to its partner (the "cover") and leaves every other op alone; `cover_even` spreads it over every op not moved, by prior. Strongest pair first, partners pinned (no chains), ties by name. Stateless: no persistence gate, hysteresis or rollback. `Selection.covers` and the redundancy log record the pairs. `config.check` accepts them only with a TrivialAugment host or sensaug. On sensaug q is per op too (each op's SA levels are scaled by one factor, so the mass `cover_single` moves is drawn at the cover's levels); every op has the same SA allotment there, so the moved op of a pair is the later name (priors within `_PRIOR_RTOL` tie). The SA loop re-selects on every round's pdf, and those picks (pruned ops, covers) are only in the training log (`[redundancy] SA round at iter ...`), not the redundancy log. Both yamls use `lambda_ramp: constant`: under the linear ramp lam reaches 1 only at the last emission, so nothing would ever be pruned during training. A declined emission leaves the last q in place (the hook's non-SA path). train.sbatch's `LAM` (default 0.25) overrides the yaml's lam 1: submit with `LAM=1`. `tests/selection/test_cover.py` |
| `nms.py` | new, the ungated arm: `NMS(tau=0.5)` (Cohen's "large" r), greedy NMS over the ops not yet pruned, visited by loss increase (`score=`, `Analysis.loss_increase`), highest first. An op whose R with a kept op is above `tau` is pruned (q = 0). R is `R_residual`, read through `pairwise(R, "signed")`, so within-op pairs and anti-correlated pairs never suppress. No budget. `lam` must be 1 (it prunes). The prune is latched: `pruned=` (the last M) stays pruned and out of the ranking, so M only grows. `run` → `(Selection, NmsResult)` (who suppressed whom, at what r); `prune_weights(rel, M)` re-applies M to a new prior. Needs a score, so it runs through `PruneHook`, never `decide` / `ApplyWeightsHook` (both refuse it) |
| `decide.py` | `decide(analysis, rel, selector, role=...)`: the training-loop policy in one place. A control probe or a `withheld` Analysis declines, and otherwise the selector runs on `redundancy_of(analysis)` (primary R + FDR survivors). Used by `tasks/cls/callbacks.py`; the mmseg `ApplyWeightsHook` predates it and inlines the same steps. Also `publication(analysis, role=, mode=, mask_within_op=)` → `PUBLISH` / `KEEP` / `CLEAR`: Sensaug's `prune_augmentations` rule for the R an SA loop holds (control, skipped or all-NaN R keeps the last one; the shared-factor alarm or an unusable `compute_red` clears it) |

A budgeted hard selector takes `budget` (measured ops kept) or is built with `lam`, which
derives the budget the way Sensaug did (`nms` has no budget: `tau` decides). Checked by hand against the submodule, with
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

## `src/madaug/`: MADAug's policy

MADAug (Hou, Zhang & Zhou, "When to Learn What: Model-Adaptive Data Augmentation
Curriculum", ICCV 2023, code github.com/JackHck/MADAug) picks each training image's ops
and levels with a policy network on the task model's feature, trained online against
the task model's validation loss after a virtual step on the policy's augmentations.
MADAug is classification-only; here it trains mmseg segmentors (the user's choice,
2026-10-25). This package is the math, in torch with no framework; the sampling is the
host (`hosts/madaug.py`), and the per-step glue is `tasks/mmseg/madaug_hooks.py`. No
tests.

| file | what |
|---|---|
| `policy.py` | `Projection` (MADAug's `networks/projection.py`: Linear → 2 × 17, or hidden ReLU layers), `predict` (levels = sigmoid, p = softmax(· / T)), `eval_mode`, `policy_levels` (the policy on every probe image, in order) |
| `relax.py` | `relaxed_batch`: MADAug's `explore`, every op at the policy's level (no δ, random sign) blended by p, each through MADAug's straight-through `stop_gradient` (d pixel / d λ = 1 in the model's input space); the labels blended too (`soft_cross_entropy`, summed over pixels / their count, mmseg's default), photometric ops and Identity sharing the unmoved label |
| `hypergrad.py` | `policy_gradient(model, theta, train_loss, val_loss, lr, mode=)`: ∇_θ L_val(w − α ∇_w L_tr(w, θ)) through `torch.func.functional_call`. `"exact"`: −α ∇_θ (∇_w L_tr · v), v = ∇_ŵ L_val, by double backward. `"finite_diff"`: DARTS' central difference, ε = 0.01 / ‖v‖. Checked by hand on a float64 toy segmentor (scratchpad): exact equals a brute-force unroll with ŵ(θ) in the graph to 0.0, finite_diff is within 1e-4 of it, and the model's weights and `.grad` are untouched |
| `search.py` | `policy_step` (Algorithm 1 lines 8–10: relax at T = 1, return the policy's gradient) and `exploit` (line 11: the policy at the host's T, `host.observe`, `host.draw`) |

`MADAugPolicyHook` (ABOVE_NORMAL, after HostProgressHook) runs both before every step:
every `search_freq` steps an update on the step's clean batch and one batch of the val
dataloader (shuffled, endless), the model in eval and called functionally on the unwrapped
module, the gradient averaged over ranks, then Adam; then the draw, installed on the
preprocessor (`set_policy_slots`), which applies it in `train_step` and raises if a batch
arrives without one. The policy reads the segmentor's deepest `extract_feat` output,
average-pooled (`backend.pooled_features`). It and its Adam state go into every checkpoint
(`madaug_policy`), and `scripts/probe.py` reads it from there (`policy_from_checkpoint`).
Checked on stand-ins for mmengine and mmseg only (scratchpad): updates on the right steps
in both modes, the model's train mode and `.grad` untouched, draws applied once with the
labels moving, checkpoint round trip, probe levels in probe order.

What differs from MADAug, and why:

- **The policy learns.** The released `main.sh` runs `main_higher.py` without
  `--bi_epochs` (default 300) for 240 / 200 epochs, and the update needs `epoch >
  bi_epochs`, so the released runs never update the policy (it stays a random linear layer
  on the features). Here it updates from the first step, as the paper's Algorithm 1 says
  (the user's choice).
- **Single ops, not pairs.** The paper's Eq. 6 relaxes pairs (17 × 16 images per input);
  the code relaxes the 17 single ops, and so does this.
- **The virtual step is plain SGD at the task's lr** (Eq. 5), not `higher` around the task
  optimizer (SGD with momentum there; AdamW for several mmseg backbones here). Adam on the
  policy is insensitive to the gradient's scale.
- **The curriculum draws each image** with p(t), as §3.1 describes; the code augments the
  first `int(rate · B)` rows of each batch with `rate = min(tanh(epoch / 40) + 0.01, 1)`,
  which at a segmentation batch of 2 rounds to 0, 1 or 2 images.
- **The policy reads the current model, in its input space.** The code reads a `deepcopy`
  refreshed at each policy update, on un-normalized [0, 1] images (its model trains on
  normalized ones).
- **Segmentation:** the backbone's pooled feature for the penultimate classifier feature;
  the relaxed label blended; the policy's validation batches from the val split (the one
  mIoU is reported on, as Sensaug's SA loop scores ops on it), not a held-out part of
  train. MADAug's Cutout(16) after the policy (`--cutout`) is not added: the dataset's
  pipeline is the base augmentation.
- **Defaults** are main.sh's for reduced CIFAR-10: k = 2, δ = 0.3, T = 3, `search_freq` 3
  (the paper's Table 5 found 1 best), Adam lr 1e-3, weight decay 1e-3, no hidden layer.

## `src/hosts/faster_autoaugment/`: Faster AutoAugment's search

Faster AutoAugment (Hataya, Zdenek, Yoshizoe & Nakayama, ECCV 2020, arXiv 1911.06987;
code github.com/moskomule/dda) searches a policy by backprop against a WGAN-GP critic,
then trains on it. Neither the paper nor dda publishes a learned policy, so dropaug runs
the search, once per dataset, before training (`scripts/faa_search.py`,
`slurm/faa_search.sbatch`). The paper and dda are classification-only; here the search
targets mmseg segmentation (the user's choice, 2026-10-25), adapted as AutoAlbument
(albumentations-team/autoalbument) adapted it. No tests.

| file | what |
|---|---|
| `host.py` | the host (numpy; see `src/hosts/`). `load_policy` checks a policy file (format, version, ops against vocab `faster_aa`, shapes, each stage's w a distribution, p in [0, 1], μ in [0, scale]); `initial_policy` is dda's starting point made deterministic |
| `policy.py` | dda's `Policy` / `SubPolicy` / `SubPolicyStage` / `_Operation`, search mode only, over the 26 ops: each op gated by a RelaxedBernoulli(p) per image, each stage the softmax(w / 0.05)-weighted sum of every op, each sub-policy run on one of `num_chunks` chunks. μ is `clamp(mu, 0, 1)` times dda's `magnitude_scale` (`faster_aa.POLICY_SCALES`). Solarize, Posterize and Cutout get dda's straight-through gradient in μ, AutoContrast and Equalize in the image. A soft mask (K + 1 channels, the last ignore) goes through the same sums: the geometric ops warp it bilinearly (out-of-frame mass to ignore), HorizontalFlip flips it, every other op leaves it. `export()` → the host's arrays |
| `search.py` | `search` / `search_step`: dda's `AdvTrainer.wgan_loss` (critic: task × 0.1 − critic(real) + critic(policy(a)) + 10 · GP; then the policy: task × 0.1 − critic(policy(a)), the critic frozen), Adam 1e-3, betas (0, 0.999). `SegmentorCritic`: one segmentor as task model and critic, dda's `Discriminator` perceptron on its deepest feature, average pooled. `soft_mask`, `soft_cross_entropy` (per image, over the non-ignore mass). `save_policy` (atomic `.npz`, read back). Writes `epoch_<N>.npz`, `final.npz`, `search_log.jsonl`, `search_config.json` |

Defaults (`configs/host/faster_autoaugment.yaml`, `host.search`): the paper's L = 10,
K = 2, 20 epochs, a 4000-image subset (capped at the split; Cityscapes has 2975), η = λ =
0.05, ε = 0.1, GP 10; batch 8 (half real, half augmented), `num_chunks` 4, the search
segmentor `mmseg_segformer_b0` with the dataset's recipe and its pretrained backbone. One
GPU, one process. Kept from dda and flagged: the gradient penalty's norm is over the
channel dimension per pixel (`grad.norm(2, dim=1)`), not the whole image; real images are
the batch's second half. Differs from dda: the RelaxedBernoulli temperature is the paper's
0.05 (dda's code 0.1); the op set is the paper's 17 split into 26 (dda's has 15 with
TranslateY twice); the task loss is a soft-target CE on a blended mask (AutoAlbument uses
BCE on one-hot masks). Checked on CPU only, with a toy segmentor (scratchpad): two epochs
run, the losses are finite, the policy moves, the blended mask sums to 1 per pixel, the
policy file loads into the host; the host's q tilt holds every stage's mass and its draws
match `sampling_probs()` over 4×10⁴ images. Never run on mmseg (not in `sensAug`).
`scripts/smoke_faster_autoaugment.py` (run by `slurm/test.sbatch`) is the GPU version of
that check: search steps on a toy segmentor, the host on the policy they write,
`augment_batch`, and the probe at the host's magnitudes (passed on CPU; not yet run on a GPU).

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
| `robustness.py` | A `Condition` is clean, one op at one level, or one corruption at one severity. `suite(vocab)` is test_robust's set: clean, every op at strengths 0.2 to 1.0, 7 ImageNet-C corruptions, and combination. `run(...)` appends one record per condition to a JSONL file and resumes. One file holds one checkpoint, and it refuses to reuse a condition measured differently. With `corrupted_loader=` (scripts/eval.py passes one) the on-the-fly corruptions run in the DataLoader's workers, as test_robust's pipeline transform did, instead of one image at a time in the main process; same pixels, same records. Also `SegMetric` (mmseg's mIoU/aAcc/mAcc, from per-class counts like mmseg's `IoUMetric`; a K×K CUDA `bincount` is slow past 1000 bins, i.e. on ADE20K), `ClsMetric`, `summarize`, `write_csv` |
| `corruptions.py` | Pre-rendered `ImageNetC`, `CifarC`, `Cityscapes`/`CityscapesC` readers; on-the-fly `corrupt` (the `imagecorruptions` package), Sensaug's `combination` (cv2), and `mce` with AlexNet's baseline. `ImageCorruption(name, severity, seed)` is one condition as `(uint8 RGB image, position) -> image`, seeded by position; `corruption_perturbation` and a backend's `corrupted_dataset` both apply it |
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
`import tasks` needs no framework. `tasks/augment.py` (training draws) and `tasks/magnitudes.py` are shared by both
backends. `tasks/magnitudes.py` gives the probe's magnitudes for a host,
`probe_magnitudes(host, snapshot, mode, fallback)`:
- `"fixed"` gives `probe.magnitude`;
- a host with `draw_bins` (randaugment, trivialaugment, autoaugment, and the five `*_diff`
  hosts) gives its own bins,
  through `vocab.magnitude_bins`, mirrored about the identity when flipped (source
  `"method"`, mode the host);
- a policy host (madaug): the policy's own level on each probe image (`levels`, which the
  caller computes over the probe set with the live or saved policy,
  `madaug_hooks.probe_set_levels`), δ-perturbed and signed as MADAug draws
  (`policy_magnitudes`; source `"method"`); without a policy, `probe.magnitude`;
- a searched-policy host (faster_autoaugment, `draw_magnitudes`): per image, one of each
  op's searched magnitudes, weighted by how often the policy applies it (`drawn_magnitudes`;
  source `"method"`);
- otherwise it is the SA snapshot (sensaug) or `probe.magnitude`.

`scripts/probe.py` uses it too. Everything next to a backend is glue: it gates and
calls `probe` / `stats` / `selection` / `hosts`, and computes nothing itself. No tests
yet. The mmseg pieces were checked against stand-ins for mmseg, never against mmseg
itself. The cls pieces were checked end to end on a torchvision resnet18 and a toy
ImageFolder, but never with timm (not in `sensAug`).

`tasks/mmseg/` (needs mmseg + mmengine; load it with
`custom_imports = dict(imports=["tasks.mmseg.register"])` and `src/` on `PYTHONPATH`):

| file | what |
|---|---|
| `backend.py` | `MMSegBackend(cfg, model=None, split="val")`, `.from_runner(runner)`. `to_unit` runs the model's preprocessor with `training=True` (mmseg pads the gt only there) with the train aug suspended, then de-normalizes. `loss` is `model.loss` per image, summed over its `loss*` keys, with the op's label swapped in and restored. `predict` is the segmentor's `inference` resized to the label. `corrupted_dataset(dataset, corrupt_image)` applies a corruption to each packed image as it is read (after Resize, before the preprocessor), for `run`'s `corrupted_loader`; an image the preprocessor pads is corrupted before its pad there, after it on the unit batch. Refuses a random val pipeline, BGR input, and labels of mixed native sizes (use batch size 1). `load_ckpt` scopes the torch ≥2.6 `weights_only=False` patch to the one load |
| `hooks.py` | `CollectGradientHook` (NORMAL): on `host.probe_role(runner.iter + 1)`, `probe.sweep` → `runner.probe_sweep`; saves/verifies `probe_set.json`. Its `magnitude_mode` (default `"mode"`, Sensaug's) takes each op's magnitude from the SA loop's last snapshot (`runner.corr_magnitudes`), falling back to `magnitude`. A host with `draw_bins` is probed at its own bins instead, except under `"fixed"` (`tasks.magnitudes`). `fade` / `fade_at` come from the host's `fade:` block (`config.py`) and go to `probe.sweep`. The host's `pruned()` ops are skipped (`static_pruned` NaN rows) unless `probe_pruned` (`prune.probe`), and `ApplyWeightsHook` sets them aside in R either way. Compile-checked only: no mmengine in `sensAug`. `ApplyWeightsHook` (LOW): `stats.analyze_sweep` → logs → `build_selector(type, lam=ramp_lambda(...))` → `host.set_weights(q)`. With `RobustValLoop` on an SA host, it only publishes R (`selection.publication`, rank 0's call broadcast), and the loop calls `reweight()` on every round's fresh pdf at Sensaug's λ, `ramp(target, (iter + 1) / max_iters)` (`GradCorrValLoop`). With no R held, or a declined selection, `reweight()` calls `host.clear_weights()`, so the round trains on its pdf as generated, as in Sensaug; the last q is never carried over. Without an SA loop, control probes, lam=0, a withheld R and a declined selection leave the host alone. Every rank sweeps; rank 0's q is broadcast. Checked (a scratchpad harness, no test yet) against Sensaug's real `_downweight_mrmr` / `reweight` / `compute_red` round by round (mRMR and soft at λ 0, 0.25, 1; withheld, skipped and FDR-empty emissions; both pdf generators): equal to 1e-12, same ops pruned. `PruneHook` (LOW) takes its place for selector `nms` (`config.py` picks it): `stats.analyze_sweep(residual_rank=True, bootstrap=False, n_min=0)` with the host's `pruned()` ops set aside, as ApplyWeightsHook does, → logs → `NMS.run` (rank 0's M broadcast) → `host.set_weights` with M zeroed, or with an SA loop `reweight()` on every fresh pdf (M empty: `clear_weights`). No gate: control probes log only, and two health checks warn (`null_max_bound` ≥ tau, \|mean off-diagonal\| > 3/√(N−1)). M and its history go to `pruned_ops.json` at every active probe and are reloaded on resume (a non-SA host gets q at once, an SA host at its first round). The NMS record goes to `corr_redundancy_log.txt`. Checked on stand-ins for mmengine (scratchpad, no test): uniform and SA-driven sensaug hosts, control probe, resume. `HostProgressHook` (VERY_HIGH): `host.set_step(runner.iter + 1)` before every step, for a host with `set_step` (smartsampling_diff's P, madaug's p(t)); `config.py` adds it |
| `transforms.py` | `GpuAugSegDataPreProcessor(host=dict(type=...), augment=True)`: Sensaug's `GpuAugSegDataPreProcessor`, batched on GPU after collation, same arithmetic as the backend. It owns the host (`.host`) the hooks read. Sampling is Sensaug's: the SensAug pdf with q (+N(0, 0.1) level jitter), else `sampling_probs()` / `none_prob` at a uniform level. DIFF32 ops with the `sensaug` / `uniform` hosts, or a `tasks.augment.BIN_HOSTS` host (each op from `sampling_probs()` / `none_prob`, at the magnitude of its `draw_bins` bin read through `vocab.magnitude_bins`, in the op's units; `tests/tasks/test_augment.py`): `trivialaugment_diff` (one op per image), `randaugment_diff` (`num_ops` rounds), `augmix_diff` (`MIX_HOSTS`: chains mixed with the clean image, the label left as it is), `smartsampling_diff` (`SLOT_HOSTS`: `host.draw_slots`, 2 ops at one level) or `autoaugment_diff` (`SLOT_HOSTS`: a sub-policy's two steps); or the `madaug` host (`tasks.augment.POLICY_HOSTS`: the slots `MADAugPolicyHook` installed for the batch, `apply_slots`, refused if none were), or the `faster_autoaugment` host (`tasks.augment.STAGE_HOSTS`: `host.draw_stages`, the stages applied in turn); other hosts raise (`check_trainable`) unless `augment=False`. The eval slot (`set_eval(op, level)` / `set_eval_none()`, Sensaug's `set_eval_spec`) applies one op to every image on the eval path, for the SA loop |
| `loops.py` | Sensaug's loops. `RobustValLoop` (val_cfg) is the SA loop: after `warmup_rounds`, every round scores each op at each curve level on `ratio` of val through the eval slot, builds the pdf (`sa.generate_pdf_new` with `descending_MA`, or `uniform` / `weighted_augs`) and calls `host.set_sa_pdf`, then `ApplyWeightsHook.reweight`, then publishes `runner.corr_magnitudes`. It re-searches the curve (`sa.sa_curve`, via `SubsetTestLoop`) every `SA_CURVE_CADENCE` = 6 rounds. Writes `sa_curve_log.txt`, `perturb_eval.txt`, `corr_magnitudes.json`. Refuses `warmup_rounds` / val_interval that differ from the host's. Also `RobustBaselineValLoop`, `SubsetTestLoop` (test_cfg, with KID), `SubsetValLoop`, `RobustIterBasedTrainLoop` (train_cfg, `init_sa`). The module docstring has the config Sensaug's train.py set |
| `madaug_hooks.py` | `MADAugPolicyHook(search_freq=3, lr=, weight_decay=, n_layers=, n_hidden=, hypergrad="exact" \| "finite_diff", fd_eps=, split="val")`, the host yaml's `policy:` block, added by `config.py` for a policy host with augment on (see `src/madaug/`). Publishes `runner.madaug_policy`, which `CollectGradientHook` runs over the probe set for a madaug host's magnitudes. `probe_set_levels`, `policy_from_checkpoint`. Compile-checked and run on stand-ins only |
| `register.py` | imports `hooks`, `loops`, `madaug_hooks` and `transforms` so the registries see them |
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

Sensaug's `--pruned-augmentations` is the `prune:` block (see `src/hosts/`): out of the
curve, every pdf, the warmup policy and (by default) the probe, as in Sensaug. The
host's q is not saved in checkpoints, so a resumed run trains on the base distribution
until the next active probe (as in Sensaug). The torchvision geometric ops refuse a
label whose size differs from the image's, which mmseg val pipelines produce on
datasets whose val resize isn't the identity (not Cityscapes).

`tasks/cls/` (torch + torchvision; timm only to build a timm model):

| file | what |
|---|---|
| `backend.py` | `ClsBackend(model, data=, mean=, std=, label_smoothing=0)`. `model` is an nn.Module or `dict(library="timm" \| "torchvision", name=, **kwargs)`. The batch is `dropaug.eval`'s item collated (`image` in [0, 1], `target`, `id`); `to_unit` returns label None and refuses images outside [0, 1]; `loss` is per-image CE; mean/std default to timm's `pretrained_cfg`, else ImageNet's. `load_ckpt` takes `state_dict` / `model` / `state_dict_ema` or a bare state dict, strips `module.`, and loads strictly. `ImageFolderItems` (resize the short side + center crop) and `CifarItems` (never downloads) give ids that match `ImageNetC` / `CifarC` |
| `callbacks.py` | `CollectGradientCallback(backend, host, ..., magnitude_mode=, fade=, fade_at=)` / `ApplyWeightsCallback(host, selector=..., **analyze kwargs)`: the mmseg hooks for a hand-written loop, `before_run(work_dir, max_iters, resume=)` then `apply.after_train_iter(it, collect.after_train_iter(it))`. Same files, same DDP handling (rank 0 writes and broadcasts q). The q policy is `selection.decide`. Applying the host's draws to training images is the loop's job; there is no cls train-time augmentation |

## `slurm/`: jobs on UMD Nexus (gamma)

Ported from Sensaug's `job_scripts/` (Della is no longer used). The checkout is assumed
to be at `/fs/gamma-projects/pruneaug` (`REPO_DIR`). The `#SBATCH --output` lines
hard-code `slurm/logs/` under it, and that directory must exist before you submit.
Every script reads its knobs from env vars, which are documented in each script's header.

**The job cap (`slurm/gate.sh`, the hard rule at the top of this file).** In flight there
is at most one A6000 job or two A5000 jobs, never a mix. A job costs 2 (A6000) or 1
(A5000) of a budget of 2, whatever its GPU count; an array costs all of it. A job's GPU
count is the one in its script's `#SBATCH --gres=gpu:<type>:N` line (train and matrix 4,
probe 1, at most 4); the gate sets the type. It is enforced twice. At submission,
`gate_sbatch` (used by `submit.sh` and `matrix.sbatch --submit`) reads `squeue` for this
repo's train/probe/matrix jobs (told apart by command path under `slurm/`, so other
repos' jobs are not counted) and makes the new job wait (`afterany`) for every in-flight
job older than the ones that fit beside it. It refuses `--gres`, `-G`, `--gpus*` and
`--dependency` in extra sbatch flags, and fails rather than guess if `squeue` fails.
Submissions are serialized on `slurm/logs/.gate.lock`. At job start, `gate_check_running`
(the train, probe, faa_search and test bodies) refuses to run if the running jobs already exceed the cap, which
catches a bare `sbatch`; that check comes after the allocation, so a bypass still holds
its GPUs for a few seconds. A matrix array is submitted with `%1` (A6000) or `%2`
(A5000), and `MAX_PARALLEL` is ignored. `eval.sbatch` is neither gated nor counted. A
job whose gres has no rtxa6000/rtxa5000 type, or more than 4 GPUs, counts as more than
the whole budget. The queue is stricter than it needs to be: a new job waits for every
older in-flight job except the ones that fit beside it, so a slot can idle while one of
two jobs outlives the other. `tests/slurm/test_gate.py` runs the real bash against
stand-in `squeue` / `sbatch`.
The entry points they call are `train.py`, `scripts/probe.py` and `scripts/eval.py`; each
one's docstring documents its flags. `scripts/probe.py` probes a dropaug run
(`--work-dir`, rebuilt from its `dropaug.yaml`, with its own host's ops) or any mmseg
checkpoint (`--backbone`/`--task`, `--dataset`, `--method`): the model is the task +
recipe for that pair, loaded strictly, and the ops, magnitudes and fade are the
`--method` host config's, whatever the model was trained with. `--dry-run` prints them
without building a model. Only the sensaug method reads an SA snapshot
(`corr_magnitudes.json`, by default next to the checkpoint).

| file | what |
|---|---|
| `train.sbatch` | `[host] [dataset] [selector] [seed]`, 4 GPUs, torchrun. From `train_nexus_gamma.sbatch`. Selector `none` is the λ=0 control; `nms` defaults `LAM` to 1 (it refuses anything else). The run dir is `experiments/<host>_<selector>-l<λ>_<dataset>_<backbone>_s<seed>`, and the script refuses to resume an existing run unless `ALLOW_RESUME=1`. With host `sensaug` it passes `--descending-MA --no-inv-aug`, as Sensaug's sbatch did on every run (`DESCENDING_MA=0` / `INV_AUG=1` turn them off). The static prune: `PRUNE_OPS="a b"`, `PRUNE_RANDOM=K`, `PRUNE_SEED`, `HOLD_NONE_PROB=0`, `PROBE_PRUNED=1` (train.py's `--pruned-augmentations` / `--random-prune-count` / `--random-prune-seed` / `--no-hold-none-prob` / `--probe-pruned`), and the run dir gets `prune_tag`'s suffix. Afterwards it queues `eval.sbatch` as a separate 1-GPU job |
| `probe.sbatch` | `<exp_dir>`: a post-hoc probe of one checkpoint of a dropaug run (`CKPT=best\|last\|path`) into `<exp_dir>/probe/<ckpt>[_<PROBE_TAG>]`. Or `<checkpoint.pth> <backbone> <dataset> <method>`: any checkpoint, probed with that method's ops, into `<ckpt dir>/probe/<ckpt>_<method>[_<PROBE_TAG>]` (`CLUSTER`, `DATA_ROOT`, `MAGNITUDES_PATH`). It refuses a directory that already holds a gradient log, because two Gs in one file would pool |
| `faa_search.sbatch` | `[dataset]`, 1 GPU: Faster AutoAugment's policy search (`scripts/faa_search.py`) into `<WORK_DIR>/faa_search_<dataset>[_<SEARCH_TAG>]/` (`final.npz`, `epoch_<N>.npz`, `search_log.jsonl`). `SEARCH_ARGS` for flags and Hydra overrides (`host.search.*`). Gated like the others. `train.sbatch faster_autoaugment <dataset>` reads its `final.npz` by default (`FAA_POLICY` overrides) and refuses to start without one |
| `eval.sbatch` | `<exp_dir>`: `dropaug.eval` on one checkpoint → `<exp_dir>/eval/<ckpt>.jsonl`. Resumes on resubmit |
| `test.sbatch` | 1 GPU, gated like a probe: `pytest -x` (`PYTEST_ARGS`, default the vocab, hosts, tasks, probe and config tests), then `scripts/smoke_diff_hosts.py` (randaugment_diff at cifar10's level, augmix_diff, smartsampling_diff: a batch augmented on the GPU, every op probed at the host's own bins, faded as its yaml fades) and `scripts/smoke_faster_autoaugment.py` |
| `matrix.sbatch` | array over `HOSTS` × `DATASETS` × `SELECTORS` × `SEEDS`, one 4-GPU `train.sbatch` per task, at `%1` (A6000) or `%2` (`GPU=a5000`). `bash slurm/matrix.sbatch` lists the grid; `--submit` submits it through the gate; `POINTS=3,17` reruns chosen indices |
| `gate.sh` | the job cap: `gate_sbatch`, `gate_deps`, `gate_check_running`, `gate_array_throttle` (sourced by `common.sh`, `submit.sh`, `matrix.sbatch`) |
| `submit.sh` | `[GPU=a5000] bash slurm/submit.sh <name> <script> [args]`: sbatch with the name prefixed to the job name, through the gate. `eval.sbatch` passes through ungated |
| `common.sh` | sourced by every sbatch script: log grouping into `slurm/logs/<job-name>/`, conda/module setup, checkpoint resolution, and `gate.sh` |

## Parked ideas (not started, low priority)

**Shard the probe sweep across DDP ranks.** Today every rank sweeps the whole probe set
(`tasks/mmseg/hooks.py` docstring; Sensaug's sharding and gather are not ported), so on a
4-GPU job three GPUs repeat the first one's work, and a probe is a large share of a run's
wall time. Sharding is the way to make probing cheaper without changing what it measures.
It must not change G, and a naive version would in three places:

- **Magnitudes.** `frozen_sweep.sweep` seeds one generator per sweep and advances it across
  batches (`rng = default_rng(probe_seed)`, one `_draw` per batch). Fixed and relative
  magnitudes are constants, so they are safe. A host with `draw_bins` (trivialaugment_diff,
  randaugment, trivialaugment, autoaugment) draws a random bin per image, so a rank that
  starts its own generator would give its first image the draw image 0 gets today. Draw the
  whole probe set's magnitudes up front and slice them per rank, or advance each rank's
  generator past the images before its slice.
- **Padding.** Do not shard with a `DistributedSampler`: it pads to equal sizes by repeating
  images, and a duplicated column changes R. Slice `probe_set.chunks(batch_size)` explicitly.
- **Order.** Pearson r ignores image order, but the cluster bootstrap resamples batches by
  index under a fixed seed. Gather back into ProbeSet order (keeping batches as separate
  entries), which is also the order of `aug_gradient_log.txt`.

Op randomness is not a problem: `apply_op` reseeds `torch.manual_seed(op_seed(probe_seed,
name))` on every call, so at batch size 1 an image gets the same draw on any rank. Expect
float tolerance, not bit equality (some CUDA backward kernels are not deterministic).
Acceptance test, as sbatch jobs: `scripts/probe.py` on one checkpoint unsharded and
sharded; magnitudes and ids equal, G within tolerance, R equal, bootstrap CIs identical.
Put it in `tests/probe/` once it exists.

**Probe at every val eval** (`probe.schedule.interval: ${task.val_interval}`; on sensaug,
`corr_cadence=1`). Not worth it on its own: R moves slowly between the probes the current
runs have, so adjacent probes mostly repeat each other, and each costs far more than a val
eval. Do it after sharding, and in two narrower forms: the early probes as `control_iters`
(logged, never a q: a model that early is not the one the selector should act on), and a
persistence gate for `cover`, which is stateless by design and sits on its `tau` threshold
(in the VOC trivialaugment_diff run the same pair cleared it on some probes and not
others). On sensaug the probe magnitudes follow the SA snapshot, which shifts at each curve
re-search, so R between rounds would mix model drift with a magnitude change; separate
those first. Measure R's probe-to-probe stability before relying on either claim: so far it
is a correlation of R's off-diagonal between consecutive probes, not of the selected pairs,
and it is not in FACTS.md.

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
