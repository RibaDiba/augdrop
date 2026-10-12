#!/usr/bin/env python
"""How much of the SA loop's KID is signal: SubsetTestLoop's estimator vs KID on all images.

    python scripts/kid_noise_check.py --work-dir experiments/ours-kidrgb-A5000-cityscapes \\
        --out experiments/kid_check/cityscapes.json [--levels 0.5 1.0] [--seeds 30] [--perms 200]

No model and no checkpoint: KID compares images only. The images are exactly the ones
SubsetTestLoop hands KID: the first ``sa.test_ratio`` of the run's test dataloader, each
op applied through GpuAugSegDataPreProcessor's eval slot, put back in the loaded form by
``_as_loaded`` and flipped to RGB (``sa.kid_rgb``). Inception features are torchmetrics'
(the 2048-d torch-fidelity InceptionV3 KID uses).

For every (op, level), two estimators of KID(clean, op):

- ``subset4``: SubsetTestLoop's, torchmetrics' ``compute``: ``n // 4`` subsets, each 4
  random clean vs 4 *independently* drawn op images (in training, ``subsets`` was the
  per-rank batch count, 1/4 of the pool on 4 GPUs). Run at ``--seeds`` seeds: the spread
  is how much one SA evaluation's KID would move on a redraw alone.
- ``full``: one subset of all n images per side, the standard KID (deterministic).

Each is compared with its own permutation null: pool the 2n features, split at random
into two halves, re-estimate, ``--perms`` times. An op is "detected" when the estimate is
above the null's 97.5th percentile. ``clean_vs_clean`` repeats Sensaug's situation
(fake = the clean images) for the subset4 estimator, to check it against the logged null.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def subset4(real, fake, n_subsets, gen):
    from torchmetrics.image.kid import poly_mmd

    scores = []
    for _ in range(n_subsets):
        r = real[torch.randperm(len(real), generator=gen)[:4]]
        f = fake[torch.randperm(len(fake), generator=gen)[:4]]
        scores.append(poly_mmd(r, f))
    return float(torch.stack(scores).mean())


def full(real, fake):
    from torchmetrics.image.kid import poly_mmd

    return float(poly_mmd(real, fake))


def perm_null(real, fake, estimator, perms, gen):
    pool = torch.cat([real, fake])
    n = len(real)
    out = []
    for _ in range(perms):
        idx = torch.randperm(len(pool), generator=gen)
        out.append(estimator(pool[idx[:n]], pool[idx[n:]]))
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--levels", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings
    from mmseg.registry import MODELS
    from torchmetrics.image.kid import KernelInceptionDistance

    from dropaug.config import load
    from tasks.mmseg.config import build
    from tasks.mmseg.loops import _as_loaded

    cfg = load(args.work_dir)
    mm = build(cfg, args.work_dir)
    import_modules_from_strings(**mm.custom_imports)
    init_default_scope("mmseg")
    ratio = float(mm.test_cfg.get("ratio", 0.25))
    kid_rgb = bool(mm.test_cfg.get("kid_rgb", True))
    dp = MODELS.build(mm.model.data_preprocessor).eval()
    dl = Runner.build_dataloader(mm.test_dataloader)
    n = int(len(dl) * ratio)
    batches = []
    it = iter(dl)
    for _ in range(n):
        batches.append(next(it))
    ops = list(dp.host.vocabulary())
    print(f"{cfg.dataset.name}: {n} images ({ratio} of {len(dl)}), {len(ops)} ops, levels {args.levels}, kid_rgb {kid_rgb}", flush=True)

    inception = KernelInceptionDistance(feature=2048, normalize=False).inception.eval()

    def features(op, level):
        if op is None:
            dp.set_eval_none()
        else:
            dp.set_eval(op, level)
        feats, buf = [], []
        with torch.no_grad():
            for b in batches:
                out = dp(b, training=False)
                img = _as_loaded(dp, out["inputs"], b["inputs"])
                if kid_rgb:
                    img = img[:, [2, 1, 0]]
                if buf and img.shape[-2:] != buf[0].shape[-2:]:  # mixed sizes (VOC, ADE): flush
                    feats.append(inception(torch.cat(buf)).double()); buf = []
                buf.append(img)
                if len(buf) == args.batch:
                    feats.append(inception(torch.cat(buf)).double()); buf = []
            if buf:
                feats.append(inception(torch.cat(buf)).double())
        dp.set_eval_none()
        return torch.cat(feats)

    t0 = time.time()
    clean = features(None, 0.0)
    print(f"clean features {tuple(clean.shape)} in {time.time() - t0:.0f}s", flush=True)
    S = max(1, n // 4)
    gen = torch.Generator().manual_seed(0)

    est4 = lambda r, f: subset4(r, f, S, gen)
    cvc = np.array([subset4(clean, clean, S, gen) for _ in range(args.seeds)])
    result = {
        "dataset": cfg.dataset.name, "n_images": n, "subset4_subsets": S, "kid_rgb": kid_rgb,
        "seeds": args.seeds, "perms": args.perms, "work_dir": args.work_dir,
        "clean_vs_clean_subset4": {"mean": cvc.mean(), "sd": cvc.std(), "p2.5": np.percentile(cvc, 2.5), "p97.5": np.percentile(cvc, 97.5)},
        "ops": {},
    }
    for op in ops:
        for level in args.levels:
            t = time.time()
            fake = features(op, level)
            draws = np.array([subset4(clean, fake, S, gen) for _ in range(args.seeds)])
            null4 = perm_null(clean, fake, est4, args.perms, gen)
            kfull = full(clean, fake)
            nullf = perm_null(clean, fake, full, args.perms, gen)
            rec = {
                "subset4_mean_of_draws": draws.mean(), "subset4_sd_of_draws": draws.std(),
                "subset4_draws_detected": float(np.mean(draws > np.percentile(null4, 97.5))),
                "subset4_null_p97.5": np.percentile(null4, 97.5), "subset4_null_sd": null4.std(),
                "full": kfull, "full_null_p97.5": np.percentile(nullf, 97.5), "full_null_sd": nullf.std(),
                "full_detected": bool(kfull > np.percentile(nullf, 97.5)),
                "subset4_draws": draws.tolist(),
            }
            result["ops"].setdefault(op, {})[str(level)] = rec
            print(f"{op:16s} {level:.1f}  subset4 {draws.mean():+.4f} ±{draws.std():.4f} (null97.5 {rec['subset4_null_p97.5']:+.4f}, draws detected {rec['subset4_draws_detected']:.0%})"
                  f"  | full {kfull:+.5f} (null97.5 {rec['full_null_p97.5']:+.5f}) {'DETECTED' if rec['full_detected'] else 'not detected'}  [{time.time() - t:.0f}s]", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
