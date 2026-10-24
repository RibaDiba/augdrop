"""GPU smoke test of the randaugment_diff and augmix_diff pipelines (slurm/test.sbatch).

For each host: build its vocab and host as configs/host/<host>.yaml does, augment a
batch through ``tasks.augment.augment_batch`` on the GPU (what the mmseg preprocessor
calls), then probe every op on a toy eval-mode model through ``probe.collect_grads`` at
the magnitudes the host's own ``draw_bins`` give, faded as the yaml fades. Checks shapes,
the [0, 1] range, finite output, and a finite G row for every probed op. A segmentation
label goes through the geometric ops of randaugment_diff (it must move) and is left
alone by augmix_diff (as in Sensaug).

    python scripts/smoke_diff_hosts.py [--batch 8] [--size 96] [--seed 0]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hosts import build_host  # noqa: E402
from probe import collect_grads, resolve_methods  # noqa: E402
from tasks.augment import augment_batch, check_trainable  # noqa: E402
from vocab import build_vocab, magnitude_bins  # noqa: E402

FADED = ("Solarize", "Posterize", "AutoContrast", "Equalize")

HOSTS = {
    "randaugment_diff": dict(family="randaug_diff", kwargs=dict(dataset="cifar10")),
    "augmix_diff": dict(family="augmix_diff", kwargs={}),
}


class Toy(nn.Module):
    def __init__(self, classes=5):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, classes))

    def forward(self, x):
        return self.net(x)


class Adapter:
    def __init__(self, model):
        self.model = model

    def to_unit(self, batch):
        return batch["image"], None

    def loss(self, images01, label, batch):
        return F.cross_entropy(self.model(images01), batch["target"], reduction="none")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def smoke(name, spec, args, device):
    vocab = build_vocab({"families": [spec["family"]], "exclude": []})
    host = build_host(name, **spec["kwargs"])
    check(list(host.vocabulary()) == [n for n in vocab if n in host.vocabulary()], f"{name}: host names are not in vocab order")
    check(set(host.vocabulary()) <= set(vocab), f"{name}: host ops missing from the vocab")
    ops = {n: vocab[n] for n in host.vocabulary()}
    check_trainable(host, ops)
    print(f"[{name}] {len(ops)} ops, num_magnitude_bins={host.num_magnitude_bins}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    images = torch.rand(args.batch, 3, args.size, args.size, generator=g).to(device)
    labels = torch.randint(0, 5, (args.batch, args.size, args.size), generator=g).to(device)
    rng = np.random.default_rng(args.seed)

    for step in range(5):
        out, out_labels = augment_batch(host, images, labels, ops, rng)
        check(out.shape == images.shape and out.dtype == images.dtype, f"{name}: shape/dtype changed")
        check(torch.isfinite(out).all().item(), f"{name}: non-finite pixels at step {step}")
        check(float(out.min()) >= -1e-4 and float(out.max()) <= 1 + 1e-4, f"{name}: left [0,1]: {float(out.min())}..{float(out.max())}")
        check(out_labels.shape == labels.shape, f"{name}: label shape changed")
    changed = float((out - images).abs().mean())
    check(changed > 0, f"{name}: augmentation changed nothing")
    if name == "augmix_diff":
        check(torch.equal(out_labels, labels), "augmix_diff must leave the mask as it is")
    print(f"[{name}] augment_batch ok x5, mean |dx| = {changed:.4f}, label moved = {not torch.equal(out_labels, labels)}", flush=True)

    # probe: every op at its host's own bin, on an eval-mode model
    model = Toy().to(device).eval()
    bins, flips = host.draw_bins(args.batch, np.random.default_rng(args.seed))
    magnitudes = {}
    for op_name, op in ops.items():
        if not op.has_magnitude:
            continue
        values, signed = magnitude_bins(op_name, host.num_magnitude_bins)
        magnitudes[op_name] = np.asarray([values[int(b)] for b in bins[op_name]], dtype=np.float64)
    fade = [f"{spec['family']}.{o}" for o in FADED]
    methods = resolve_methods(ops, fade)
    batch = {"image": images, "target": torch.randint(0, 5, (args.batch,), generator=g).to(device)}
    rows = collect_grads(Adapter(model), batch, ops, magnitudes, methods=methods, check=True)
    bad = [n for n, r in rows.items() if methods[n] != "unprobed" and not np.isfinite(r).all()]
    check(not bad, f"{name}: non-finite G rows: {bad}")
    count = {}
    for m in methods.values():
        count[m] = count.get(m, 0) + 1
    zero = [n for n, r in rows.items() if np.isfinite(r).all() and not np.any(r)]
    print(f"[{name}] probe ok: {len(rows)} rows, methods {count}, all-zero rows: {zero}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    check(torch.cuda.is_available(), "no GPU visible: this smoke test is for a GPU job")
    device = torch.device("cuda")
    print(f"device {torch.cuda.get_device_name(0)}, torch {torch.__version__}", flush=True)
    for name, spec in HOSTS.items():
        smoke(name, spec, args, device)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
