"""GPU smoke test of Faster AutoAugment, search to probe (slurm/test.sbatch).

1. The policy search (``hosts/faster_autoaugment/search.py``) for a few steps on a toy
   segmentor: the losses stay finite, the policy moves, and the policy file it writes
   loads into the host.
2. Training draws (``tasks.augment.augment_batch``) from that host on the GPU: shapes,
   the [0, 1] range, finite output, and the label moving with the geometric ops.
3. The probe on a toy eval-mode model through ``probe.collect_grads`` at the host's own
   magnitudes (``draw_magnitudes``), faded as configs/host/faster_autoaugment.yaml fades:
   a finite row for every op but SamplePairing, which is ``unprobed`` (NaN).

No mmseg: scripts/faa_search.py is the mmseg search, and needs a cluster run.

    python scripts/smoke_faster_autoaugment.py [--batch 8] [--size 96] [--seed 0]
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hosts import build_host  # noqa: E402
from hosts.faster_autoaugment.policy import Policy  # noqa: E402
from hosts.faster_autoaugment.search import SearchConfig, SegmentorCritic, search  # noqa: E402
from probe import collect_grads, resolve_methods  # noqa: E402
from tasks.augment import augment_batch, check_trainable  # noqa: E402
from vocab import build_vocab  # noqa: E402

CLASSES = 5
FADED = ("HorizontalFlip", "Solarize", "Posterize", "Invert", "AutoContrast", "Equalize", "Cutout")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class ToySegmentor(nn.Module):
    """``extract_feat`` + ``decode_head``, as ``SegmentorCritic`` reads an mmseg segmentor."""

    align_corners = False

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.BatchNorm2d(8), nn.ReLU())
        self.deep = nn.Sequential(nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU())
        self.decode_head = _Head(16)

    def extract_feat(self, x):
        shallow = self.stem(x)
        return [shallow, self.deep(shallow)]


class _Head(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cls = nn.Conv2d(channels, CLASSES, 1)

    def forward(self, feats):
        return self.cls(feats[-1])


class Adapter:
    """The probe's adapter for a toy classifier on the probe batch."""

    def __init__(self, model):
        self.model = model

    def to_unit(self, batch):
        return batch["image"], batch["label"]

    def loss(self, images01, label, batch):
        logits = F.interpolate(self.model(images01), size=label.shape[-2:], mode="bilinear", align_corners=False)
        return F.cross_entropy(logits, label, ignore_index=255, reduction="none").mean(dim=(1, 2))


class ToyDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, CLASSES, 1))

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    check(torch.cuda.is_available(), "no GPU visible: this smoke test is for a GPU job")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    print(f"device {torch.cuda.get_device_name(0)}, torch {torch.__version__}", flush=True)

    # 1. the search
    model = SegmentorCritic(ToySegmentor(), 16).to(device)
    policy = Policy(num_chunks=2).to(device)
    before = [q.detach().clone() for q in policy.parameters()]
    g = torch.Generator(device="cpu").manual_seed(args.seed)

    def batches(epoch):
        for _ in range(3):
            yield (
                torch.rand(args.batch, 3, args.size, args.size, generator=g).to(device),
                torch.randint(0, CLASSES, (args.batch, args.size, args.size), generator=g).to(device),
            )

    with tempfile.TemporaryDirectory() as out:
        search(
            model,
            policy,
            batches,
            normalize=lambda x: (x - 0.5) / 0.25,
            num_classes=CLASSES,
            config=SearchConfig(epochs=2, log_interval=3),
            out_dir=out,
            log=lambda line: print(f"[search] {line}", flush=True),
        )
        moved = sum(bool((a != b).any()) for a, b in zip(before, policy.parameters()))
        check(moved > 0, "the search moved no policy parameter")
        host = build_host("faster_autoaugment", policy_file=os.path.join(out, "final.npz"))
    print(f"[search] ok: {moved} of {len(before)} policy tensors moved; host none_prob {host.none_prob:.3f}", flush=True)

    # 2. training draws
    ops = build_vocab({"families": ["faster_aa"]})
    check(tuple(ops) == host.vocabulary(), "the host's vocabulary is not vocab faster_aa's")
    check_trainable(host, ops)
    images = torch.rand(args.batch, 3, args.size, args.size, generator=g).to(device)
    labels = torch.randint(0, CLASSES, (args.batch, args.size, args.size), generator=g).to(device)
    rng = np.random.default_rng(args.seed)
    label_moved = False
    for step in range(5):
        out, out_labels = augment_batch(host, images, labels, ops, rng)
        check(out.shape == images.shape and out.dtype == images.dtype, "shape/dtype changed")
        check(torch.isfinite(out).all().item(), f"non-finite pixels at step {step}")
        check(float(out.min()) >= -1e-4 and float(out.max()) <= 1 + 1e-4, f"left [0, 1] at step {step}")
        check(out_labels.shape == labels.shape, "label shape changed")
        label_moved |= not torch.equal(out_labels, labels)
    print(f"[train] augment_batch ok x5, label moved = {label_moved}", flush=True)

    # 3. the probe
    dense = ToyDense().to(device).eval()
    magnitudes = host.draw_magnitudes(args.batch, np.random.default_rng(args.seed))
    methods = resolve_methods(ops, [f"faster_aa.{o}" for o in FADED])
    batch = {"image": images, "label": labels}
    rows = collect_grads(Adapter(dense), batch, ops, magnitudes, methods=methods, check=True)
    bad = [n for n, r in rows.items() if methods[n] != "unprobed" and not np.isfinite(r).all()]
    check(not bad, f"non-finite G rows: {bad}")
    check(np.isnan(rows["faster_aa.SamplePairing"]).all(), "SamplePairing must be unprobed")
    count = {}
    for m in methods.values():
        count[m] = count.get(m, 0) + 1
    print(f"[probe] ok: {len(rows)} rows, methods {count}", flush=True)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
