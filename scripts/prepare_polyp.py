"""Prepare the PraNet polyp benchmark in mmseg's img_dir / ann_dir layout.

PraNet protocol (Fan et al., MICCAI 2020), from the README's TrainDataset / TestDataset:

    train   TrainDataset              1450  (900 Kvasir-SEG + 550 CVC-ClinicDB)
    val     Kvasir (100) + CVC-ClinicDB (62) of TestDataset            162
    zero-shot, never trained on:
            CVC-ColonDB 380, ETIS-LaribPolypDB 196, CVC-300 60

The raw Kvasir-SEG 880/120 split is not used.

    python scripts/prepare_polyp.py [--raw data/raw/polyp_pranet] [--out data/polyp]
                                    [--zeroshot-out data/polyp_zeroshot]

Output (images are re-encoded as lossless PNG; masks are {0, 1} uint8 PNG):

    <out>/{img_dir,ann_dir}/{train,val}/*.png
    <zeroshot-out>/<name>/{img_dir,ann_dir}/val/*.png     name: colondb | etis | cvc300
    <out>/manifest.json                                    sha256 of every source and output file

Masks are binarized at > 127 after printing the unique values of each set. It refuses a
mask whose size differs from its image, a missing partner, and any count other than the above.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

TRAIN_COUNT = 1450
# (key in TestDataset, output name, expected count)
VAL_SETS = [("Kvasir", "kvasir", 100), ("CVC-ClinicDB", "clinicdb", 62)]
VAL_COUNT = 162
ZEROSHOT_SETS = [
    ("CVC-ColonDB", "colondb", 380),
    ("ETIS-LaribPolypDB", "etis", 196),
    ("CVC-300", "cvc300", 60),
]
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dir(root, *names):
    """The first existing directory root/<name>, searched at most 3 levels down."""
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath[len(root):].count(os.sep) > 3:
            dirnames[:] = []
            continue
        for n in names:
            if n in dirnames:
                return os.path.join(dirpath, n)
    raise SystemExit(f"no directory named {' / '.join(names)} under {root}")


def listing(d):
    out = {}
    for f in sorted(os.listdir(d)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXT and not f.startswith("."):
            if stem in out:
                raise SystemExit(f"{d}: two files with stem {stem!r}")
            out[stem] = os.path.join(d, f)
    return out


def pairs(img_dir, mask_dir):
    imgs, masks = listing(img_dir), listing(mask_dir)
    if imgs.keys() != masks.keys():
        raise SystemExit(
            f"{img_dir} vs {mask_dir}: images without masks {sorted(imgs.keys() - masks.keys())[:5]}, "
            f"masks without images {sorted(masks.keys() - imgs.keys())[:5]}"
        )
    return [(stem, imgs[stem], masks[stem]) for stem in sorted(imgs)]


def print_unique(label, items):
    seen, ndtypes, modes = {}, set(), set()
    for _, _, m in items:
        with Image.open(m) as im:
            modes.add(im.mode)
            a = np.asarray(im.convert("L") if im.mode in ("1", "P") else im)
        ndtypes.add(str(a.dtype) + str(a.shape[2:]))
        for v, c in zip(*np.unique(a, return_counts=True)):
            seen[int(v)] = seen.get(int(v), 0) + int(c)
    vals = sorted(seen)
    shown = vals if len(vals) <= 16 else vals[:8] + ["..."] + vals[-8:]
    print(f"[masks] {label}: {len(items)} files, modes {sorted(modes)}, dtype/extra-dims {sorted(ndtypes)}, "
          f"{len(vals)} unique values {shown}", flush=True)


def binarize(path):
    with Image.open(path) as im:
        # mode "1" arrays are bool (> 127 would be all False) and "P" is palette indices
        a = np.asarray(im.convert("L") if im.mode in ("1", "P") else im)
    if a.ndim == 3:  # RGB(A) mask: any channel
        a = a[..., :3].max(axis=2)
    return (a > 127).astype(np.uint8)


def write_split(label, items, out_root, split, prefix, manifest):
    """Write (stem, image, mask) triples as <out_root>/{img_dir,ann_dir}/<split>/<prefix><stem>.png."""
    img_out = os.path.join(out_root, "img_dir", split)
    ann_out = os.path.join(out_root, "ann_dir", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)
    fg = 0
    for stem, ip, mp in items:
        name = f"{prefix}{stem}.png"
        with Image.open(ip) as im:
            im = im.convert("RGB")
            size = im.size
            m = binarize(mp)
            if (m.shape[1], m.shape[0]) != size:
                raise SystemExit(f"{label}/{stem}: image {size} vs mask {(m.shape[1], m.shape[0])}")
            im.save(os.path.join(img_out, name))
        Image.fromarray(m, mode="L").save(os.path.join(ann_out, name))
        fg += int(m.sum()) > 0
        manifest["files"].append({
            "set": label, "split": split, "name": name,
            "image_src": os.path.relpath(ip), "image_src_sha256": sha256(ip),
            "mask_src": os.path.relpath(mp), "mask_src_sha256": sha256(mp),
            "image_sha256": sha256(os.path.join(img_out, name)),
            "mask_sha256": sha256(os.path.join(ann_out, name)),
        })
    print(f"[write] {label}: {len(items)} -> {out_root} ({split}); {fg} with a foreground pixel", flush=True)
    # a mask that binarized to all zeros (e.g. a bool-valued mode "1" mask) must not pass silently
    assert fg >= 0.95 * len(items), f"{label}: only {fg}/{len(items)} masks have a foreground pixel"
    return len(items)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", default="data/raw/polyp_pranet")
    p.add_argument("--out", default="data/polyp")
    p.add_argument("--zeroshot-out", default="data/polyp_zeroshot")
    p.add_argument("--force", action="store_true", help="replace an existing output")
    args = p.parse_args()

    for d in (args.out, args.zeroshot_out):
        if os.path.exists(d):
            if not args.force:
                raise SystemExit(f"{d} exists; --force to replace it")
            shutil.rmtree(d)

    train_root = find_dir(args.raw, "TrainDataset")
    test_root = find_dir(args.raw, "TestDataset")

    # ---- gather and count before writing anything ----
    train = pairs(find_dir(train_root, "image", "images"), find_dir(train_root, "masks", "mask"))
    val = {}
    for key, tag, n in VAL_SETS:
        d = os.path.join(test_root, key)
        val[tag] = (key, n, pairs(find_dir(d, "images", "image"), find_dir(d, "masks", "mask")))
    zero = {}
    for key, tag, n in ZEROSHOT_SETS:
        d = os.path.join(test_root, key)
        zero[tag] = (key, n, pairs(find_dir(d, "images", "image"), find_dir(d, "masks", "mask")))

    counts = {"train": len(train), **{t: len(v[2]) for t, v in val.items()}, **{t: len(v[2]) for t, v in zero.items()}}
    print("[counts]", counts, flush=True)
    assert len(train) == TRAIN_COUNT, f"train {len(train)} != {TRAIN_COUNT}"
    for tag, (key, n, items) in {**val, **zero}.items():
        assert len(items) == n, f"{key}: {len(items)} != {n}"
    assert sum(len(v[2]) for v in val.values()) == VAL_COUNT

    # ---- unique mask values, before any binarizing ----
    print_unique("train", train)
    for tag, (key, _, items) in {**val, **zero}.items():
        print_unique(key, items)

    # ---- write ----
    manifest = {"protocol": "PraNet", "counts": counts, "files": []}
    write_split("TrainDataset", train, args.out, "train", "", manifest)
    for tag, (key, _, items) in val.items():
        write_split(key, items, args.out, "val", f"{tag}_", manifest)
    for tag, (key, _, items) in zero.items():
        write_split(key, items, os.path.join(args.zeroshot_out, tag), "val", "", manifest)

    # ---- check the output against the contract ----
    for split, want in (("train", TRAIN_COUNT), ("val", VAL_COUNT)):
        for kind in ("img_dir", "ann_dir"):
            got = len(os.listdir(os.path.join(args.out, kind, split)))
            assert got == want, f"{args.out}/{kind}/{split}: {got} != {want}"
    for tag, (_, n, _) in zero.items():
        for kind in ("img_dir", "ann_dir"):
            got = len(os.listdir(os.path.join(args.zeroshot_out, tag, kind, "val")))
            assert got == n, f"{tag}/{kind}: {got} != {n}"
    uniq = set()
    for root, _, files in os.walk(args.out):
        if "ann_dir" in root:
            for f in files:
                uniq |= set(np.unique(np.asarray(Image.open(os.path.join(root, f)))).tolist())
    assert uniq == {0, 1}, f"ann values {uniq}"
    print(f"[check] counts ok; ann values {sorted(uniq)}", flush=True)

    path = os.path.join(args.out, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[manifest] {path} sha256 {sha256(path)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
