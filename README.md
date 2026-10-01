# dropaug

**WIP.** A gradient-based probe for data augmentations, lifted out of
[SensAug](https://github.com/RibaDiba/SensAug) so it runs on any model, any
augmentation method and any vocabulary of ops.

For every op and every probe image, the probe measures `d loss / d magnitude` on a
frozen model, giving G (ops × images). Correlating G's rows across images gives R:
*do these two augmentations hurt the same images?* A selector turns R into weights q
over the ops, and an augmentation method (a *host*) samples with q.

```
vocab ──► probe ──► G ──► stats ──► R ──► selection ──► q ──► hosts
 ops       sweep          analyze        mRMR, KL tilt,        RandAugment, SensAug,
                          (+ FDR)        DPP, QPFS, ...        TrivialAugment, ...
                                   dropaug.eval scores the trained models
```

## Layout (`src/`)

| package | what |
|---|---|
| `vocab/` | what an augmentation is: `OpSpec`, the op families, `build_vocab`, `audit` |
| `probe/` | measure G: the probe set, per-image gradients, finite differences, sweeps |
| `stats/` | G → R: correlation, alias merging, shrinkage, bootstrap + FDR, red(a), diagnostics, Sensaug-format logs |
| `selection/` | R → q: the selectors, and `decide`, the training-loop policy |
| `hosts/` | one adapter per augmentation method (sampling policy only) |
| `tasks/` | one backend per framework: `mmseg/` (segmentation), `cls/` (classification) |
| `dropaug/eval/` | score trained models: op-matched robustness, corruptions, control vs pruned |
| `dropaug/io/` | dropaug's G / R / q / eval artifacts, with versioned headers and hashes |

`slurm/` holds the UMD Nexus job scripts. **CLAUDE.md** documents each package and
the invariants the probe carries over from Sensaug. **FACTS.md** is the ledger that
every reported number traces back to.

## Probing a trained classifier

```python
import probe, stats
from vocab import build_vocab
from tasks.cls import ClsBackend
from dropaug.io import write_R

backend = ClsBackend(dict(library="timm", name="resnet50"),
                     data=dict(type="imagefolder", root=VAL_ROOT))
backend.build_model()
backend.load_ckpt(CKPT)
data = backend.probe_data()
probe_set = probe.ProbeSet.build(data.ids, 2000)
ops = build_vocab()
g = probe.sweep(backend, probe_set, data.dataset, ops, probe.Magnitudes.relative(0.5), batch_size=32)
analysis = stats.analyze_sweep(g)
write_R("R.json", analysis, ops)
```

Paths come from configs, never from code. For segmentation, see `tasks/mmseg/__init__.py`.

## Setup

```bash
git submodule update --init          # third_party/Sensaug, for the parity tests
conda activate sensAug               # no dropaug env yet
pytest tests/
```
