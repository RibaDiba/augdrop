"""MMSegmentation backend (SegFormer and every other mmseg segmentor).

    backend.py     MMSegBackend: build_model / load_ckpt / probe_data, and the probe's
                   (and dropaug.eval's) adapter over an mmseg segmentor
    hooks.py       CollectGradientHook, ApplyWeightsHook, PruneHook (the nms arm):
                   mmengine glue around probe, stats and selection
    transforms.py  GpuAugSegDataPreProcessor: the host's draws applied on GPU after
                   collation (Sensaug's DIFF32 ops only, for now)
    config.py      build(): dropaug's composed config -> the mmengine Config of a run
    register.py    imports hooks and transforms so the registries see them

Importing this package imports no mmseg; ``backend`` loads it on first use, and
``hooks`` / ``transforms`` / ``register`` need it at import.

Post hoc (Sensaug's scripts/compute_grad_corr.py)::

    backend = MMSegBackend("work_dir/segformer.py")
    backend.build_model()
    backend.load_ckpt("work_dir/best_mIoU_iter_160000.pth")
    data = backend.probe_data()
    probe_set = ProbeSet.build(data.ids)
    g = sweep(backend, probe_set, data.dataset, build_vocab(), Magnitudes.relative(0.5))
    analysis = stats.analyze_sweep(g)
"""

from .backend import MMSegBackend
from .config import build as build_config

__all__ = ["MMSegBackend", "build_config"]
