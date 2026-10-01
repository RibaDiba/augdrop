"""Image-classification backend (timm, torchvision, or any logits model).

    backend.py     ClsBackend: build_model / load_ckpt / probe_data, and the probe's (and
                   dropaug.eval's) adapter; ImageFolderItems / CifarItems, clean readers
                   whose items and ids match dropaug.eval's ImageNetC / CifarC
    callbacks.py   CollectGradientCallback, ApplyWeightsCallback: the mmseg hooks as
                   plain callbacks for a hand-written training loop

No framework beyond torch/torchvision; timm is imported only to build a timm model.

Post hoc::

    backend = ClsBackend(dict(library="timm", name="resnet50"),
                         data=dict(type="imagefolder", root=val_root))
    backend.build_model()
    backend.load_ckpt("model_best.pth.tar")
    data = backend.probe_data()
    probe_set = ProbeSet.build(data.ids, 2000)
    g = sweep(backend, probe_set, data.dataset, build_vocab(), Magnitudes.relative(0.5),
              batch_size=32)
    analysis = stats.analyze_sweep(g)
"""

from .backend import ClsBackend, CifarItems, ImageFolderItems, build_dataset
from .callbacks import ApplyWeightsCallback, CollectGradientCallback

__all__ = [
    "ApplyWeightsCallback",
    "ClsBackend",
    "CifarItems",
    "CollectGradientCallback",
    "ImageFolderItems",
    "build_dataset",
]
