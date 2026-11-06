"""Deep AutoAugment's policy search, with no training framework (Zheng, Zhang, Yan &
Zhang, ICLR 2022, arXiv 2203.06172; code ``third_party/DeepAA``, ``DeepAA_search.py``).

DeepAA trains a network first (``pretrain()``, its default augmentation only), freezes
it, then searches a K-layer policy one layer at a time by regularized gradient matching:
each layer's categorical over the (op, level) transforms moves toward the transforms whose
gradients bring the augmented training gradient into line with the validation gradient.
Here the frozen network is the run's own model at the probe-and-prune pause (the user's
choice, 2026-11-05: train, pause, prune, search, resume), and the search runs over the
transforms the prune left.

    space.py   the 139 transforms as vocab ``deepaa`` draws; ``kept_transforms`` (DeepAA's
               ``available_policies`` after a prune)
    policy.py  ``LayerPolicy`` (DeepAA's identity-initialized logits over the kept
               transforms); ``export`` / ``save_npz`` (DeepAA's own ``.npz``)
    search.py  ``Task`` (the frozen model: weighted gradients, JVPs), ``image_importance``,
               ``Search`` (the layers' Adam, resumable), ``run_step``, ``pass_counts``

The host is ``hosts/deep_autoaugment.py`` (``install_policy``); the glue is
``tasks/mmseg/deepaa_hooks.py``.
"""

from .policy import LayerPolicy, export, save_npz
from .search import MODES, Search, Settings, Task, image_importance, pass_counts, run_step
from .space import NUM_TRANSFORMS, draws, kept_transforms, transform_label

__all__ = [
    "LayerPolicy",
    "MODES",
    "NUM_TRANSFORMS",
    "Search",
    "Settings",
    "Task",
    "draws",
    "export",
    "image_importance",
    "kept_transforms",
    "pass_counts",
    "run_step",
    "save_npz",
    "transform_label",
]
