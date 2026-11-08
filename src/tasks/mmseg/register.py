"""Import this module to put dropaug's mmseg pieces into the MMEngine registries.

    custom_imports = dict(imports=["tasks.mmseg.register"], allow_failed_imports=False)

``src/`` must be on the path (``PYTHONPATH=src``). Registers:

    HOOKS   CollectGradientHook, ApplyWeightsHook,  (mmengine's root registry, which
            PruneHook, HostProgressHook,            mmseg's scope falls back to)
            MADAugPolicyHook, AdvAAControllerHook,
            DeepAASearchHook, AdaAugPolicyHook
    MODELS  GpuAugSegDataPreProcessor               (mmseg's, as Sensaug registered it)
    LOOPS   RobustValLoop, RobustBaselineValLoop,   (mmseg's, as Sensaug registered them)
            SubsetTestLoop, SubsetValLoop,
            RobustIterBasedTrainLoop
"""

from . import adaaug_hooks, advaa_hooks, deepaa_hooks, hooks, loops, madaug_hooks, transforms  # noqa: F401
