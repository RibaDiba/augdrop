"""SCORE trained models: op-matched robustness, common corruptions, control vs pruned.

    robustness.py   run(adapter, loader, conditions, metric, results_path) -> records.
                    Conditions from suite / op_conditions / corruption_conditions /
                    prerendered_conditions; SegMetric, ClsMetric; summarize, write_csv.
                    Replaces Sensaug's test_robust.py, eval_all_perturbations.py and the
                    per-op table of eval_all_augmentations.py.
    corruptions.py  ImageNet-C / CIFAR-C / Cityscapes-C readers, on-the-fly corrupt(),
                    Sensaug's combination, and mce.
    compare.py      diff_table -> condition_table -> group_table: control vs pruned.

Like the probe, this knows no training framework. It talks to an adapter (the probe's
``to_unit`` plus ``predict`` and an optional ``target``; see robustness.py) and to the
vocab's OpSpecs.
"""

from .compare import GROUPS, condition_table, diff_table, group_table, markdown, sensaug_prune_history
from .corruptions import (
    ALEXNET_ERR,
    CATEGORIES,
    COMBINATION,
    IMAGENET_C,
    IMAGENET_C_EXTRA,
    SENSAUG_UNSEEN,
    SEVERITIES,
    CifarC,
    Cityscapes,
    CityscapesC,
    ImageNetC,
    corrupt,
    corruption_perturbation,
    mce,
)
from .robustness import (
    CLEAN,
    DEFAULT_STRENGTHS,
    ClsMetric,
    Condition,
    SegMetric,
    clean,
    corruption_conditions,
    evaluate,
    load_results,
    mce_from_records,
    op_conditions,
    prerendered_conditions,
    run,
    suite,
    summarize,
    write_csv,
)

__all__ = [
    "ALEXNET_ERR",
    "CATEGORIES",
    "CLEAN",
    "COMBINATION",
    "DEFAULT_STRENGTHS",
    "GROUPS",
    "IMAGENET_C",
    "IMAGENET_C_EXTRA",
    "SENSAUG_UNSEEN",
    "SEVERITIES",
    "CifarC",
    "Cityscapes",
    "CityscapesC",
    "ClsMetric",
    "Condition",
    "ImageNetC",
    "SegMetric",
    "clean",
    "condition_table",
    "corrupt",
    "corruption_conditions",
    "corruption_perturbation",
    "diff_table",
    "evaluate",
    "group_table",
    "load_results",
    "markdown",
    "mce",
    "mce_from_records",
    "op_conditions",
    "prerendered_conditions",
    "run",
    "sensaug_prune_history",
    "suite",
    "summarize",
    "write_csv",
]
