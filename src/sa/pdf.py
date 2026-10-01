"""The SA round's mIoU table -> the training pdf over ``(op, level)`` pairs.

Ported from Sensaug's ``RobustValLoop`` (``sensaug/loops/sensaug_loop.py``):
``generate_pdf_new``, ``generate_pdf_new_weighted_aug``, ``generate_uniform_pdf``,
``_none_prob_denominator``, ``_remove_H_perturbations`` and ``dict_mean``. The pdf
arithmetic is unchanged. What moved out is the loop's own work: running
``test_perturbed_new`` to get the table, Lever 3's reweighting (``_apply_redundancy
_reweighting``, which dropaug's host and ApplyWeightsHook do), and storing the pdf.

``miou_record`` is ``{op: {level: miou}}``, the round-eval's table. Every generator
returns ``{(op, level): prob, ("none", 0): prob}`` summing to 1, which is what
``hosts.SensAugHost.set_sa_pdf`` takes.

``descending_MA`` is Sensaug's flag, with Sensaug's behaviour. Each op's levels are
sorted by mIoU with ``reverse = not descending_MA`` and rank i gets
``betabinom.pmf(i, n, 0.75, 1.0)``, which decreases with i. So with
``descending_MA=True`` the sort is ascending and the level with the LOWEST mIoU, the
one that hurts the current model most, gets the most mass. With False the mildest
level does. (Sensaug's help text and docs describe True the other way round.)

Two quirks kept as Sensaug had them:

- ``generate_pdf_new`` evaluates the pmf over ``n`` levels at ``0..n-1`` only, one
  short of its support, so each op's block sums to under its allotment and the
  shortfall goes to ``("none", 0)``. With the SA curve's 4 levels per op, 15.8% of
  every allotment leaks, so P(none) is 0.185 at 30 ops, not ``1/31``, and 0.234 at 10.
- ``generate_pdf_new_weighted_aug`` truncates the same way over all (op, level) pairs:
  P(none) is 0.0559 at 30 ops x 4 levels and 0.0675 at 10.

These depend only on the op and level counts (computed, not quoted from Sensaug,
whose comments say 16% and 6.4% / 5.5%).
"""

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from scipy.stats import betabinom

__all__ = [
    "NONE_KEY",
    "REMOVE_H_NAMES",
    "dict_mean",
    "generate_pdf_new",
    "generate_pdf_new_weighted_aug",
    "generate_uniform_pdf",
    "level_metrics",
    "none_prob_denominator",
    "remove_H_perturbations",
]

#: The pdf's no-op entry (``hosts.sensaug.NONE_KEY``).
NONE_KEY = ("none", 0)

#: The ops ``--no-inv-aug`` (``remove_H``) drops from the DIFF32 vocabulary.
REMOVE_H_NAMES = ("lighter_H", "darker_H")

MiouRecord = Mapping[str, Mapping[float, float]]
Pdf = Dict[Tuple[str, float], float]


def dict_mean(dict_list: Sequence[Mapping]) -> dict:
    """Key-wise mean of a list of metric dicts (an op's metrics across its levels)."""
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def remove_H_perturbations(miou_record: dict, names: Iterable[str] = REMOVE_H_NAMES) -> None:
    """Drop the ``--no-inv-aug`` ops from the table, in place. A no-op on ops already
    absent: the SA curve normally excludes them at its source."""
    for name in names:
        miou_record.pop(name, None)


def none_prob_denominator(
    n_surviving: int, n_static_excluded: int = 0, hold_none_prob: bool = False
) -> int:
    """The op count the ``("none", 0)`` mass is computed against.

    By default the surviving count. Under ``hold_none_prob`` it also counts the ops
    ``pruned_augmentations`` removed, so a static prune moves mass between ops
    without changing how often any augmentation fires. This fixes the allotment, not
    P(none) itself (see the module docstring on the truncated pmf)."""
    return n_surviving + (n_static_excluded if hold_none_prob else 0)


def generate_uniform_pdf(miou_record: MiouRecord, denom: int) -> Pdf:
    """Uniform over ops and, within an op, over its levels (Sensaug's ``--uniform``).
    ``denom`` is ``none_prob_denominator(len(miou_record), ...)``."""
    pdf_dict = {}
    n_surviving = len(miou_record.keys())

    for perturbation, levels in sorted(miou_record.items()):
        num_levels = len(levels.keys())
        for level, _ in levels.items():
            # One fused division, so with denom == n_surviving this is bit-identical
            # to the original 1 / ((N + 1) * num_levels).
            pdf_dict[(perturbation, level)] = denom / float(
                (denom + 1) * n_surviving * num_levels
            )

    pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

    # add a "none" perturbation to pdf
    pdf_dict[NONE_KEY] = 1.0 - pdf_dict_perturb_prob
    return pdf_dict


def generate_pdf_new(miou_record: MiouRecord, descending_MA: bool, denom: int) -> Pdf:
    """Each op gets an equal allotment, shared out between its levels by their mIoU
    rank (Sensaug's default generator). ``denom`` is ``none_prob_denominator(...)``."""
    pdf_dict = {}

    num_perturbations = len(miou_record.keys())
    perturbation_total_prob = denom / float(denom + 1)
    perturbation_uniform_prob = (
        perturbation_total_prob / num_perturbations if num_perturbations else 0.0
    )

    for perturbation, levels in sorted(miou_record.items()):

        def lambda_bb(x):
            return betabinom.pmf(x, len(levels), 0.75, 1.0)

        # iterate through order of decreasing miou
        reverse = not descending_MA
        for i, (level, _) in enumerate(
            sorted(levels.items(), key=lambda x: x[1], reverse=reverse)
        ):
            pdf_dict[(perturbation, level)] = lambda_bb(i) * perturbation_uniform_prob

    pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

    # add a "none" perturbation
    pdf_dict[NONE_KEY] = 1.0 - pdf_dict_perturb_prob
    return pdf_dict


def generate_pdf_new_weighted_aug(miou_record: MiouRecord, descending_MA: bool) -> Pdf:
    """``generate_pdf_new`` without the equal allotment per op: every (op, level) pair
    is ranked by mIoU together, so the ranking also moves mass between ops
    (Sensaug's ``--weighted-augs``). P(none) is not held by ``hold_none_prob``."""
    pdf_dict = {}
    reverse = not descending_MA

    all_mious = {}
    for perturbation, levels in sorted(miou_record.items()):
        for i, (level, miou) in enumerate(
            sorted(levels.items(), key=lambda x: x[1], reverse=reverse)
        ):
            all_mious[(perturbation, level)] = miou

    for i, ((perturbation, level), miou) in enumerate(
        sorted(all_mious.items(), key=lambda x: x[1], reverse=reverse)
    ):

        def lambda_bb(x):
            return betabinom.pmf(x, len(all_mious), 0.75, 1.0)

        pdf_dict[(perturbation, level)] = lambda_bb(i) * 0.95

    pdf_dict_perturb_prob = sum(list(pdf_dict.values()))

    # add a "none" perturbation (see the module docstring on the truncated pmf)
    pdf_dict[NONE_KEY] = 1.0 - pdf_dict_perturb_prob
    return pdf_dict


def level_metrics(metrics_record: Mapping[str, List[Mapping]]) -> dict:
    """``{op: [metrics per level]}`` -> the flat ``{"<op without _>_<metric>": mean}``
    record Sensaug's ``test_perturbed_new`` logged to ``perturb_eval.txt``."""
    final_metrics = {}
    for p_type, per_level in metrics_record.items():
        for metric, value in dict_mean(per_level).items():
            final_metrics[p_type.replace("_", "") + f"_{metric}"] = value
    return final_metrics
