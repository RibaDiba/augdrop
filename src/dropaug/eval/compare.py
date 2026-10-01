"""Control vs pruned: diff tables over robustness results.

A pruned run is judged against its control on the same conditions:
``robustness.run`` over the control's whole vocabulary, including the ops the pruned
run dropped. Cells are matched on (dataset, kind, condition, level_kind, level), and
both arms must have measured the same thing there: the same number of images and the
same op-unit magnitude. A mismatch raises, because a table across different eval sets
or magnitudes compares nothing.

    control = [load_results(p) for p in control_paths]    # one run per seed
    pruned = [load_results(p) for p in pruned_paths]
    cells = diff_table(control, pruned, pruned_ops=["noise", "blur"])
    write_csv("diff_levels.csv", cells)
    print(markdown(group_table(condition_table(cells))))

Each arm may hold several runs (seeds). A cell's value is the mean over those runs,
with the sample std and the run count beside it. ``delta`` is pruned minus control of
those means, so a negative delta means pruning cost robustness (every metric here is
higher-is-better). Runs are not paired: seed k of one arm is not assumed to match seed k
of the other. With one run per arm, delta is a single difference with no error bar.
Sensaug's effects were small, so read it that way.

Groups:
    clean       the unperturbed eval
    pruned      an op the pruned run trained without
    kept        an op both arms trained on
    unseen_op   an op outside ``trained_ops`` (only when that is given)
    corruption  a corruption, on the fly or pre-rendered, that neither arm trained on
"""

import logging
import math
import os
import re
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .robustness import latest, load_results, write_csv

logger = logging.getLogger(__name__)

__all__ = [
    "GROUPS",
    "condition_table",
    "diff_table",
    "group_table",
    "load_results",
    "markdown",
    "sensaug_prune_history",
    "write_csv",
]

GROUPS = ("clean", "kept", "pruned", "unseen_op", "corruption")

Run = Sequence[Mapping]


def diff_table(
    control,
    pruned,
    *,
    pruned_ops: Optional[Iterable[str]] = None,
    trained_ops: Optional[Iterable[str]] = None,
    metric: Optional[str] = None,
) -> list[dict]:
    """One row per cell (dataset, kind, condition, level_kind, level), control's order
    first. ``control`` and ``pruned`` are each one run (a list of records) or a list of
    runs.

    ``pruned_ops`` defaults to ``meta["pruned_ops"]`` on the pruned run's records.
    ``trained_ops``, the control's training vocabulary, separates ``kept`` from
    ``unseen_op``. Without it, every op is taken to be one the control trained on.
    ``metric`` defaults to the records' primary metric. ``missing`` names the arm a cell
    is absent from, if either.
    """
    control_runs, pruned_runs = _runs(control, "control"), _runs(pruned, "pruned")
    pruned_set = set(_pruned_ops_from_meta(pruned_runs) if pruned_ops is None else pruned_ops)
    trained = None if trained_ops is None else set(trained_ops)
    cells = {"control": _cells(control_runs), "pruned": _cells(pruned_runs)}

    keys = list(cells["control"]) + [key for key in cells["pruned"] if key not in cells["control"]]
    rows = []
    for key in keys:
        c, p = cells["control"].get(key, []), cells["pruned"].get(key, [])
        both = c + p
        _check_same_measurement(key, both)
        first = both[0]
        metric_name = metric or _primary(key, both)
        control_values, pruned_values = _values(c, metric_name, key), _values(p, metric_name, key)
        control_mean, pruned_mean = _mean(control_values), _mean(pruned_values)
        info = first.get("info", {})
        dataset, kind, name, level_kind, level = key
        rows.append(
            {
                "dataset": dataset,
                "kind": kind,
                "condition": name,
                "group": _group(kind, name, pruned_set, trained),
                "category": info.get("category"),
                "family": info.get("family"),
                "level_kind": level_kind,
                "level": level,
                "magnitude": info.get("magnitude"),
                "metric": metric_name,
                "control": control_mean,
                "control_std": _std(control_values),
                "n_control": len(control_values),
                "pruned": pruned_mean,
                "pruned_std": _std(pruned_values),
                "n_pruned": len(pruned_values),
                "delta": None if control_mean is None or pruned_mean is None else pruned_mean - control_mean,
                "missing": "control" if not c else "pruned" if not p else "",
            }
        )

    evaluated = {row["condition"] for row in rows if row["kind"] == "op"}
    unevaluated = sorted(pruned_set - evaluated)
    if unevaluated:
        logger.warning(
            "pruned op(s) %s were never evaluated; the comparison a pruned run is judged on is missing",
            unevaluated,
        )
    return rows


def condition_table(rows: Iterable[Mapping]) -> list[dict]:
    """``diff_table`` rows averaged over levels: one row per (dataset, kind, condition,
    level_kind). Only cells both arms measured count toward the means. ``incomplete``
    counts the ones left out."""
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["kind"], row["condition"], row["level_kind"]), []).append(row)
    out = []
    for (dataset, kind, name, level_kind), cells in groups.items():
        complete = [cell for cell in cells if cell["delta"] is not None]
        out.append(
            {
                "dataset": dataset,
                "kind": kind,
                "condition": name,
                "group": cells[0]["group"],
                "category": cells[0]["category"],
                "family": cells[0]["family"],
                "level_kind": level_kind,
                "metric": cells[0]["metric"],
                "n_levels": len(complete),
                "incomplete": len(cells) - len(complete),
                "control": _mean([cell["control"] for cell in complete]),
                "pruned": _mean([cell["pruned"] for cell in complete]),
                "delta": _mean([cell["delta"] for cell in complete]),
            }
        )
    return out


def group_table(condition_rows: Iterable[Mapping]) -> list[dict]:
    """``condition_table`` rows averaged per (dataset, group), every condition weighted
    equally, groups in ``GROUPS`` order."""
    groups: dict[tuple, list] = {}
    for row in condition_rows:
        if row["delta"] is not None:
            groups.setdefault((row["dataset"], row["group"]), []).append(row)
    datasets = list(dict.fromkeys(dataset for dataset, _ in groups))
    out = []
    for dataset in datasets:
        for group in GROUPS:
            rows = groups.get((dataset, group))
            if not rows:
                continue
            out.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "metric": rows[0]["metric"],
                    "n_conditions": len(rows),
                    "control": _mean([row["control"] for row in rows]),
                    "pruned": _mean([row["pruned"] for row in rows]),
                    "delta": _mean([row["delta"] for row in rows]),
                }
            )
    return out


def markdown(rows: Sequence[Mapping], columns: Optional[Sequence[str]] = None, floatfmt: str = ".2f") -> str:
    """``rows`` as a GitHub markdown table. Columns default to the first row's keys."""
    if not rows:
        return ""
    columns = list(columns or rows[0])

    def cell(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return "nan" if math.isnan(value) else format(value, floatfmt)
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join(lines)


_PRUNE_LINE = re.compile(r"\[redundancy\] mRMR pruned \d+/\d+ ops at .*?\): (.*)$")


def sensaug_prune_history(work_dir: str) -> tuple[dict, Optional[set]]:
    """``({op: rounds pruned}, ops pruned at the last logged round)`` from the
    ``[redundancy] mRMR pruned ...`` lines of a Sensaug run's logs
    (``<work_dir>/*/*.log``), as eval_all_perturbations.py read them. ``({}, None)`` for
    a run that never logged one (every non-mRMR arm). The last set is the usual
    ``pruned_ops`` for a Sensaug mRMR run."""
    counts: dict[str, int] = {}
    last = None
    for timestamp in sorted(os.listdir(work_dir)):
        folder = os.path.join(work_dir, timestamp)
        if not os.path.isdir(folder):
            continue
        for file in sorted(os.listdir(folder)):
            if not file.endswith(".log"):
                continue
            with open(os.path.join(folder, file), errors="replace") as f:
                for line in f:
                    match = _PRUNE_LINE.search(line.rstrip())
                    if match:
                        last = {op.strip() for op in match.group(1).split(",") if op.strip()}
                        for op in last:
                            counts[op] = counts.get(op, 0) + 1
    return counts, last


def _runs(arm, label: str) -> list[Run]:
    arm = list(arm)
    if not arm:
        raise ValueError(f"the {label} arm holds no runs")
    runs = [arm] if isinstance(arm[0], Mapping) else [list(run) for run in arm]
    for run in runs:
        checkpoints = {record.get("checkpoint") for record in run}
        if len(checkpoints) > 1:
            raise ValueError(
                f"one {label} run holds several checkpoints {sorted(map(str, checkpoints))}; "
                f"a run is one results file"
            )
    return runs


def _cells(runs: Sequence[Run]) -> dict:
    cells: dict[tuple, list] = {}
    for run in runs:
        for key, record in latest(run).items():
            cells.setdefault(key, []).append(record)
    return cells


def _pruned_ops_from_meta(runs: Sequence[Run]) -> list:
    found = {
        tuple(sorted(record["meta"]["pruned_ops"]))
        for run in runs
        for record in run
        if "pruned_ops" in (record.get("meta") or {})
    }
    if not found:
        raise ValueError("no pruned_ops given and none recorded in the pruned run's meta")
    if len(found) > 1:
        raise ValueError(f"the pruned runs record different pruned_ops: {sorted(found)}")
    return list(found.pop())


def _check_same_measurement(key, records):
    counts = {record["n_images"] for record in records}
    if len(counts) > 1:
        raise ValueError(f"{key} was scored on different numbers of images {sorted(counts)}")
    magnitudes = {(record.get("info") or {}).get("magnitude") for record in records}
    if len(magnitudes) > 1:
        raise ValueError(f"{key} was measured at different magnitudes {sorted(map(str, magnitudes))}")


def _primary(key, records):
    primaries = {record.get("primary") for record in records}
    if len(primaries) > 1:
        raise ValueError(f"{key}: the runs disagree on the primary metric {sorted(map(str, primaries))}")
    return primaries.pop()


def _values(records, metric, key):
    values = []
    for record in records:
        if metric not in record["metrics"]:
            raise KeyError(f"{key} has no metric {metric!r}; it has {sorted(record['metrics'])}")
        if record["metrics"][metric] is not None:
            values.append(float(record["metrics"][metric]))
    return values


def _group(kind, name, pruned, trained):
    if kind == "clean":
        return "clean"
    if kind != "op":
        return "corruption"
    if name in pruned:
        return "pruned"
    if trained is not None and name not in trained:
        return "unseen_op"
    return "kept"


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None

