"""dropaug.eval.compare: control vs pruned tables."""

import logging

import numpy as np
import pytest

from dropaug.eval import compare as K
from dropaug.eval import robustness as R
from eval_helpers import SegAdapter, loader, seg_items

vocab = pytest.importorskip("vocab")


def rec(name, value, *, kind="op", level=0.5, checkpoint="ckpt", n=100, magnitude=None, meta=None):
    level_kind = {"clean": "none", "op": "strength", "corruption": "severity", "prerendered": "severity"}[kind]
    return {
        "dataset": "val",
        "checkpoint": checkpoint,
        "kind": kind,
        "condition": name,
        "level_kind": level_kind,
        "level": None if kind == "clean" else level,
        "info": {"magnitude": magnitude if magnitude is not None else (level if kind == "op" else None)},
        "primary": "mIoU",
        "metrics": {"mIoU": value},
        "n_images": n,
        "meta": meta or {},
    }


def arm(values, checkpoint="ckpt", meta=None):
    """{(kind, name, level): value} -> one run."""
    return [rec(name, v, kind=kind, level=level, checkpoint=checkpoint, meta=meta) for (kind, name, level), v in values.items()]


CONTROL = {
    ("clean", "clean", None): 80.0,
    ("op", "blur", 0.5): 70.0,
    ("op", "blur", 1.0): 60.0,
    ("op", "noise", 0.5): 75.0,
    ("corruption", "fog", 1): 65.0,
}
PRUNED = {
    ("clean", "clean", None): 81.0,
    ("op", "blur", 0.5): 66.0,
    ("op", "blur", 1.0): 50.0,
    ("op", "noise", 0.5): 76.0,
    ("corruption", "fog", 1): 64.0,
}


def test_cells_are_grouped_and_differenced():
    rows = K.diff_table(arm(CONTROL), arm(PRUNED), pruned_ops=["blur"])
    by_key = {(r["condition"], r["level"]): r for r in rows}
    assert [r["condition"] for r in rows] == ["clean", "blur", "blur", "noise", "fog"]
    assert by_key[("clean", None)]["group"] == "clean"
    assert by_key[("blur", 1.0)]["group"] == "pruned"
    assert by_key[("blur", 1.0)]["delta"] == pytest.approx(-10.0)
    assert by_key[("noise", 0.5)]["group"] == "kept"
    assert by_key[("fog", 1)]["group"] == "corruption"
    assert all(r["missing"] == "" and r["n_control"] == r["n_pruned"] == 1 for r in rows)
    assert by_key[("blur", 1.0)]["control_std"] is None


def test_condition_and_group_tables_average_levels_then_conditions():
    rows = K.condition_table(K.diff_table(arm(CONTROL), arm(PRUNED), pruned_ops=["blur"]))
    blur = next(r for r in rows if r["condition"] == "blur")
    assert blur["n_levels"] == 2 and blur["delta"] == pytest.approx(-7.0)
    assert blur["control"] == pytest.approx(65.0) and blur["pruned"] == pytest.approx(58.0)
    groups = {r["group"]: r for r in K.group_table(rows)}
    assert list(groups) == ["clean", "kept", "pruned", "corruption"]
    assert groups["pruned"]["delta"] == pytest.approx(-7.0)
    assert groups["kept"]["delta"] == pytest.approx(1.0)
    assert groups["clean"]["n_conditions"] == 1


def test_trained_ops_separates_unseen_ops():
    rows = K.diff_table(arm(CONTROL), arm(PRUNED), pruned_ops=["blur"], trained_ops=["blur"])
    assert {r["condition"]: r["group"] for r in rows}["noise"] == "unseen_op"


def test_several_runs_per_arm_give_mean_and_std():
    control = [arm(CONTROL, checkpoint=f"c{s}") for s in range(2)]
    control[1] = [dict(r, metrics={"mIoU": r["metrics"]["mIoU"] + 2.0}) for r in control[1]]
    rows = K.diff_table(control, arm(PRUNED), pruned_ops=["blur"])
    blur = next(r for r in rows if r["condition"] == "blur" and r["level"] == 1.0)
    assert blur["control"] == pytest.approx(61.0) and blur["n_control"] == 2
    assert blur["control_std"] == pytest.approx(np.std([60.0, 62.0], ddof=1))
    assert blur["delta"] == pytest.approx(50.0 - 61.0)


def test_a_missing_cell_is_flagged_and_left_out_of_the_means():
    pruned = {k: v for k, v in PRUNED.items() if k != ("op", "blur", 1.0)}
    rows = K.diff_table(arm(CONTROL), arm(pruned), pruned_ops=["blur"])
    cell = next(r for r in rows if r["condition"] == "blur" and r["level"] == 1.0)
    assert cell["missing"] == "pruned" and cell["delta"] is None
    blur = next(r for r in K.condition_table(rows) if r["condition"] == "blur")
    assert blur["n_levels"] == 1 and blur["incomplete"] == 1 and blur["delta"] == pytest.approx(-4.0)


def test_different_measurements_do_not_compare():
    pruned = arm(PRUNED)
    pruned[1] = dict(pruned[1], n_images=99)
    with pytest.raises(ValueError, match="numbers of images"):
        K.diff_table(arm(CONTROL), pruned, pruned_ops=["blur"])
    pruned = arm(PRUNED)
    pruned[1] = dict(pruned[1], info={"magnitude": 0.7})
    with pytest.raises(ValueError, match="magnitudes"):
        K.diff_table(arm(CONTROL), pruned, pruned_ops=["blur"])


def test_a_run_is_one_checkpoint():
    mixed = arm(CONTROL)
    mixed[0] = dict(mixed[0], checkpoint="other")
    with pytest.raises(ValueError, match="several checkpoints"):
        K.diff_table(mixed, arm(PRUNED), pruned_ops=["blur"])


def test_pruned_ops_come_from_meta_when_not_given(caplog):
    rows = K.diff_table(arm(CONTROL), arm(PRUNED, meta={"pruned_ops": ["noise"]}))
    assert {r["condition"]: r["group"] for r in rows}["noise"] == "pruned"
    with pytest.raises(ValueError, match="no pruned_ops"):
        K.diff_table(arm(CONTROL), arm(PRUNED))
    runs = [arm(PRUNED, checkpoint="a", meta={"pruned_ops": ["noise"]}), arm(PRUNED, checkpoint="b", meta={"pruned_ops": ["blur"]})]
    with pytest.raises(ValueError, match="different pruned_ops"):
        K.diff_table(arm(CONTROL), runs)
    with caplog.at_level(logging.WARNING):
        K.diff_table(arm(CONTROL), arm(PRUNED), pruned_ops=["shear_x_pos"])
    assert "never evaluated" in caplog.text


def test_markdown():
    text = K.markdown([{"group": "pruned", "delta": -7.0, "n": 2, "std": None}])
    assert text.splitlines() == ["| group | delta | n | std |", "|---|---|---|---|", "| pruned | -7.00 | 2 |  |"]
    assert K.markdown([]) == ""


def test_sensaug_prune_history(tmp_path):
    (tmp_path / "20240101_000000").mkdir()
    (tmp_path / "20240101_000000" / "20240101_000000.log").write_text(
        "noise\n"
        "2024/01/01 - mmengine - INFO - [redundancy] mRMR pruned 2/32 ops at iter 2000 (lambda=0.5): blur, noise\n"
        "2024/01/01 - mmengine - INFO - [redundancy] mRMR pruned 1/32 ops at iter 4000 (lambda=0.5): noise\n"
    )
    counts, last = K.sensaug_prune_history(str(tmp_path))
    assert counts == {"blur": 1, "noise": 2} and last == {"noise"}
    (tmp_path / "empty").mkdir()
    assert K.sensaug_prune_history(str(tmp_path / "empty")) == ({}, None)


def test_end_to_end_same_model_has_zero_delta(tmp_path):
    ops = {name: vocab.get_op(name) for name in ("lighter_R", "rotate_pos", "noise")}
    conditions = [R.clean(), *R.op_conditions(ops, [0.5, 1.0])]
    runs = {}
    for name, meta in (("control", {}), ("pruned", {"pruned_ops": ["noise"]})):
        runs[name] = R.run(
            SegAdapter(), loader(seg_items()), conditions, lambda: R.SegMetric(3), str(tmp_path / f"{name}.jsonl"),
            checkpoint=f"{name}.pth", meta=meta,
        )
    rows = K.diff_table(R.load_results(str(tmp_path / "control.jsonl")), runs["pruned"])
    assert len(rows) == len(conditions)
    assert all(r["delta"] == pytest.approx(0.0) for r in rows)
    groups = [r["group"] for r in K.group_table(K.condition_table(rows))]
    assert groups == ["clean", "kept", "pruned"]
