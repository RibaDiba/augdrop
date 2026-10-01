"""dropaug.eval.robustness: metrics, the per-condition pass, conditions, the resumable
run, and the summary table."""

import csv
import json
import logging
import math
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dropaug.eval import corruptions
from dropaug.eval import robustness as R
from eval_helpers import ChannelOracle, ClsAdapter, FakeGeometric, SegAdapter, cls_items, loader, seg_items
from probe import ProbeError, strength_to_magnitude

vocab = pytest.importorskip("vocab")


def seg_metric():
    return R.SegMetric(3)


def one_hot_scores(pred, k):
    return F.one_hot(pred, k).permute(0, 3, 1, 2).float()


# --- metrics --------------------------------------------------------------------------


def test_seg_metric_matches_a_hand_count():
    target = torch.tensor([[[0, 1], [2, 255]]])
    pred = torch.tensor([[[0, 1], [1, 0]]])
    metric = R.SegMetric(4)
    metric.update(one_hot_scores(pred, 4), target)
    out = metric.compute()
    # IoU: class 0 = 1/1, class 1 = 1/2, class 2 = 0/1, class 3 = 0/0 (left out).
    assert out["mIoU"] == pytest.approx(100 * (1 + 0.5 + 0) / 3)
    assert out["aAcc"] == pytest.approx(100 * 2 / 3)
    assert out["mAcc"] == pytest.approx(100 * (1 + 1 + 0) / 3)
    per_class = metric.per_class()
    assert math.isnan(per_class["IoU"][3])


def test_seg_metric_accumulates_over_batches_not_per_batch_means():
    rng = torch.Generator().manual_seed(0)
    target = torch.randint(0, 3, (4, 5, 5), generator=rng)
    pred = torch.randint(0, 3, (4, 5, 5), generator=rng)
    whole, split = R.SegMetric(3), R.SegMetric(3)
    whole.update(one_hot_scores(pred, 3), target)
    split.update(one_hot_scores(pred[:1], 3), target[:1])
    split.update(one_hot_scores(pred[1:], 3), target[1:])
    assert whole.compute() == pytest.approx(split.compute())


def test_seg_metric_rejects_a_resolution_mismatch_and_stray_ids():
    metric = R.SegMetric(3)
    with pytest.raises(ValueError, match="label's resolution"):
        metric.update(torch.zeros(1, 3, 4, 4), torch.zeros(1, 8, 8, dtype=torch.long))
    with pytest.raises(ValueError, match="outside"):
        metric.update(torch.zeros(1, 3, 2, 2), torch.tensor([[[0, 7], [1, 2]]]))


def test_cls_metric_topk():
    scores = torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    metric = R.ClsMetric(topk=(1, 5))
    metric.update(scores, torch.tensor([0, 0]))
    assert metric.compute() == {"top1": 50.0, "top5": 50.0}
    metric.update(scores, torch.tensor([1, 2]))
    assert metric.compute() == {"top1": 25.0, "top5": 75.0}


def test_ignore_index_is_the_vocab_label_fill():
    from vocab.diff_ops import LABEL_IGNORE_INDEX

    assert R.IGNORE_INDEX == LABEL_IGNORE_INDEX


# --- one pass -------------------------------------------------------------------------


def test_clean_oracle_scores_perfectly():
    out = R.evaluate(SegAdapter(), loader(seg_items()), seg_metric())
    assert out["metrics"]["mIoU"] == pytest.approx(100.0)
    assert out["n_images"] == 6
    assert out["unchanged"] is None


def test_geometric_op_is_graded_against_the_moved_label():
    op = vocab.get_op("rotate_pos")
    (condition,) = R.op_conditions({"rotate_pos": op}, [1.0])
    moved = R.evaluate(SegAdapter(), loader(seg_items()), seg_metric(), condition.perturb)["metrics"]["mIoU"]

    def image_only(images01, label, index):
        mag = torch.full((images01.shape[0],), 1.0)
        return op.apply(images01, mag, None)[0], label

    unmoved = R.evaluate(SegAdapter(), loader(seg_items()), seg_metric(), image_only)["metrics"]["mIoU"]
    # The oracle is pixelwise, so only interpolated stripe edges cost it anything
    # once the label moves too.
    assert moved > 85.0
    assert unmoved < moved - 10.0


def test_photometric_op_costs_the_oracle():
    (condition,) = R.op_conditions({"lighter_R": vocab.get_op("lighter_R")}, [1.0])
    out = R.evaluate(SegAdapter(), loader(seg_items()), seg_metric(), condition.perturb)
    assert out["metrics"]["mIoU"] < 40.0
    assert out["unchanged"] is False


def test_an_op_that_leaves_the_label_behind_is_refused():
    (condition,) = R.op_conditions({"fake_shift": FakeGeometric()}, [1.0])
    with pytest.raises(ProbeError, match="unwarped"):
        R.evaluate(SegAdapter(), loader(seg_items()), seg_metric(), condition.perturb)


def test_random_ops_are_pinned_and_consume_no_caller_rng():
    (condition,) = R.op_conditions({"noise": vocab.get_op("noise")}, [1.0], seed=3)
    torch.manual_seed(123)
    np.random.seed(123)
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    first = R.evaluate(SegAdapter(), loader(seg_items()), seg_metric(), condition.perturb)
    second = R.evaluate(SegAdapter(), loader(seg_items()), seg_metric(), condition.perturb)
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["mIoU"] < 100.0
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.random.get_state()[1].tolist() == numpy_state[1].tolist()


def test_the_model_is_frozen_and_its_modes_restored():
    adapter = SegAdapter()
    adapter.model.train()
    seen = []
    original = adapter.predict

    def predict(images01, batch):
        seen.append(adapter.model.training)
        return original(images01, batch)

    adapter.predict = predict
    R.evaluate(adapter, loader(seg_items()), seg_metric())
    assert seen and not any(seen)
    assert adapter.model.training


def test_classification_grades_against_the_batch_targets():
    out = R.evaluate(ClsAdapter(), loader(cls_items()), R.ClsMetric(topk=(1,)))
    assert out["metrics"]["top1"] == pytest.approx(100.0)
    (condition,) = R.op_conditions({"lighter_R": vocab.get_op("lighter_R")}, [1.0])
    out = R.evaluate(ClsAdapter(), loader(cls_items()), R.ClsMetric(topk=(1,)), condition.perturb)
    # Red saturates, so every image reads as class 0: a third of them are.
    assert out["metrics"]["top1"] == pytest.approx(100.0 / 3)


def test_a_classifier_without_target_is_refused():
    class NoTarget:
        model = ChannelOracle()

        def to_unit(self, batch):
            return batch["image"], None

        def predict(self, images01, batch):
            return self.model(images01)

    with pytest.raises(ValueError, match="target"):
        R.evaluate(NoTarget(), loader(cls_items()), R.ClsMetric())


# --- conditions -----------------------------------------------------------------------


def test_strengths_become_op_unit_magnitudes():
    ops = {name: vocab.get_op(name) for name in ("randaug_default.Solarize", "randaug_default.Equalize")}
    conditions = R.op_conditions(ops, [0.5, 1.0])
    by_key = {(c.name, c.level): c for c in conditions}
    assert by_key[("randaug_default.Solarize", 0.5)].info["magnitude"] == pytest.approx(128.0)
    assert by_key[("randaug_default.Solarize", 1.0)].info["magnitude"] == pytest.approx(0.0)
    (equalize,) = [c for c in conditions if c.name == "randaug_default.Equalize"]
    assert equalize.level is None and equalize.level_kind == "none"
    assert equalize.info["magnitude"] is None


def test_both_directions_mirrors_only_ops_with_two_sides():
    ops = {name: vocab.get_op(name) for name in ("randaug_default.Rotate", "randaug_default.Solarize")}
    conditions = R.op_conditions(ops, [0.5], both_directions=True)
    rotate = {c.level: c.info["magnitude"] for c in conditions if c.name == "randaug_default.Rotate"}
    assert rotate == {0.5: pytest.approx(15.0), -0.5: pytest.approx(-15.0)}
    assert [c.level for c in conditions if c.name == "randaug_default.Solarize"] == [0.5]


def test_level_grids_are_checked():
    ops = {name: vocab.get_op(name) for name in ("blur", "noise")}
    with pytest.raises(ValueError, match="no levels"):
        R.op_conditions(ops, {"blur": [0.5]})
    with pytest.raises(ValueError, match="not in the vocab"):
        R.op_conditions(ops, {"blur": [0.5], "noise": [0.5], "typo": [0.1]})
    with pytest.raises(ValueError, match="mag_range"):
        R.op_conditions(ops, [1.5], level_kind="magnitude")
    with pytest.raises(ValueError):
        R.op_conditions(ops, [1.5])
    per_op = R.op_conditions(ops, {"blur": [0.1, 0.3], "noise": [0.7]})
    assert [(c.name, c.level) for c in per_op] == [("blur", 0.1), ("blur", 0.3), ("noise", 0.7)]


def test_suite_is_test_robusts_set_over_the_vocab(monkeypatch):
    ops = vocab.build_vocab()
    conditions = R.suite(ops, corruptions=(), combination=False)
    assert len(conditions) == 1 + len(ops) * len(R.DEFAULT_STRENGTHS)
    assert conditions[0].key == R.clean().key
    # A corruption backend that isn't installed fails when the suite is built.
    monkeypatch.setitem(sys.modules, "imagecorruptions", None)
    with pytest.raises(ImportError, match="imagecorruptions"):
        R.suite(ops, combination=False)


def test_corruption_conditions_use_their_own_severities():
    def identity(image, name, severity, seed):
        return image

    conditions = R.corruption_conditions(["fog", corruptions.COMBINATION], fn=identity)
    levels = {name: [c.level for c in conditions if c.name == name] for name in ("fog", "combination")}
    assert levels == {"fog": [1, 2, 3, 4, 5], "combination": [1, 2, 3, 4, 5, 6]}
    assert {c.info["category"] for c in conditions} == {"weather", "combination"}
    assert {c.info["source"] for c in conditions} == {"identity"}


# --- run ------------------------------------------------------------------------------


def small_conditions(seed=0):
    ops = {name: vocab.get_op(name) for name in ("lighter_R", "rotate_pos")}
    return [R.clean(), *R.op_conditions(ops, [0.5, 1.0], seed=seed)]


def test_run_resumes_instead_of_re_evaluating(tmp_path):
    path = tmp_path / "out" / "results.jsonl"
    adapter = SegAdapter()
    first = R.run(adapter, loader(seg_items()), small_conditions(), seg_metric, str(path), checkpoint="a.pth")
    calls = adapter.calls
    assert len(first) == 5 and len(R.load_results(str(path))) == 5
    second = R.run(adapter, loader(seg_items()), small_conditions(), seg_metric, str(path), checkpoint="a.pth")
    assert adapter.calls == calls
    assert second == first
    record = first[1]
    assert record["condition"] == "lighter_R" and record["level"] == 0.5
    assert record["info"]["magnitude"] == 0.5 and record["info"]["family"] == "legacy_hsv"
    assert record["primary"] == "mIoU" and record["n_images"] == 6


def test_run_keeps_one_checkpoint_per_file(tmp_path):
    path = str(tmp_path / "results.jsonl")
    R.run(SegAdapter(), loader(seg_items()), [R.clean()], seg_metric, path, checkpoint="a.pth")
    with pytest.raises(ValueError, match="one checkpoint"):
        R.run(SegAdapter(), loader(seg_items()), [R.clean()], seg_metric, path, checkpoint="b.pth")


def test_run_refuses_to_reuse_a_differently_measured_condition(tmp_path):
    path = str(tmp_path / "results.jsonl")
    R.run(SegAdapter(), loader(seg_items()), small_conditions(seed=0), seg_metric, path)
    with pytest.raises(ValueError, match="measured as"):
        R.run(SegAdapter(), loader(seg_items()), small_conditions(seed=1), seg_metric, path)
    records = R.run(SegAdapter(), loader(seg_items()), small_conditions(seed=1), seg_metric, path, overwrite=True)
    assert {r["info"].get("seed") for r in records if r["kind"] == "op"} == {1}
    assert len(R.load_results(path)) == 10


def test_run_rejects_duplicate_conditions(tmp_path):
    with pytest.raises(ValueError, match="repeat"):
        R.run(SegAdapter(), [], [R.clean(), R.clean()], seg_metric, str(tmp_path / "r.jsonl"))


def test_an_identity_perturbation_is_flagged(tmp_path, caplog):
    (condition,) = R.op_conditions({"lighter_R": vocab.get_op("lighter_R")}, [0.0])
    with caplog.at_level(logging.WARNING):
        (record,) = R.run(SegAdapter(), loader(seg_items()), [condition], seg_metric, str(tmp_path / "r.jsonl"))
    assert record["unchanged"] is True
    assert "bit-identical" in caplog.text


def test_prerendered_loaders_are_built_only_when_they_run(tmp_path):
    built = []

    def make_loader(name, severity):
        built.append((name, severity))
        return loader(seg_items(seed=severity))

    conditions = R.prerendered_conditions(make_loader, ["fog"], [1, 2], source="toy")
    assert built == []
    path = str(tmp_path / "r.jsonl")
    R.run(SegAdapter(), loader(seg_items()), conditions, seg_metric, path)
    assert built == [("fog", 1), ("fog", 2)]
    R.run(SegAdapter(), loader(seg_items()), conditions, seg_metric, path)
    assert len(built) == 2


def test_on_the_fly_corruption_runs_through_run(tmp_path):
    def invert(image, name, severity, seed):
        return 255 - image

    conditions = [R.clean(), *R.corruption_conditions(["fog"], [1], fn=invert)]
    records = R.run(SegAdapter(), loader(seg_items()), conditions, seg_metric, str(tmp_path / "r.jsonl"))
    # Red 1/6 <-> 5/6 swap classes 0 and 2; class 1 stays at 1/2.
    assert records[1]["metrics"]["mIoU"] == pytest.approx(100.0 / 3, abs=1.0)
    assert records[1]["kind"] == "corruption" and records[1]["level_kind"] == "severity"


# --- tables ---------------------------------------------------------------------------


def test_summarize_and_write_csv(tmp_path):
    records = R.run(SegAdapter(), loader(seg_items()), small_conditions(), seg_metric, str(tmp_path / "r.jsonl"))
    rows = R.summarize(records, pruned_ops=["rotate_pos"])
    by_name = {row["condition"]: row for row in rows}
    clean_value = by_name["clean"]["mean"]
    assert clean_value == pytest.approx(100.0) and by_name["clean"]["drop_vs_clean"] == 0.0
    lighter = by_name["lighter_R"]
    values = [r["metrics"]["mIoU"] for r in records if r["condition"] == "lighter_R"]
    assert lighter["mean"] == pytest.approx(np.mean(values))
    assert lighter["worst"] == pytest.approx(min(values)) and lighter["worst_level"] == 1.0
    assert lighter["drop_vs_clean"] == pytest.approx(clean_value - np.mean(values))
    assert lighter["pruned"] is False and by_name["rotate_pos"]["pruned"] is True
    assert lighter["strength=0.5"] == pytest.approx(values[0])

    path = tmp_path / "summary.csv"
    R.write_csv(str(path), rows)
    with open(path) as f:
        read = list(csv.DictReader(f))
    assert [row["condition"] for row in read] == ["clean", "lighter_R", "rotate_pos"]
    assert "strength=1.0" in read[0] and read[0]["strength=1.0"] == ""


def test_mce_from_records_reproduces_alexnet_at_alexnets_accuracy():
    records = [
        {
            "dataset": "imagenet",
            "kind": "prerendered",
            "condition": name,
            "level_kind": "severity",
            "level": severity,
            "metrics": {"top1": 100.0 * (1 - err)},
        }
        for name, err in corruptions.ALEXNET_ERR.items()
        for severity in corruptions.SEVERITIES
    ]
    out = R.mce_from_records(records, dataset="imagenet")
    assert out["mCE"] == pytest.approx(100.0)


def test_records_are_valid_json_without_nan(tmp_path):
    class FiveClasses(SegAdapter):
        def predict(self, images01, batch):
            scores = super().predict(images01, batch)
            # Classes 3 and 4 are never predicted or labelled: IoU 0/0.
            return torch.cat([scores, torch.full_like(scores[:, :2], -10.0)], dim=1)

    path = tmp_path / "r.jsonl"
    R.run(FiveClasses(), loader(seg_items()), [R.clean()], lambda: R.SegMetric(5), str(path))
    text = path.read_text()
    assert "NaN" not in text
    (record,) = [json.loads(line) for line in text.splitlines()]
    assert record["per_class"]["IoU"][3] is None
