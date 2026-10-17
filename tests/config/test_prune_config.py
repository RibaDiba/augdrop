"""The prune: block through dropaug.config, train.py's flags and its prune.json.

The resolver itself is tests/hosts/test_prune.py. What this pins is the glue: the
composed block becomes the same Prune everywhere (host, run name, record), the old
sa keys are refused, and the run name's suffix is the one train.sbatch spells.
"""

import hashlib
import importlib.util
import json
import os

import pytest

pytest.importorskip("hydra")

from dropaug.config import ConfigError, check, compose, host_kwargs, host_of, prune_of, run_name, vocab_of  # noqa: E402
from hosts import draw_random_prune  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _train_module():
    spec = importlib.util.spec_from_file_location("dropaug_train", os.path.join(REPO, "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(names):
    return hashlib.sha1("".join(f"{n}\n" for n in sorted(set(names))).encode()).hexdigest()[:6]


def test_default_prunes_nothing():
    cfg = compose(["host=uniform"])
    assert not prune_of(cfg)
    assert "pruned" not in host_kwargs(cfg)
    assert host_of(cfg).pruned() == ()
    assert "_prune" not in run_name(cfg)


def test_random_draw_follows_the_run_seed():
    cfg = compose(["host=uniform", "prune.random=4", "seed=3"])
    prune = prune_of(cfg)
    names = list(vocab_of(cfg))
    assert prune.seed == 3
    assert prune.drawn == draw_random_prune(names, 4, 3)
    assert host_of(cfg).pruned() == prune.ops
    assert run_name(cfg).endswith("_s3_prune-r4")


def test_an_explicit_prune_seed_is_in_the_run_name():
    cfg = compose(["host=uniform", "prune.random=4", "prune.seed=7"])
    assert prune_of(cfg).seed == 7
    assert run_name(cfg).endswith("_s0_prune-r4s7")


def test_named_ops_with_dots_compose_and_hash_like_train_sbatch():
    ops = ["trivialaug_diff.Solarize", "trivialaug_diff.Rotate_pos"]
    cfg = compose(["host=trivialaugment_diff", "prune.ops=[" + ",".join(f"'{o}'" for o in ops) + "]", "prune.random=2"])
    prune = prune_of(cfg)
    assert set(prune.named) == set(ops)
    assert run_name(cfg).endswith(f"_prune-o2.{_hash(ops)}-r2")
    # what slurm/train.sbatch computes for PRUNE_OPS="trivialaug_diff.Solarize trivialaug_diff.Rotate_pos"
    assert _hash(ops) == hashlib.sha1(b"trivialaug_diff.Rotate_pos\ntrivialaug_diff.Solarize\n").hexdigest()[:6]


def test_sensaug_remove_h_keeps_the_h_ops_out_of_the_pool():
    cfg = compose(["host=sensaug", "sa.remove_H=true", "prune.random=3"])
    pool = prune_of(cfg).pool
    assert "lighter_H" not in pool and "darker_H" not in pool
    cfg = compose(["host=sensaug", "prune.random=3"])
    assert "lighter_H" in prune_of(cfg).pool


def test_host_kwargs_carry_the_prune():
    cfg = compose(["host=uniform", "prune.ops=[blur]", "prune.hold_none_prob=false"])
    kwargs = host_kwargs(cfg)
    assert kwargs["pruned"] == ["blur"] and kwargs["hold_none_prob"] is False


@pytest.mark.parametrize(
    "overrides, match",
    [
        (["prune.ops=[not_an_op]"], "not in the host's vocabulary"),
        (["prune.random=40"], "no op to train on"),
        (["+prune.bogus=1"], "unknown key"),
        (["+sa.pruned_augmentations=[blur]"], "moved to the prune"),
        (["+sa.hold_none_prob=true"], "moved to the prune"),
    ],
)
def test_check_refuses_a_bad_prune(overrides, match):
    cfg = compose(["host=uniform", *overrides])
    with pytest.raises(ConfigError, match=match):
        check(cfg, training=False)


def test_check_warns_on_a_prune_with_no_effect():
    warnings = check(compose(["host=uniform", "prune.ops=[blur]", "augment=false"]), training=False)
    assert any("augment=false" in w for w in warnings)
    warnings = check(compose(["host=uniform", "prune.probe=true"]), training=False)
    assert any("nothing is pruned" in w for w in warnings)


def test_check_warns_on_ops_autoaugment_prunes_by_implication():
    cfg = compose(["host=autoaugment", "prune.ops=['autoaugment.Invert']", "augment=false"])
    warnings = check(cfg, training=False)
    assert any("randaug_default.Sharpness" in w for w in warnings)


# --- train.py ---------------------------------------------------------------------


def test_train_flags_are_prune_overrides():
    train = _train_module()
    args = train.build_parser().parse_args(
        [
            "--host",
            "trivialaugment_diff",
            "--pruned-augmentations",
            "trivialaug_diff.Solarize",
            "trivialaug_diff.Equalize",
            "--random-prune-count",
            "3",
            "--random-prune-seed",
            "9",
            "--no-hold-none-prob",
            "--probe-pruned",
        ]
    )
    cfg = train.compose_config(args)
    prune = prune_of(cfg)
    assert prune.named == ("trivialaug_diff.Equalize", "trivialaug_diff.Solarize")  # vocabulary order
    assert prune.random == 3 and prune.seed == 9
    assert prune.hold_none_prob is False and prune.probe is True


def test_prune_json_is_written_and_guards_the_run(tmp_path):
    train = _train_module()
    cfg = compose(["host=uniform", "prune.random=4"])
    train.write_prune_record(cfg, str(tmp_path))
    record = json.loads((tmp_path / "prune.json").read_text())
    assert record["pruned"] == list(prune_of(cfg).ops)
    assert record["host"] == "uniform" and record["vocab_hash"]

    train.write_prune_record(cfg, str(tmp_path))  # the same prune again: fine
    with pytest.raises(SystemExit):
        train.write_prune_record(compose(["host=uniform", "prune.random=4", "prune.seed=1"]), str(tmp_path))
    with pytest.raises(SystemExit):
        train.write_prune_record(compose(["host=uniform"]), str(tmp_path))


def test_no_prune_writes_no_record(tmp_path):
    train = _train_module()
    train.write_prune_record(compose(["host=uniform"]), str(tmp_path))
    assert not (tmp_path / "prune.json").exists()
