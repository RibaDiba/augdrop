"""tasks.augment: the training draws of a host with its own magnitude bins."""

import numpy as np
import pytest

pytest.importorskip("torch")

from hosts import build_host  # noqa: E402
from tasks.augment import NO_OP, check_trainable, sample_draws  # noqa: E402
from vocab import build_vocab, get_op, magnitude_bins  # noqa: E402


def test_trivialaugment_diff_draws_at_its_bins():
    host = build_host("trivialaugment_diff")
    draws = sample_draws(host, 4000, np.random.RandomState(0))
    names = [name for name, _ in draws]
    assert set(names) <= set(host.vocabulary()) | {NO_OP}
    assert abs(names.count(NO_OP) / len(names) - host.none_prob) < 0.02
    for name, magnitude in draws:
        if name == NO_OP or not get_op(name).has_magnitude:
            assert magnitude == 0.0
            continue
        values, _ = magnitude_bins(name, host.num_magnitude_bins)
        assert min(abs(magnitude - v) for v in values) < 1e-9, (name, magnitude)


def test_trivialaugment_diff_draws_follow_q():
    host = build_host("trivialaugment_diff")
    q = np.zeros(len(host.vocabulary()))
    q[host.vocabulary().index("trivialaug_diff.Rotate_neg")] = 1.0
    host.set_weights(q)
    names = {name for name, _ in sample_draws(host, 500, np.random.default_rng(0))}
    assert names == {"trivialaug_diff.Rotate_neg", NO_OP}


def test_draws_are_seeded_by_the_callers_rng():
    host = build_host("trivialaugment_diff")
    a = sample_draws(host, 50, np.random.RandomState(1))
    b = sample_draws(host, 50, np.random.RandomState(1))
    assert a == b


def test_check_trainable():
    check_trainable(build_host("trivialaugment_diff"), build_vocab({"families": ["trivialaug_diff"]}))
    check_trainable(build_host("uniform"), build_vocab(None))
    wide = build_vocab({"families": ["trivialaug_wide"], "exclude": ["trivialaug_wide.Identity"]})
    with pytest.raises(NotImplementedError):
        check_trainable(build_host("trivialaugment"), wide)
    with pytest.raises(NotImplementedError):
        check_trainable(build_host("trivialaugment_diff"), build_vocab(None))
