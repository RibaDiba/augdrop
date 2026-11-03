"""tasks.augment: the training draws of a host with its own magnitude bins."""

import numpy as np
import pytest

pytest.importorskip("torch")

from hosts import build_host  # noqa: E402
import torch  # noqa: E402

from tasks.augment import NO_OP, augment_batch, augmix, check_trainable, sample_draws  # noqa: E402
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
    check_trainable(build_host("randaugment_diff", dataset="cifar100"), build_vocab({"families": ["randaug_diff"]}))
    check_trainable(build_host("augmix_diff"), build_vocab({"families": ["augmix_diff"]}))
    check_trainable(build_host("uniform"), build_vocab(None))
    wide = build_vocab({"families": ["trivialaug_wide"], "exclude": ["trivialaug_wide.Identity"]})
    with pytest.raises(NotImplementedError):
        check_trainable(build_host("trivialaugment"), wide)
    with pytest.raises(NotImplementedError):
        check_trainable(build_host("trivialaugment_diff"), build_vocab(None))


# --- randaugment_diff: N draws at the published level --------------------------------


class _Shift:
    """A stand-in op: adds 1 to the image and records the label it was handed."""

    def __init__(self):
        self.labels = []

    def apply(self, images, magnitude, label=None):
        self.labels.append(label)
        return images + 1.0, label


@pytest.mark.parametrize("dataset", ["cityscapes", "cifar100"])
def test_randaugment_diff_draws_at_its_level(dataset):
    host = build_host("randaugment_diff", dataset=dataset)
    for name, magnitude in sample_draws(host, 2000, np.random.RandomState(0)):
        if name == NO_OP or not get_op(name).has_magnitude:
            continue
        values, _ = magnitude_bins(name, host.num_magnitude_bins)
        assert magnitude == values[host.magnitude], name


def test_randaugment_diff_applies_n_draws_and_moves_the_label():
    host = build_host("randaugment_diff", dataset="cifar10")  # N = 3
    shift = _Shift()
    ops = {name: shift for name in host.vocabulary()}
    x = torch.zeros(400, 3, 4, 4)
    labels = torch.zeros(400, 1, 4, 4, dtype=torch.long)
    out, out_labels = augment_batch(host, x, labels, ops, np.random.RandomState(0))
    applied = out[:, 0, 0, 0]
    assert set(applied.tolist()) <= {0.0, 1.0, 2.0, 3.0} and applied.max() == host.num_ops
    # Each of the N draws is an op with p = 13/14.
    assert abs(applied.mean().item() - host.num_ops * 13 / 14) < 0.1
    assert out_labels is labels or torch.equal(out_labels, labels)
    assert all(label is not None for label in shift.labels)


# --- augmix_diff: chains mixed with the clean image, the mask left alone ------------------


def test_augmix_mixes_its_chains_and_leaves_the_mask():
    host = build_host("augmix_diff")
    shift = _Shift()
    ops = {name: shift for name in host.vocabulary()}
    x = torch.rand(64, 3, 4, 4)
    labels = torch.randint(0, 19, (64, 1, 4, 4))
    out, out_labels = augment_batch(host, x, labels, ops, np.random.default_rng(0))

    assert out_labels is labels
    assert shift.labels and all(label is None for label in shift.labels)
    # Every step applies an op (no none draw), so chain i is x + depth_i, and the mix is
    # x + sum_i w_i depth_i with augmix's own draw of the depths and weights.
    seed = np.random.default_rng(0).integers(0, 2**31 - 1)
    depths, clean, chains = host.draw_mix(64, np.random.default_rng(int(seed)))
    expected = x + torch.as_tensor((chains * depths).sum(1), dtype=x.dtype).view(-1, 1, 1, 1)
    torch.testing.assert_close(out, expected)


def test_augmix_on_real_ops_stays_an_image():
    host = build_host("augmix_diff")
    ops = build_vocab({"families": ["augmix_diff"]})
    x = torch.rand(6, 3, 24, 32)
    out = augmix(host, x, ops, np.random.RandomState(0))
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6
    assert not torch.equal(out, x)
