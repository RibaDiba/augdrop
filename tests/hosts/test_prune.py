"""The static prune: AugHost.prune on every host, and hosts.prune's resolver.

Pins the rules in hosts/base.py's docstring: pruned ops keep their place in the
vocabulary at base probability 0, the rest keep their ratios, P(none) is held unless
asked otherwise (then the no-op draw is renormalized with the surviving ops, as
Sensaug's bank did), and no q can bring a pruned op back.
"""

import numpy as np
import pytest

from hosts import Prune, build_host, draw_random_prune, resolve_prune
from hosts.autoaugment import IMAGENET_POLICY, AutoAugmentHost
from hosts.randaugment import RandAugmentHost
from hosts.sensaug import DIFF32_OPS, NONE_KEY, SensAugHost
from hosts.trivialaugment import TrivialAugmentDiffHost, TrivialAugmentHost
from hosts.uniform import UniformHost

# Every host whose sampling unit is one op (AutoAugment's is a sub-policy: below).
PER_OP = {
    "sensaug": lambda: SensAugHost(max_iters=20000),
    "randaugment": lambda: RandAugmentHost(probe_interval=1000),
    "trivialaugment": lambda: TrivialAugmentHost(probe_interval=1000),
    "trivialaugment_diff": lambda: TrivialAugmentDiffHost(probe_interval=1000),
    "uniform": lambda: UniformHost(probe_interval=1000),
}


def _two(host):
    """Two ops to prune, not adjacent, so the order of pruned() is tested."""
    names = host.vocabulary()
    return [names[5], names[1]]


@pytest.mark.parametrize("name", sorted(PER_OP))
def test_prune_zeroes_the_ops_and_keeps_the_rest_in_ratio(name):
    host = PER_OP[name]()
    before = host.base_probs()
    vocabulary = host.vocabulary()
    pruned = _two(host)

    assert host.prune(pruned) == ()
    assert host.vocabulary() == vocabulary
    assert host.pruned() == (vocabulary[1], vocabulary[5])
    base = host.base_probs()
    keep = np.array([n not in pruned for n in vocabulary])
    assert np.all(base[~keep] == 0.0)
    np.testing.assert_allclose(base.sum(), 1.0)
    np.testing.assert_allclose(base[keep], before[keep] / before[keep].sum())
    np.testing.assert_array_equal(host.sampling_probs(), base)


@pytest.mark.parametrize("name", sorted(PER_OP))
def test_hold_none_prob_keeps_p_none(name):
    host = PER_OP[name]()
    none = host.none_prob
    host.prune(_two(host))
    assert host.none_prob == none


def test_without_hold_the_bank_shrinks_as_sensaugs_did():
    """Sensaug's warmup / --random-aug bank: uniform over the surviving ops plus none."""
    host = UniformHost()
    pruned = ["blur", "noise", "rotate_pos"]
    host.prune(pruned, hold_none_prob=False)
    n_left = len(DIFF32_OPS) - len(pruned)
    assert host.none_prob == pytest.approx(1.0 / (n_left + 1))
    op_probs = (1.0 - host.none_prob) * host.sampling_probs()
    kept = [i for i, n in enumerate(host.vocabulary()) if n not in pruned]
    np.testing.assert_allclose(op_probs[kept], host.none_prob)


def test_without_hold_trivialaugment_diff_drops_the_entries_from_its_joint():
    host = TrivialAugmentDiffHost()
    joint = np.append((1.0 - host.none_prob) * host.base_probs(), host.none_prob)
    pruned = ["trivialaug_diff.Rotate_pos", "trivialaug_diff.Solarize"]
    index = [host.vocabulary().index(n) for n in pruned]
    host.prune(pruned, hold_none_prob=False)

    joint[index] = 0.0
    joint /= joint.sum()
    got = np.append((1.0 - host.none_prob) * host.sampling_probs(), host.none_prob)
    np.testing.assert_allclose(got, joint)


def test_q_cannot_revive_a_pruned_op():
    host = UniformHost()
    host.prune(["blur"])
    q = dict(zip(host.vocabulary(), np.ones(len(DIFF32_OPS))))
    with pytest.raises(ValueError, match="prunes"):
        host.set_weights(q)


def test_a_q_derived_from_base_probs_is_accepted():
    host = UniformHost()
    host.prune(["blur", "noise"])
    base = host.base_probs()
    tilt = base * np.linspace(0.5, 1.5, base.size)  # a KL-style tilt: 0 stays 0
    host.set_weights(tilt)
    np.testing.assert_allclose(host.sampling_probs(), tilt / tilt.sum())
    assert host.sampling_probs()[host.vocabulary().index("blur")] == 0.0


def test_prune_clears_a_q_in_effect():
    host = UniformHost()
    host.set_weights(np.linspace(1, 2, len(DIFF32_OPS)))
    host.prune(["blur"])
    np.testing.assert_array_equal(host.sampling_probs(), host.base_probs())


def test_an_empty_prune_is_a_no_op():
    host = UniformHost()
    base = host.base_probs()
    assert host.prune([]) == ()
    assert host.pruned() == ()
    np.testing.assert_array_equal(host.base_probs(), base)


@pytest.mark.parametrize(
    "names, exc, match",
    [
        (["not_an_op"], ValueError, "not in the vocabulary"),
        (list(DIFF32_OPS), ValueError, "names every op"),
    ],
)
def test_prune_refuses(names, exc, match):
    with pytest.raises(exc, match=match):
        UniformHost().prune(names)


def test_prune_happens_once():
    host = UniformHost()
    host.prune(["blur"])
    with pytest.raises(RuntimeError, match="prune once"):
        host.prune(["noise"])


def test_build_host_applies_the_prune():
    host = build_host("uniform", pruned=["blur"], hold_none_prob=False)
    assert host.pruned() == ("blur",)
    assert host.none_prob == pytest.approx(1.0 / len(DIFF32_OPS))
    assert build_host("uniform").pruned() == ()


# --- sensaug: the warmup policy and the SA pdf -------------------------------------


def _pdf(ops, none=0.1):
    mass = (1.0 - none) / (2 * len(ops))
    pdf = {(op, level): mass for op in ops for level in (0.4, 0.8)}
    pdf[NONE_KEY] = none
    return pdf


def test_sensaug_warmup_policy_leaves_the_pruned_ops_out():
    host = SensAugHost(max_iters=20000)
    host.prune(["blur", "noise"], hold_none_prob=False)
    assert host.pdf() is None  # still warmup
    assert host.none_prob == pytest.approx(1.0 / (len(DIFF32_OPS) - 1))
    assert host.sampling_probs()[host.vocabulary().index("blur")] == 0.0


def test_sensaug_refuses_a_pdf_that_samples_a_pruned_op():
    host = SensAugHost(max_iters=20000)
    host.prune(["blur"])
    with pytest.raises(ValueError, match="prunes"):
        host.set_sa_pdf(_pdf(DIFF32_OPS))


def test_sensaug_takes_a_pdf_without_the_pruned_ops():
    host = SensAugHost(max_iters=20000)
    host.prune(["blur"])
    kept = [op for op in DIFF32_OPS if op != "blur"]
    pdf = _pdf(kept, none=0.2)
    host.set_sa_pdf(pdf)
    assert host.pdf() == pdf
    assert host.none_prob == 0.2  # the pdf's, whatever hold_none_prob said for warmup
    with pytest.raises(ValueError, match="prunes"):
        host.set_weights(np.ones(len(DIFF32_OPS)))


# --- autoaugment: the sub-policies ------------------------------------------------


def test_autoaugment_prune_removes_every_subpolicy_that_applies_the_op():
    host = AutoAugmentHost()
    host.prune(["randaug_default.Rotate"])
    probs = host.subpolicy_probs()
    for s, sub in enumerate(IMAGENET_POLICY):
        uses = any(op == "Rotate" and p > 0 for op, p, _ in sub)
        assert (probs[s] == 0.0) == uses
    np.testing.assert_allclose(probs.sum(), 1.0)
    assert host.base_probs()[host.vocabulary().index("randaug_default.Rotate")] == 0.0
    np.testing.assert_allclose(host.sampling_probs(), host.base_probs())


def test_autoaugment_prune_reports_the_ops_it_implies():
    # Sharpness is applied only by (Sharpness, Invert): pruning Invert leaves it nothing.
    host = AutoAugmentHost()
    assert host.prune(["autoaugment.Invert"]) == ("randaug_default.Sharpness",)
    assert set(host.pruned()) == {"autoaugment.Invert", "randaug_default.Sharpness"}


def test_autoaugment_q_after_a_prune_keeps_the_removed_subpolicies_out():
    host = AutoAugmentHost()
    host.prune(["randaug_default.Rotate"])
    removed = host.subpolicy_probs() == 0.0
    host.set_weights(host.base_probs())
    assert np.all(host.subpolicy_probs()[removed] == 0.0)
    np.testing.assert_allclose(host.subpolicy_probs()[~removed], 1.0 / (~removed).sum())


def test_autoaugment_refuses_a_prune_that_leaves_no_subpolicy():
    host = AutoAugmentHost(policy=((("Rotate", 0.5, 3), ("Color", 0.5, 1)),))
    with pytest.raises(ValueError, match="leaves nothing to draw"):
        host.prune(["randaug_default.Rotate"])


# --- resolve_prune: the named ops and the random draw -----------------------------


def test_draw_is_sensaugs_draw_random_prune():
    pool = ["c", "a", "e", "b", "d", "f"]
    expected = sorted(np.random.RandomState(7).choice(sorted(pool), size=3, replace=False).tolist())
    assert draw_random_prune(pool, 3, 7) == tuple(expected)
    assert draw_random_prune(list(reversed(pool)), 3, 7) == tuple(expected)  # order-free


def test_resolve_draws_from_the_vocabulary_minus_named_and_excluded():
    prune = resolve_prune(DIFF32_OPS, ops=["noise", "blur"], random=4, seed=3, exclude=["lighter_H", "darker_H"])
    assert prune.named == ("blur", "noise")  # vocabulary order
    assert prune.pool == tuple(sorted(set(DIFF32_OPS) - {"noise", "blur", "lighter_H", "darker_H"}))
    assert prune.drawn == draw_random_prune(prune.pool, 4, 3)
    assert set(prune.ops) == {"noise", "blur", *prune.drawn}
    assert list(prune.ops) == [n for n in DIFF32_OPS if n in prune.ops]
    assert prune and prune.random == 4 and prune.seed == 3


def test_resolve_is_a_pure_function_of_its_arguments():
    a = resolve_prune(DIFF32_OPS, random=6, seed=0)
    b = resolve_prune(DIFF32_OPS, random=6, seed=0)
    c = resolve_prune(DIFF32_OPS, random=6, seed=1)
    assert a == b
    assert a.drawn != c.drawn


def test_resolve_with_nothing_pruned():
    prune = resolve_prune(DIFF32_OPS, seed=5)
    assert not prune
    assert prune == Prune(seed=None)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(ops=["nope"]), "not in the host's vocabulary"),
        (dict(ops=list(DIFF32_OPS)), "names every op"),
        (dict(random=2), "prune.seed"),
        (dict(random=-1, seed=0), "count >= 0"),
        (dict(random=True, seed=0), "count >= 0"),
        (dict(random=30, seed=0, exclude=["lighter_H", "darker_H"]), "no op to train on"),
    ],
)
def test_resolve_refuses(kwargs, match):
    with pytest.raises(ValueError, match=match):
        resolve_prune(DIFF32_OPS, **kwargs)


def test_record_lists_what_is_kept():
    prune = resolve_prune(DIFF32_OPS, ops=["blur"], random=2, seed=0)
    record = prune.record(DIFF32_OPS, host="uniform")
    assert record["pruned"] == list(prune.ops)
    assert record["kept"] == [n for n in DIFF32_OPS if n not in prune.ops]
    assert record["drawn"] == list(prune.drawn) and record["host"] == "uniform"
