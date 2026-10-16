import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from hosts import HOSTS, AugHost, build_host
from hosts.autoaugment import IMAGENET_POLICY, AutoAugmentHost, vocab_name
from hosts.randaugment import IDENTITY, OP_ORDER, RandAugmentHost
from hosts.sensaug import (
    DIFF32_OPS,
    NONE_KEY,
    SensAugHost,
    derive_corr_rounds,
    round_fire_iters,
)
from hosts.trivialaugment import TrivialAugmentDiffHost, TrivialAugmentHost
from hosts.uniform import UniformHost

SENSAUG_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Sensaug"

# One instance of every host, each with a schedule so construction is exercised.
FACTORIES = {
    "sensaug": lambda: SensAugHost(max_iters=20000),
    "randaugment": lambda: RandAugmentHost(probe_interval=1000),
    "trivialaugment": lambda: TrivialAugmentHost(probe_interval=1000),
    "trivialaugment_diff": lambda: TrivialAugmentDiffHost(probe_interval=1000),
    "autoaugment": lambda: AutoAugmentHost(probe_interval=1000),
    "uniform": lambda: UniformHost(probe_interval=1000),
}


def _sensaug_module(name):
    """Import a module from the vendored Sensaug checkout, or skip."""
    if not SENSAUG_ROOT.exists():
        pytest.skip("third_party/Sensaug is not checked out")
    if str(SENSAUG_ROOT) not in sys.path:
        sys.path.insert(0, str(SENSAUG_ROOT))
    return pytest.importorskip(name)


def _sa_pdf(ops, levels=(0.25, 0.5, 0.75, 1.0), none=0.1, seed=0):
    """A pdf shaped like Sensaug's: uneven level shapes, uneven op masses, a none entry."""
    rng = np.random.default_rng(seed)
    raw = {(op, lvl): float(rng.uniform(0.1, 1.0)) for op in ops for lvl in levels}
    scale = (1.0 - none) / sum(raw.values())
    pdf = {key: value * scale for key, value in raw.items()}
    pdf[NONE_KEY] = none
    return pdf


# --- the shared contract -------------------------------------------------------


def test_registry_covers_every_host():
    assert set(HOSTS) == set(FACTORIES)
    for name, cls in HOSTS.items():
        assert issubclass(cls, AugHost) and cls.name == name


def test_build_host_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown host"):
        build_host("mixup")


def test_build_host_passes_kwargs():
    host = build_host("randaugment", num_ops=3, probe_interval=500)
    assert isinstance(host, RandAugmentHost) and host.num_ops == 3
    assert host.should_probe(500)


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_initial_state(name):
    host = FACTORIES[name]()
    names = host.vocabulary()
    assert len(names) == len(set(names)) > 0
    np.testing.assert_allclose(host.base_probs().sum(), 1.0)
    np.testing.assert_array_equal(host.sampling_probs(), host.base_probs())
    assert 0.0 <= host.none_prob < 1.0


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_mapping_and_sequence_weights_agree(name):
    a, b = FACTORIES[name](), FACTORIES[name]()
    if isinstance(a, SensAugHost):
        a.set_sa_pdf(_sa_pdf(a.vocabulary()))
        b.set_sa_pdf(_sa_pdf(b.vocabulary()))
    q = np.linspace(1.0, 2.0, len(a.vocabulary()))
    a.set_weights(q)
    b.set_weights(dict(zip(b.vocabulary(), q)))
    np.testing.assert_allclose(a.sampling_probs(), b.sampling_probs())
    np.testing.assert_allclose(a.sampling_probs().sum(), 1.0)


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_set_weights_holds_none_prob(name):
    host = FACTORIES[name]()
    if isinstance(host, SensAugHost):
        host.set_sa_pdf(_sa_pdf(host.vocabulary()))
    before = host.none_prob
    q = np.ones(len(host.vocabulary()))
    q[0] = 5.0
    host.set_weights(q)
    assert host.none_prob == before


@pytest.mark.parametrize(
    "q, match",
    [
        (lambda n: np.full(n, np.nan), "non-finite"),
        (lambda n: np.r_[-1.0, np.ones(n - 1)], "negative"),
        (lambda n: np.zeros(n), "nothing to sample"),
        (lambda n: np.ones(n + 1), "shape"),
    ],
)
def test_set_weights_rejects_bad_q(q, match):
    host = UniformHost()
    with pytest.raises(ValueError, match=match):
        host.set_weights(q(len(host.vocabulary())))


def test_set_weights_mapping_must_cover_vocabulary_exactly():
    host = UniformHost(vocabulary=("a", "b"))
    with pytest.raises(ValueError, match="missing"):
        host.set_weights({"a": 1.0})
    with pytest.raises(ValueError, match="unknown"):
        host.set_weights({"a": 1.0, "b": 1.0, "c": 1.0})


def test_q_replaces_rather_than_compounds():
    host = UniformHost(vocabulary=("a", "b"))
    host.set_weights([3.0, 1.0])
    host.set_weights([3.0, 1.0])
    np.testing.assert_allclose(host.sampling_probs(), [0.75, 0.25])


def test_duplicate_vocabulary_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        UniformHost(vocabulary=("a", "b", "a"))


# --- the probe gate --------------------------------------------------------------


def test_interval_gate():
    host = UniformHost(probe_interval=100)
    assert [i for i in range(0, 401) if host.should_probe(i)] == [100, 200, 300, 400]
    assert host.probe_role(100) == "active" and host.probe_role(150) is None


def test_explicit_iters_and_control_role():
    host = UniformHost(probe_iters=[300, 100], control_iters=[100])
    assert host.should_probe(100) and host.should_probe(300) and not host.should_probe(200)
    assert host.probe_role(100) == "control" and host.probe_role(300) == "active"


def test_no_schedule_never_probes():
    host = UniformHost()
    assert not any(host.should_probe(i) for i in range(0, 10000, 7))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(probe_interval=10, probe_iters=[10]), "both"),
        (dict(probe_iters=[]), "empty"),
        (dict(probe_iters=[0, 10]), "positive"),
        (dict(probe_interval=0), "positive"),
        (dict(probe_interval=10, control_iters=[15]), "never fires"),
    ],
)
def test_gate_rejects_bad_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        UniformHost(**kwargs)


# --- the names line up with vocab ------------------------------------------------

@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_vocabulary_is_a_vocab(name):
    """Every host op is a vocab op, so the probe can take the host's vocabulary
    as its op list, in the host's order."""
    vocab = pytest.importorskip("vocab")
    names = FACTORIES[name]().vocabulary()
    assert tuple(vocab.build_vocab({"ops": names})) == names


@pytest.mark.parametrize(
    "host, config",
    [
        (lambda: SensAugHost(max_iters=20000), None),
        (lambda: UniformHost(), None),
        (RandAugmentHost, {"families": ["randaug_default"], "exclude": ["randaug_default.Identity"]}),
        (TrivialAugmentHost, {"families": ["trivialaug_wide"], "exclude": ["trivialaug_wide.Identity"]}),
        (TrivialAugmentDiffHost, {"families": ["trivialaug_diff"]}),
    ],
)
def test_vocabulary_order_is_vocab_order(host, config):
    vocab = pytest.importorskip("vocab")
    assert host().vocabulary() == tuple(vocab.build_vocab(config))


# --- the torchvision hosts --------------------------------------------------------


def test_torchvision_op_spaces_match():
    v2 = pytest.importorskip("torchvision.transforms.v2")
    assert set(v2.RandAugment()._AUGMENTATION_SPACE) == set(OP_ORDER)
    assert set(v2.TrivialAugmentWide()._AUGMENTATION_SPACE) == set(OP_ORDER)
    assert set(v2.AutoAugment()._AUGMENTATION_SPACE) == set(OP_ORDER) - {IDENTITY} | {"Invert"}


def test_imagenet_policy_matches_torchvision():
    tv = pytest.importorskip("torchvision.transforms")
    v2 = pytest.importorskip("torchvision.transforms.v2")
    expected = v2.AutoAugment()._get_policies(tv.AutoAugmentPolicy.IMAGENET)
    assert tuple(tuple(tuple(step) for step in sub) for sub in expected) == IMAGENET_POLICY


def test_randaugment_defaults_match_torchvision():
    v2 = pytest.importorskip("torchvision.transforms.v2")
    ref, host = v2.RandAugment(), RandAugmentHost()
    assert (host.num_ops, host.magnitude, host.num_magnitude_bins) == (
        ref.num_ops,
        ref.magnitude,
        ref.num_magnitude_bins,
    )
    assert TrivialAugmentHost().num_magnitude_bins == v2.TrivialAugmentWide().num_magnitude_bins


@pytest.mark.parametrize("cls", [RandAugmentHost, TrivialAugmentHost])
def test_identity_is_held_out(cls):
    host = cls()
    assert f"{host.family}.{IDENTITY}" not in host.vocabulary()
    assert all(n.startswith(f"{host.family}.") for n in host.vocabulary())
    assert host.none_prob == pytest.approx(1 / 14)


def test_randaugment_rejects_out_of_range_magnitude():
    with pytest.raises(ValueError, match="bin index"):
        RandAugmentHost(magnitude=31)


# --- AutoAugment ---------------------------------------------------------------------


def test_autoaugment_vocabulary_is_the_ops_the_policy_applies():
    host = AutoAugmentHost()
    applied = {vocab_name(op) for sub in IMAGENET_POLICY for op, p, _ in sub if p > 0}
    assert set(host.vocabulary()) == applied
    assert host.none_prob == 0.0


def test_autoaugment_q_equal_to_base_keeps_the_policy_uniform():
    host = AutoAugmentHost()
    host.set_weights(host.base_probs())
    np.testing.assert_allclose(host.subpolicy_probs(), 1 / len(IMAGENET_POLICY))
    np.testing.assert_allclose(host.sampling_probs(), host.base_probs())


def test_autoaugment_dropping_an_op_removes_every_subpolicy_that_applies_it():
    host = AutoAugmentHost()
    q = dict(zip(host.vocabulary(), host.base_probs()))
    q["randaug_default.Rotate"] = 0.0
    host.set_weights(q)
    probs = host.subpolicy_probs()
    for s, sub in enumerate(IMAGENET_POLICY):
        if any(op == "Rotate" and p > 0 for op, p, _ in sub):
            assert probs[s] == 0.0
        else:
            assert probs[s] > 0.0
    assert host.sampling_probs()[host.vocabulary().index("randaug_default.Rotate")] == 0.0
    np.testing.assert_allclose(probs.sum(), 1.0)


def test_autoaugment_p_zero_step_does_not_drop_its_row():
    # Row 13 is (("Equalize", 0.0), ("Equalize", 0.8)); only the 0.8 step counts.
    policy = ((("Equalize", 0.0, None), ("Rotate", 0.5, 3)), (("Rotate", 0.5, 3), ("Color", 0.5, 1)))
    host = AutoAugmentHost(policy=policy)
    assert host.vocabulary() == ("randaug_default.Rotate", "randaug_default.Color")
    host.set_weights({"randaug_default.Rotate": 1.0, "randaug_default.Color": 0.0})
    np.testing.assert_allclose(host.subpolicy_probs(), [1.0, 0.0])


def test_autoaugment_refuses_to_drop_everything():
    host = AutoAugmentHost()
    q = dict.fromkeys(host.vocabulary(), 0.0)
    q["autoaugment.Invert"] = 1.0  # every row applying Invert also applies Sharpness or Equalize
    with pytest.raises(ValueError, match="no sub-policy is left"):
        host.set_weights(q)


def test_autoaugment_rejects_unknown_op():
    with pytest.raises(ValueError, match="not an AutoAugment op"):
        AutoAugmentHost(policy=((("Mixup", 0.5, 1), ("Rotate", 0.5, 1)),))


# --- SensAug: the schedule --------------------------------------------------------


def test_sensaug_default_schedule_is_the_shipped_one():
    host = SensAugHost(max_iters=20000)
    assert host.round_interval == 1000 and host.n_rounds == 20
    assert host.rounds == (3, 4, 7, 10, 13, 16)
    fired = [i for i in range(20001) if host.should_probe(i)]
    assert fired == [4000, 5000, 8000, 11000, 14000, 17000]
    assert host.probe_role(4000) == "control"
    assert all(host.probe_role(i) == "active" for i in fired[1:])


def test_sensaug_cadence_six_is_curve_recompute_only():
    assert SensAugHost(max_iters=20000, corr_cadence=6).rounds == (3, 4, 10, 16)


def test_sensaug_no_warmup_has_no_control():
    host = SensAugHost(max_iters=20000, warmup_rounds=0)
    fired = [i for i in range(20001) if host.should_probe(i)]
    assert fired and all(host.probe_role(i) == "active" for i in fired)


def test_sensaug_probe_interval_overrides_the_rounds():
    host = SensAugHost(max_iters=20000, probe_interval=5000)
    assert host.rounds == ()
    assert [i for i in range(20001) if host.should_probe(i)] == [5000, 10000, 15000, 20000]


def test_sensaug_rejects_bad_grids():
    with pytest.raises(ValueError, match="no post-warmup"):
        SensAugHost(max_iters=4000, round_interval=1000, warmup_rounds=4)
    with pytest.raises(ValueError, match="outside"):
        SensAugHost(max_iters=20000, corr_rounds=[3, 25])
    with pytest.raises(ValueError, match="not in corr_rounds"):
        SensAugHost(max_iters=20000, corr_rounds=[4, 10], control_rounds=[3])


def test_sensaug_warns_on_final_round():
    with pytest.warns(UserWarning, match="final round"):
        SensAugHost(max_iters=20000, corr_rounds=[4, 19])


def test_schedule_matches_sensaug_round_schedule():
    rs = _sensaug_module("sensaug.round_schedule")
    for n, warmup, cadence in itertools.product(range(1, 30), range(0, 8), (1, 2, 3, 6)):
        assert derive_corr_rounds(n, warmup, cadence) == rs.derive_corr_rounds(n, warmup, cadence)
    for rounds, interval, pre in itertools.product(([0, 3, 4], [1, 7]), (1, 250), (False, True)):
        assert round_fire_iters(rounds, interval, pre) == rs.round_fire_iters(rounds, interval, pre)

    cfg = rs.RoundConfig(n_rounds=20, warmup_rounds=4, corr_cadence=3)
    for max_iters, interval, pre in [(20000, None, False), (80000, 2000, False), (20000, None, True)]:
        host = SensAugHost(max_iters=max_iters, round_interval=interval, pre_train_round=pre)
        ref = rs.resolve_schedule(cfg, host.n_rounds, 4, host.round_interval, pre)
        fired = tuple(i for i in range(max_iters + 1) if host.should_probe(i))
        controls = tuple(i for i in fired if host.probe_role(i) == "control")
        assert (host.rounds, fired, controls) == (ref.rounds, ref.fire_iters, ref.control_iters)


# --- SensAug: the pdf -----------------------------------------------------------------


def test_sensaug_warmup_policy():
    host = SensAugHost(max_iters=20000)
    assert host.pdf() is None
    assert host.none_prob == pytest.approx(1 / 33)
    with pytest.raises(RuntimeError, match="warmup"):
        host.set_weights(np.ones(32))


def test_sensaug_pdf_without_q_is_bit_identical():
    host = SensAugHost(max_iters=20000)
    pdf = _sa_pdf(DIFF32_OPS)
    host.set_sa_pdf(pdf)
    assert host.pdf() == pdf
    assert host.none_prob == pdf[NONE_KEY]


def test_sensaug_q_moves_mass_between_ops_only():
    host = SensAugHost(max_iters=20000)
    pdf = _sa_pdf(DIFF32_OPS)
    host.set_sa_pdf(pdf)
    q = np.linspace(0.5, 2.0, 32)
    host.set_weights(q)
    out = host.pdf()

    assert set(out) == set(pdf)
    assert out[NONE_KEY] == pdf[NONE_KEY]
    np.testing.assert_allclose(sum(out.values()), 1.0)
    for i, op in enumerate(DIFF32_OPS):
        old = {k: v for k, v in pdf.items() if k[0] == op}
        new = {k: out[k] for k in old}
        # The op's total is (1 - none) * q, and its level shape is unchanged.
        np.testing.assert_allclose(sum(new.values()), (1 - pdf[NONE_KEY]) * q[i] / q.sum())
        for k in old:
            np.testing.assert_allclose(new[k] / sum(new.values()), old[k] / sum(old.values()))


def test_sensaug_matches_lever3_reweight():
    red_mod = _sensaug_module("sensaug.redundancy")
    host = SensAugHost(max_iters=20000)
    pdf = _sa_pdf(DIFF32_OPS)
    host.set_sa_pdf(pdf)
    rng = np.random.default_rng(1)
    red = dict(zip(DIFF32_OPS, rng.standard_normal(32)))
    lam = 0.25

    expected = red_mod.reweight(pdf, red, lam).pdf
    base = host.base_probs()
    host.set_weights(base * np.exp(-lam * np.array([red[n] for n in DIFF32_OPS])))
    out = host.pdf()
    for key in pdf:
        np.testing.assert_allclose(out[key], expected[key], rtol=1e-12, atol=1e-15)


def test_sensaug_q_persists_across_sa_pdfs():
    host = SensAugHost(max_iters=20000)
    host.set_sa_pdf(_sa_pdf(DIFF32_OPS, seed=0))
    q = np.ones(32)
    q[DIFF32_OPS.index("blur")] = 0.0
    host.set_weights(q)
    host.set_sa_pdf(_sa_pdf(DIFF32_OPS, seed=1))
    assert all(v == 0.0 for k, v in host.pdf().items() if k[0] == "blur")
    np.testing.assert_allclose(sum(host.pdf().values()), 1.0)


def test_sensaug_ops_missing_from_the_pdf():
    host = SensAugHost(max_iters=20000)
    kept = [n for n in DIFF32_OPS if n not in ("lighter_H", "darker_H")]
    host.set_sa_pdf(_sa_pdf(kept))
    base = dict(zip(DIFF32_OPS, host.base_probs()))
    assert base["lighter_H"] == base["darker_H"] == 0.0
    with pytest.raises(ValueError, match="excludes"):
        host.set_weights(np.ones(32))
    host.set_weights(host.base_probs())


@pytest.mark.parametrize(
    "mutate, exc, match",
    [
        (lambda p: p.update({("mixup", 0.5): 0.0}), KeyError, "vocabulary"),
        (lambda p: p.update({NONE_KEY: p[NONE_KEY] + 0.1}), ValueError, "sums to"),
        (lambda p: p.update({("blur", 0.25): float("nan")}), ValueError, "nan"),
    ],
)
def test_sensaug_rejects_bad_pdf(mutate, exc, match):
    host = SensAugHost(max_iters=20000)
    pdf = _sa_pdf(DIFF32_OPS)
    mutate(pdf)
    with pytest.raises(exc, match=match):
        host.set_sa_pdf(pdf)


def test_trivialaugment_diff_splits_trivialaugments_distribution():
    """Each signed op's share is split between its two directions: uniform q is
    TrivialAugmentWide's own distribution, direction by direction."""
    host, wide = TrivialAugmentDiffHost(), TrivialAugmentHost()
    assert host.none_prob == wide.none_prob == pytest.approx(1 / 14)
    probs = dict(zip(host.vocabulary(), host.base_probs()))
    for name, p in zip(wide.vocabulary(), wide.base_probs()):
        base = name.split(".", 1)[1]
        pos, neg = f"trivialaug_diff.{base}_pos", f"trivialaug_diff.{base}_neg"
        if pos in probs:
            assert probs[pos] == probs[neg] == pytest.approx(p / 2)
        else:
            assert probs[f"trivialaug_diff.{base}"] == pytest.approx(p)


def test_trivialaugment_diff_draws_one_shared_bin():
    host = TrivialAugmentDiffHost()
    bins, _ = host.draw_bins(64, np.random.default_rng(0))
    columns = np.stack([bins[name] for name in host.vocabulary()])
    assert (columns == columns[0]).all()
    assert columns.min() >= 0 and columns.max() < host.num_magnitude_bins
