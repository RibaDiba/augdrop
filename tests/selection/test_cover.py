"""cover_single / cover_even: the same ops lose the same mass; only where it goes differs."""

import numpy as np
import pytest

from hosts import build_host
from selection import COVER_SELECTORS, SELECTORS, Redundancy, build_selector

NAMES = ("f.A", "f.B", "f.C_pos", "f.C_neg", "f.D")


def redundancy(cells, names=NAMES, survives=None):
    """R with ``{(a, b): (r, ci_lo)}`` set symmetrically, everything else r = 0 and
    ci_lo = -0.5."""
    n = len(names)
    r = np.eye(n)
    lo = np.full((n, n), -0.5)
    index = {name: i for i, name in enumerate(names)}
    for (a, b), (value, low) in cells.items():
        i, j = index[a], index[b]
        r[i, j] = r[j, i] = value
        lo[i, j] = lo[j, i] = low
    return Redundancy(names, r, survives=survives, ci_lo=lo)


def flat(names=NAMES):
    return {name: 1.0 / len(names) for name in names}


# --- shared: which ops move, and by how much ----------------------------------------


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_registered_as_hard(name):
    assert SELECTORS[name].hard


@pytest.mark.parametrize("name", COVER_SELECTORS)
@pytest.mark.parametrize("lam", [0.25, 1.0])
def test_moved_op_ends_at_one_minus_lam_of_its_prior(name, lam):
    R = redundancy({("f.A", "f.D"): (0.8, 0.6)})
    sel = build_selector(name, lam=lam).select(R, flat())
    assert sel.covers == (("f.D", "f.A"),)  # tie on prior: the later name moves
    assert sel.as_dict()["f.D"] == pytest.approx(0.2 * (1 - lam))
    assert sel.q.sum() == pytest.approx(1.0)
    assert sel.pruned == (("f.D",) if lam == 1.0 else ())


@pytest.mark.parametrize("lam", [0.25, 1.0])
def test_both_arms_move_the_same_ops(lam):
    R = redundancy(
        {("f.A", "f.B"): (0.9, 0.8), ("f.B", "f.D"): (0.85, 0.7), ("f.A", "f.D"): (0.6, 0.4)}
    )
    single, even = (build_selector(n, lam=lam).select(R, flat()) for n in COVER_SELECTORS)
    assert single.covers == even.covers
    assert single.pruned == even.pruned
    for moved, _ in single.covers:
        assert single.as_dict()[moved] == pytest.approx(even.as_dict()[moved])


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_smaller_prior_moves(name):
    R = redundancy({("f.A", "f.D"): (0.8, 0.6)})
    rel = {"f.A": 0.1, "f.B": 0.2, "f.C_pos": 0.2, "f.C_neg": 0.2, "f.D": 0.3}
    assert build_selector(name, lam=1.0).select(R, rel).covers == (("f.A", "f.D"),)


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_effect_size_gate_not_significance(name):
    # r is large and survives FDR, but the CI's lower bound is under tau.
    R = redundancy({("f.A", "f.D"): (0.5, 0.25)})
    sel = build_selector(name, lam=1.0, tau=0.3).select(R, flat())
    assert not sel.applied
    assert "tau" in sel.reason


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_fdr_gate_respected(name):
    survives = np.zeros((5, 5), dtype=bool)
    R = redundancy({("f.A", "f.D"): (0.8, 0.6)}, survives=survives)
    assert not build_selector(name, lam=1.0).select(R, flat()).applied


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_within_op_halves_are_never_a_pair(name):
    R = redundancy({("f.C_pos", "f.C_neg"): (0.9, 0.8)})
    assert not build_selector(name, lam=1.0).select(R, flat()).applied


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_no_chains_and_strongest_first(name):
    # A-B is the strongest pair: B moves, A is pinned. B-D can't use B (moved);
    # A-D moves D, since A is pinned.
    R = redundancy(
        {("f.A", "f.B"): (0.9, 0.8), ("f.B", "f.D"): (0.85, 0.7), ("f.A", "f.D"): (0.6, 0.4)}
    )
    sel = build_selector(name, lam=1.0).select(R, flat())
    assert dict(sel.covers) == {"f.B": "f.A", "f.D": "f.A"}


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_declines(name):
    R = redundancy({("f.A", "f.D"): (0.8, 0.6)})
    assert not build_selector(name, lam=0.0).select(R, flat()).applied
    bare = Redundancy(R.names, R.r)
    assert "CI" in build_selector(name, lam=1.0).select(bare, flat()).reason


@pytest.mark.parametrize("name", COVER_SELECTORS)
@pytest.mark.parametrize("lam", [None, -0.1, 1.5, float("nan")])
def test_lam_validated(name, lam):
    with pytest.raises(ValueError):
        build_selector(name, lam=lam)


# --- where the mass goes ----------------------------------------------------------


def test_single_gives_the_mass_to_the_cover_only():
    R = redundancy({("f.A", "f.D"): (0.8, 0.6)})
    q = build_selector("cover_single", lam=1.0).select(R, flat()).as_dict()
    assert q["f.A"] == pytest.approx(0.4)
    for name in ("f.B", "f.C_pos", "f.C_neg"):
        assert q[name] == pytest.approx(0.2)


def test_even_spreads_the_mass_by_prior_over_every_op_not_moved():
    R = redundancy({("f.A", "f.D"): (0.8, 0.6)})
    rel = {"f.A": 0.1, "f.B": 0.2, "f.C_pos": 0.2, "f.C_neg": 0.2, "f.D": 0.3}
    sel = build_selector("cover_even", lam=0.5).select(R, rel)
    assert sel.covers == (("f.A", "f.D"),)
    q = sel.as_dict()
    assert q["f.A"] == pytest.approx(0.05)
    # 0.05 taken, spread over B, C_pos, C_neg, D (0.9 of prior) by their prior.
    for name, p in rel.items():
        if name != "f.A":
            assert q[name] == pytest.approx(p * (1 + 0.05 / 0.9))
    # Their ratios to each other are the prior's.
    assert q["f.D"] / q["f.B"] == pytest.approx(1.5)


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_trivialaugment_diff_q_installs_and_holds_none_prob(name):
    host = build_host("trivialaugment_diff")
    names = host.vocabulary()
    cells = {("trivialaug_diff.Brightness_neg", "trivialaug_diff.Contrast_neg"): (0.76, 0.55)}
    R = redundancy(cells, names=names)
    base = dict(zip(names, host.base_probs()))
    sel = build_selector(name, lam=1.0).select(R, dict(base))
    assert sel.covers == (("trivialaug_diff.Contrast_neg", "trivialaug_diff.Brightness_neg"),)
    none_before = host.none_prob
    host.set_weights(sel.q)
    probs = dict(zip(names, host.sampling_probs()))
    assert host.none_prob == none_before
    assert probs["trivialaug_diff.Contrast_neg"] == 0.0
    moved = base["trivialaug_diff.Contrast_neg"]
    others = [n for n in names if n not in ("trivialaug_diff.Contrast_neg", "trivialaug_diff.Brightness_neg")]
    if name == "cover_single":
        assert probs["trivialaug_diff.Brightness_neg"] == pytest.approx(2 * base["trivialaug_diff.Brightness_neg"])
        for n in others:
            assert probs[n] == pytest.approx(base[n])
    else:
        for n in others + ["trivialaug_diff.Brightness_neg"]:
            assert probs[n] == pytest.approx(base[n] / (1 - moved))


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_priors_equal_up_to_round_off_are_a_tie(name):
    R = redundancy({("f.A", "f.D"): (0.8, 0.6)})
    rel = flat()
    rel["f.A"] = np.nextafter(rel["f.A"], 0.0)  # one ulp smaller: still a tie
    assert build_selector(name, lam=1.0).select(R, rel).covers == (("f.D", "f.A"),)


@pytest.mark.parametrize("name", COVER_SELECTORS)
def test_sensaug_q_is_per_op_and_keeps_each_ops_level_shape(name):
    from hosts.sensaug import NONE_KEY
    from sa import generate_pdf_new, none_prob_denominator

    host = build_host("sensaug", max_iters=2000)
    names = host.vocabulary()
    ops = [n for n in names if n not in ("lighter_H", "darker_H")]  # --no-inv-aug
    rng = np.random.default_rng(0)
    miou = {op: {level: float(rng.uniform(30, 70)) for level in (0.2, 0.4, 0.6, 0.8)} for op in ops}
    pdf = generate_pdf_new(miou, True, none_prob_denominator(len(ops)))
    host.set_sa_pdf(pdf)
    base = dict(zip(names, host.base_probs()))

    R = redundancy({("blur", "noise"): (0.8, 0.5)}, names=names)
    sel = build_selector(name, lam=1.0).select(R, dict(base))
    assert sel.covers == (("noise", "blur"),)  # every op has the same allotment: the later name moves
    assert sel.pruned == ("noise",)
    host.set_weights(sel.q)
    tilted = host.pdf()

    assert tilted[NONE_KEY] == pdf[NONE_KEY]
    assert sum(tilted.values()) == pytest.approx(1.0)
    others = 1.0 / (1.0 - base["noise"])
    for op in ops:
        scale = np.array([tilted[key] / pdf[key] for key in pdf if key[0] == op])
        if op == "noise":
            assert (scale == 0.0).all()
            continue
        if name == "cover_single":
            expected = 2.0 if op == "blur" else 1.0
        else:
            expected = others
        # One factor per op: its levels keep the shape the SA pdf gave them.
        assert scale == pytest.approx(np.full(len(scale), expected), rel=1e-12)


@pytest.mark.parametrize(
    "host, ok", [("sensaug", True), ("trivialaugment_diff", True), ("autoaugment", False), ("uniform", False)]
)
def test_config_check_accepts_cover_on_trivialaugment_and_sensaug_only(host, ok):
    pytest.importorskip("hydra")
    from dropaug.config import ConfigError, check, compose

    for selector in COVER_SELECTORS:
        cfg = compose([f"host={host}", f"selector={selector}"])
        if ok:
            check(cfg, training=False)
        else:
            with pytest.raises(ConfigError, match="per-op q"):
                check(cfg, training=False)
