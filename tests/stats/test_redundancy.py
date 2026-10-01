"""Tests for stats/redundancy.py, ported from Sensaug's tests/test_redundancy.py (the
compute_red half; reweight lives with the hosts), plus parity against the original.
"""

import os
import sys

import numpy as np
import pytest

from stats.redundancy import MODES, compute_red, summarise, within_op_pairs

SENSAUG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party",
    "Sensaug",
)

NAMES_14 = [
    "lighter_R", "darker_R", "lighter_G", "darker_G", "lighter_B", "darker_B",
    "lighter_H", "darker_H", "lighter_S", "darker_S", "lighter_V", "darker_V",
    "blur", "noise",
]  # fmt: skip

NAMES_32 = NAMES_14 + [
    "rotate_pos", "rotate_neg", "shear_x_pos", "shear_x_neg",
    "shear_y_pos", "shear_y_neg", "translate_x_pos", "translate_x_neg",
    "translate_y_pos", "translate_y_neg", "brightness_pos", "brightness_neg",
    "contrast_pos", "contrast_neg", "sharpness_pos", "sharpness_neg",
    "color_pos", "color_neg",
]  # fmt: skip


def _random_r(names, seed=0):
    """A symmetric correlation-shaped matrix with a unit diagonal."""
    rng = np.random.default_rng(seed)
    n = len(names)
    m = rng.normal(0, 0.3, size=(n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return np.clip(m, -1.0, 1.0)


# --- within-op pair derivation ------------------------------------------------


def test_within_op_pairs_finds_six_in_the_legacy_vocabulary():
    """The RGB/HSV lighter/darker pairs, and nothing else -- blur and noise are
    unsigned and have no partner."""
    pairs = within_op_pairs(NAMES_14)
    assert len(pairs) == 6
    for i, j in pairs:
        assert NAMES_14[i].startswith("lighter_")
        assert NAMES_14[j].startswith("darker_")


def test_within_op_pairs_finds_fifteen_in_the_merged_vocabulary():
    """6 lighter/darker + 5 geometric _pos/_neg + 4 photometric _pos/_neg. 15 of
    the 496 off-diagonal cells, which is why this is a footnote rather than a
    headline: it moves the raw row-sum mean by about a tenth and the standard
    deviation barely at all."""
    assert len(within_op_pairs(NAMES_32)) == 15


def test_within_op_pairs_ignores_a_partnerless_name():
    """A vocabulary subset (--photometric-only, say) can contain one half of a pair
    without the other. Pairing it against a missing index would raise."""
    assert within_op_pairs(["lighter_R", "blur", "color_pos"]) == []


# --- compute_red --------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_red_is_standardized_in_every_mode(mode):
    """The premise that makes one lambda portable across runs and checkpoints. The
    raw row sums are not comparable -- on the logged matrices their standard
    deviations span 0.24 to 3.69 -- so lambda tuned on one would be meaningless on
    the next."""
    score = compute_red(_random_r(NAMES_32), NAMES_32, mode=mode)

    assert score.std.mean() == pytest.approx(0.0, abs=1e-9)
    assert score.std.std() == pytest.approx(1.0, rel=1e-6)


def test_red_ignores_the_diagonal():
    """r[a, a] is 1 by construction and identical for every op, so including it
    would add a constant -- harmless after standardization, but only by accident."""
    r = _random_r(NAMES_14)
    with_unit_diagonal = compute_red(r, NAMES_14)

    r_scaled = r.copy()
    np.fill_diagonal(r_scaled, 0.5)
    assert np.allclose(with_unit_diagonal.raw, compute_red(r_scaled, NAMES_14).raw)


def test_red_excludes_within_op_cells_when_asked():
    """An op's two directions must not count as evidence that the op is redundant
    with the bank."""
    r = _random_r(NAMES_32)
    masked = compute_red(r, NAMES_32, mask_within_op=True)
    unmasked = compute_red(r, NAMES_32, mask_within_op=False)

    assert not np.allclose(masked.raw, unmasked.raw)
    # blur and noise have no partner, so their rows are untouched either way.
    blur = NAMES_32.index("blur")
    assert masked.raw[blur] == pytest.approx(unmasked.raw[blur])


def test_red_applies_the_fdr_survivor_mask():
    """Cells that did not survive multiplicity correction at 496 simultaneous
    tests contribute nothing."""
    r = _random_r(NAMES_14)
    survivors = np.zeros_like(r, dtype=bool)
    # lighter_R/lighter_G, deliberately NOT a within-op pair -- picking one would
    # test nothing, since the within-op mask zeroes it before the survivor mask
    # is ever consulted.
    survivors[0, 2] = survivors[2, 0] = True

    score = compute_red(r, NAMES_14, survivor_mask=survivors)

    assert score.raw[0] > 0 and score.raw[2] > 0
    assert score.raw[[1, *range(3, 14)]].sum() == pytest.approx(0.0)


def test_the_within_op_mask_is_applied_before_the_survivor_mask():
    """Order matters and the composition is AND, not OR: a within-op cell that
    survives FDR must still be excluded. Surviving multiplicity correction says the
    correlation is real, which is exactly what a parameterization artefact looks
    like."""
    r = _random_r(NAMES_32)
    survivors = np.ones_like(r, dtype=bool)

    both = compute_red(r, NAMES_32, survivor_mask=survivors, mask_within_op=True)
    fdr_only = compute_red(r, NAMES_32, survivor_mask=survivors, mask_within_op=False)

    assert not np.allclose(both.raw, fdr_only.raw)
    assert np.allclose(both.raw, compute_red(r, NAMES_32, mask_within_op=True).raw)


def test_nan_cells_contribute_zero_rather_than_poisoning_the_row():
    """correlate() leaves a dropped op's row and column NaN by design, and the json
    log writes them as null. A single NaN propagating through the row sum would take
    out every op that op touches, i.e. all of them."""
    r = _random_r(NAMES_14)
    r[3, :] = np.nan
    r[:, 3] = np.nan

    score = compute_red(r, NAMES_14)

    assert np.isfinite(score.raw).all()
    assert np.isfinite(score.std).all()
    assert score.raw[3] == pytest.approx(0.0)


def test_a_fully_nan_row_is_reported_as_dropped():
    """Distinguishable from an op that genuinely correlates with nothing -- both
    score 0, but only one of them was measured."""
    r = _random_r(NAMES_14)
    r[5, :] = np.nan
    r[:, 5] = np.nan

    assert compute_red(r, NAMES_14).dropped == ["darker_B"]
    assert compute_red(_random_r(NAMES_14), NAMES_14).dropped == []


def test_squared_and_abs_are_sign_blind_but_signed_is_not():
    """Why `squared` is the default: it cannot be flipped by a sign-convention
    audit landing the other way, and it does not depend on how closure resolves."""
    r = _random_r(NAMES_14)
    flipped = -r.copy()
    np.fill_diagonal(flipped, 1.0)

    assert np.allclose(
        compute_red(r, NAMES_14, mode="squared").raw,
        compute_red(flipped, NAMES_14, mode="squared").raw,
    )
    assert np.allclose(
        compute_red(r, NAMES_14, mode="abs").raw,
        compute_red(flipped, NAMES_14, mode="abs").raw,
    )
    assert not np.allclose(
        compute_red(r, NAMES_14, mode="signed").raw,
        compute_red(flipped, NAMES_14, mode="signed").raw,
    )


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown red mode"):
        compute_red(_random_r(NAMES_14), NAMES_14, mode="cubed")


def test_mismatched_names_and_matrix_raise():
    """The one error that would otherwise be silent and catastrophic: scoring a
    32x32 matrix against 14 names would mislabel every op."""
    with pytest.raises(ValueError, match="14 names"):
        compute_red(_random_r(NAMES_32), NAMES_14)


# --- the degenerate guards ----------------------------------------------------


def test_an_all_zero_matrix_reports_a_reason_rather_than_a_silent_no_op():
    """The failure mode most likely to waste a week: reweighting runs, changes
    nothing, and the run looks like a real experimental arm."""
    score = compute_red(np.eye(len(NAMES_14)), NAMES_14)

    assert not score.usable
    assert "no cell survived" in score.reason


def test_zero_survivors_reports_a_reason():
    score = compute_red(
        _random_r(NAMES_14),
        NAMES_14,
        survivor_mask=np.zeros((len(NAMES_14),) * 2, dtype=bool),
    )
    assert not score.usable
    assert "FDR" in score.reason


def test_a_uniform_matrix_reports_zero_variance():
    """Every op equally redundant. Standardization would divide by ~0 and amplify
    float noise into a ranking that means nothing."""
    r = np.full((len(NAMES_14),) * 2, 0.4)
    np.fill_diagonal(r, 1.0)

    score = compute_red(r, NAMES_14, mask_within_op=False)

    assert not score.usable
    assert "standard deviation" in score.reason


def test_a_healthy_matrix_reports_no_reason():
    assert compute_red(_random_r(NAMES_32), NAMES_32).usable


# --- the logged summary -------------------------------------------------------


def test_summarise_names_the_extremes_of_a_usable_score():
    score = compute_red(_random_r(NAMES_32), NAMES_32)
    line = summarise(score)

    assert "most redundant" in line and "least" in line
    assert str(len(NAMES_32)) in line


def test_summarise_reports_the_reason_when_the_score_is_unusable():
    line = summarise(compute_red(np.eye(len(NAMES_14)), NAMES_14))
    assert "unusable" in line


# --- parity with Sensaug ------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("mask_within_op", [True, False])
@pytest.mark.parametrize("gated", [False, True])
def test_compute_red_matches_sensaug(mode, mask_within_op, gated):
    """sensaug/redundancy.py is pure numpy, so the original is imported straight
    from the submodule."""
    if not os.path.isdir(os.path.join(SENSAUG, "sensaug")):
        pytest.skip("third_party/Sensaug is not checked out")
    if SENSAUG not in sys.path:
        sys.path.insert(0, SENSAUG)
    from sensaug import redundancy as original

    r = _random_r(NAMES_32, seed=4)
    r[7, :] = r[:, 7] = np.nan
    rng = np.random.default_rng(5)
    survivors = rng.uniform(size=r.shape) < 0.3 if gated else None
    if gated:
        survivors = survivors | survivors.T

    ours = compute_red(r, NAMES_32, mode, mask_within_op, survivors)
    theirs = original.compute_red(r, NAMES_32, mode, mask_within_op, survivors)

    np.testing.assert_array_equal(ours.raw, theirs.raw)
    np.testing.assert_array_equal(ours.std, theirs.std)
    assert (ours.dropped, ours.mode, ours.reason) == (theirs.dropped, theirs.mode, theirs.reason)
    assert within_op_pairs(NAMES_32) == original.within_op_pairs(NAMES_32)
    assert summarise(ours) == original.summarise(theirs)
