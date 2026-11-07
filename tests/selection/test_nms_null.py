"""random_null: NMS's matched null draws by (seed, key), reproducibly, from what NMS ranked."""

from selection import NmsResult, random_null

ORDER = tuple(f"f.op{i:02d}" for i in range(12))
BEFORE = ("f.gone",)
REL = {name: 1.0 for name in ORDER + BEFORE + ("f.exempt",)}


def nms_pass(new=3):
    """An NMS pass that ranked ORDER and newly pruned its last ``new`` ops."""
    suppressed = {name: (ORDER[0], 0.9) for name in ORDER[-new:]}
    return NmsResult(
        order=ORDER,
        kept=tuple(n for n in ORDER if n not in suppressed) + ("f.exempt",),
        suppressed=suppressed,
        exempt=("f.exempt",),
        pruned_before=BEFORE,
        pruned=BEFORE + tuple(suppressed),
        score={name: float(len(ORDER) - i) for i, name in enumerate(ORDER)},
        tau=0.5,
    )


def test_same_seed_and_key_draw_the_same():
    result = nms_pass()
    first = random_null(result, REL, seed=3, key=1000)
    again = random_null(nms_pass(), dict(REL), seed=3, key=1000)
    assert first[2] == again[2]
    assert first[1].pruned == again[1].pruned
    assert list(first[0].q) == list(again[0].q)


def test_the_draw_matches_the_pass():
    selection, null, drawn = random_null(nms_pass(), REL, seed=0, key=500)
    assert len(drawn) == 3 and set(drawn) <= set(ORDER)
    assert null.pruned == BEFORE + drawn
    assert null.suppressed == {}
    assert "f.exempt" in null.kept and not set(drawn) & set(null.kept)
    assert selection.applied
    assert {n for n, v in zip(selection.names, selection.q) if v == 0} == set(BEFORE + drawn)


def test_the_key_can_change_the_draw():
    # Not every key need differ (C(12, 3) = 220 draws), but some must, or each probe of a
    # run would prune the same ops.
    draws = {random_null(nms_pass(), REL, seed=0, key=key)[2] for key in range(0, 20_000, 1000)}
    assert len(draws) > 1


def test_nothing_new_draws_nothing():
    result = nms_pass()._replace(suppressed={}, pruned=BEFORE)
    _, null, drawn = random_null(result, REL, seed=0, key=1)
    assert drawn == () and null.pruned == BEFORE
