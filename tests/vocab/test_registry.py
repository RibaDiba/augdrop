import pytest
import torch

from vocab import DEFAULT_FAMILIES, FAMILIES, OPS, OpSpec, build_vocab, get_op
from vocab.diff_ops import geometric, photometric, rotate_matrix


def test_every_op_is_indexed_once_under_its_family():
    total = sum(len(specs) for specs in FAMILIES.values())
    assert len(OPS) == total == 61
    for family, specs in FAMILIES.items():
        for spec in specs:
            assert spec.family == family
            assert OPS[spec.name] is spec


def test_default_vocab_is_sensaugs_32_in_order():
    vocab = build_vocab()
    assert DEFAULT_FAMILIES == ("legacy_hsv", "geometric", "pil_autoaug")
    assert list(vocab) == [s.name for f in DEFAULT_FAMILIES for s in FAMILIES[f]]
    assert len(vocab) == 32


def test_families_config_keeps_the_given_family_order():
    vocab = build_vocab({"families": ["pil_autoaug", "legacy_hsv"]})
    assert list(vocab) == [s.name for s in FAMILIES["pil_autoaug"] + FAMILIES["legacy_hsv"]]


def test_ops_config_keeps_the_given_op_order():
    names = ["rotate_neg", "blur", "randaug_default.Solarize"]
    assert list(build_vocab({"ops": names})) == names


def test_exclude_drops_ops_and_ignores_real_ops_outside_the_selection():
    vocab = build_vocab({"families": ["legacy_hsv"], "exclude": ["noise", "rotate_pos"]})
    assert "noise" not in vocab and len(vocab) == 13


@pytest.mark.parametrize(
    "config, error",
    [
        ({"families": ["nope"]}, KeyError),
        ({"ops": ["lighter_R", "not_an_op"]}, KeyError),
        ({"exclude": ["not_an_op"]}, KeyError),
        ({"families": ["legacy_hsv"], "ops": ["blur"]}, ValueError),
        ({"familes": ["legacy_hsv"]}, ValueError),
        ({"ops": ["blur", "blur"]}, ValueError),
        ({"families": ["legacy_hsv", "legacy_hsv"]}, ValueError),
        ({"ops": ["blur"], "exclude": ["blur"]}, ValueError),
    ],
)
def test_bad_configs_raise(config, error):
    with pytest.raises(error):
        build_vocab(config)


def test_get_op_names_the_known_ops_on_a_miss():
    assert get_op("blur") is OPS["blur"]
    with pytest.raises(KeyError, match="known ops"):
        get_op("blurr")


# --- OpSpec invariants -------------------------------------------------------


def _spec(**overrides):
    fields = dict(
        name="x", family="f", kind="photometric", source="test", mag_range=(0.0, 1.0),
        identity_at=0.0, grad_mode="autograd", warps_label=False,
        fn=photometric(lambda images, d: images),
    )
    fields.update(overrides)
    return OpSpec(**fields)


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "colour"},
        {"grad_mode": "sometimes"},
        {"kind": "geometric", "warps_label": False},  # a geometric op must move the label
        {"warps_label": True},  # a photometric op must not
        {"mag_range": (1.0, 0.0)},
        {"identity_at": 2.0},
        {"identity_at": None},
        {"mag_range": None},
    ],
)
def test_malformed_specs_cannot_be_constructed(overrides):
    with pytest.raises(ValueError):
        _spec(**overrides)


def test_a_well_formed_geometric_spec_constructs():
    spec = _spec(kind="geometric", warps_label=True, fn=geometric(rotate_matrix))
    assert spec.has_magnitude


def test_apply_requires_a_magnitude_when_the_op_has_one():
    with pytest.raises(ValueError, match="takes a magnitude"):
        OPS["blur"].apply(torch.rand(1, 3, 8, 8))
