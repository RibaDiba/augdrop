"""collect_grads: one clean batch -> the batch's segment of every row of G.

Several of these are ported from Sensaug's tests/test_grad_hook.py; the rest cover
what dropaug adds (finite differences, unprobed ops, runtime coupling checks).
"""

import numpy as np
import pytest
import torch

from probe.grad_collect import ProbeError, collect_grads, op_seed

from probe_helpers import (
    Op,
    ScalarLossAdapter,
    ToyAdapter,
    darker_g,
    enhance,
    lighter_r,
    make_batch,
    no_grad_path,
    noise,
    pooled,
    translate_x,
    unwarped_translate_x,
)


def _fixed(ops, value, batch_size=4):
    return {name: np.full(batch_size, value) for name in ops}


def _grad_wrt_delta(adapter, batch, deltas, fn=lighter_r):
    delta = deltas.clone().requires_grad_(True)
    loss = adapter.loss(fn(batch["image"], delta), batch["label"], batch).sum()
    return torch.autograd.grad(loss, delta)[0]


# --- the premise: per-image independence ------------------------------------------


def test_eval_mode_makes_per_image_gradients_independent(adapter, batch):
    """The assumption the whole probe rests on: d loss / d delta[i] depends on image
    i's delta alone. Bumping delta[2] must move entry 2 of the gradient and nothing
    else."""
    base = _grad_wrt_delta(adapter, batch, torch.tensor([0.5, 0.5, 0.5, 0.5]))
    bumped = _grad_wrt_delta(adapter, batch, torch.tensor([0.5, 0.5, 0.9, 0.5]))

    for i in (0, 1, 3):
        assert torch.allclose(base[i], bumped[i], atol=1e-6), f"image {i} moved"
    assert not torch.isclose(base[2], bumped[2], atol=1e-6)


def test_train_mode_batchnorm_couples_images(adapter, batch):
    """The counterexample, which is why eval() is load-bearing. If this ever starts
    passing without coupling, the eval() requirement could be revisited."""
    adapter.model.train()
    base = _grad_wrt_delta(adapter, batch, torch.tensor([0.5, 0.5, 0.5, 0.5]))
    bumped = _grad_wrt_delta(adapter, batch, torch.tensor([0.5, 0.5, 0.9, 0.5]))
    assert not torch.allclose(base[0], bumped[0], atol=1e-6)


def test_refuses_a_model_with_any_module_in_train_mode(adapter, batch, ops):
    adapter.model[1].train()  # just the BatchNorm, root still says eval
    with pytest.raises(ProbeError, match="train mode"):
        collect_grads(adapter, batch, ops, _fixed(ops, 0.3))


# --- the rows -------------------------------------------------------------------


def test_one_finite_value_per_image_for_every_op(adapter, batch, ops):
    rows = collect_grads(adapter, batch, ops, _fixed(ops, 0.3))

    assert list(rows) == list(ops)
    for name, row in rows.items():
        assert row.shape == (4,) and row.dtype == np.float64, name
        assert np.isfinite(row).all(), name
        # The images differ, so their sensitivities should too; a constant vector
        # would mean the per-image delta collapsed somewhere.
        assert row.std() > 0, name


def test_accepts_scalar_or_per_image_magnitudes(adapter, batch, ops):
    scalar = collect_grads(adapter, batch, ops, {name: 0.3 for name in ops})
    vector = collect_grads(adapter, batch, ops, _fixed(ops, 0.3))
    for name in ops:
        np.testing.assert_array_equal(scalar[name], vector[name])


def test_an_op_without_a_magnitude_gets_a_nan_row(adapter, batch, ops):
    """NaN, never 0: a zero row reads as a real gradient of exactly zero."""
    ops = {**ops, "equalize": Op("equalize", lambda x, m: x, mag_range=None, identity_at=None, grad_mode="none")}
    rows = collect_grads(adapter, batch, ops, _fixed(ops, 0.3))
    assert np.isnan(rows["equalize"]).all()
    assert np.isfinite(rows["lighter_R"]).all()


def test_scalar_batch_mean_loss_gives_the_per_image_gradients_over_b(batch, ops):
    """An mmseg-style batch-mean loss still yields per-image gradients (in eval mode),
    scaled by 1/B, which the analysis's per-image normalization divides out."""
    per_image = collect_grads(ToyAdapter(), batch, ops, _fixed(ops, 0.3))
    scalar = collect_grads(ScalarLossAdapter(), batch, ops, _fixed(ops, 0.3))
    for name in ops:
        np.testing.assert_allclose(scalar[name], per_image[name] / 4, rtol=1e-4, atol=1e-8)


# --- randomness ---------------------------------------------------------------------


def test_noise_draws_the_same_sample_every_batch(adapter, batch, ops):
    first = collect_grads(adapter, batch, ops, _fixed(ops, 0.3), probe_seed=7)
    second = collect_grads(adapter, batch, ops, _fixed(ops, 0.3), probe_seed=7)
    np.testing.assert_array_equal(first["noise"], second["noise"])


def test_noise_sample_does_not_depend_on_which_other_ops_run(adapter, batch, ops):
    """Seeded per op, not per batch: pruning or reordering the vocab must not change
    the sample one random op draws."""
    alone = {"noise": ops["noise"]}
    with_others = collect_grads(adapter, batch, ops, _fixed(ops, 0.3))
    by_itself = collect_grads(adapter, batch, alone, _fixed(alone, 0.3))
    np.testing.assert_array_equal(with_others["noise"], by_itself["noise"])


def test_two_random_ops_do_not_share_a_sample():
    assert op_seed(0, "noise") != op_seed(0, "speckle")
    assert op_seed(0, "noise") != op_seed(1, "noise")


def test_restores_the_rng_state(adapter, batch, ops):
    before = torch.get_rng_state()
    collect_grads(adapter, batch, ops, _fixed(ops, 0.3), probe_seed=1234)
    assert torch.equal(torch.get_rng_state(), before)


# --- labels -------------------------------------------------------------------------


def test_geometric_op_hands_the_loss_a_warped_label(adapter, batch):
    ops = {
        "lighter_R": Op("lighter_R", lighter_r),
        "translate_x": Op("translate_x", translate_x, kind="geometric"),
    }
    collect_grads(adapter, batch, ops, _fixed(ops, 1.0))

    photometric_label, geometric_label = adapter.seen_labels
    assert photometric_label is batch["label"]  # the clean label, untouched
    assert geometric_label is not batch["label"]
    # Shifted right by a quarter of the width: the vacated columns are ignore, and
    # the rest is the original label moved over.
    shift = 3  # 1.0 * 0.25 * 12
    assert (geometric_label[..., :shift] == 255).all()
    torch.testing.assert_close(geometric_label[..., shift:], batch["label"][..., :-shift])


def test_a_geometric_op_that_leaves_its_label_behind_is_refused(adapter, batch):
    ops = {"translate_x": Op("translate_x", unwarped_translate_x, kind="geometric")}
    with pytest.raises(ProbeError, match="unwarped"):
        collect_grads(adapter, batch, ops, _fixed(ops, 0.5))


# --- finite differences ---------------------------------------------------------------


def test_finite_differences_agree_with_autograd_on_a_smooth_op(adapter, batch):
    ops = {
        "autograd": Op("autograd", darker_g),
        "fd": Op("fd", no_grad_path(darker_g), grad_mode="none"),
    }
    rows = collect_grads(adapter, batch, ops, _fixed(ops, 0.4), fd_step=0.01)
    np.testing.assert_allclose(rows["fd"], rows["autograd"], rtol=1e-2, atol=1e-5)


def test_finite_differences_use_one_noise_sample_on_both_sides(adapter, batch):
    """Without common random numbers the difference is the change of sample over 2h,
    which dwarfs the real slope. Same name, so both ops draw the same sample."""
    autograd = collect_grads(adapter, batch, {"noise": Op("noise", noise)}, {"noise": 0.3})
    fd_op = {"noise": Op("noise", no_grad_path(noise), grad_mode="none")}
    fd = collect_grads(adapter, batch, fd_op, {"noise": 0.3}, fd_step=0.01)
    np.testing.assert_allclose(fd["noise"], autograd["noise"], rtol=5e-2, atol=1e-4)


def test_finite_difference_step_scales_with_the_ops_own_range(adapter, batch):
    """An enhance factor in [0.1, 1.9]: the default step is 5% of the width, 0.09
    factor units, not 0.05, which would mean something else in every family."""
    seen = []

    def spy(images, factor):
        seen.append(float(torch.as_tensor(factor)[0]))
        return enhance(images, torch.as_tensor(factor).detach())

    ops = {"fd": Op("fd", spy, mag_range=(0.1, 1.9), identity_at=1.0, grad_mode="none")}
    collect_grads(adapter, batch, ops, _fixed(ops, 0.8))
    np.testing.assert_allclose(sorted(seen), [0.71, 0.89], rtol=1e-6)

    seen.clear()
    collect_grads(adapter, batch, ops, _fixed(ops, 0.8), fd_step={"fd": 0.25})
    np.testing.assert_allclose(sorted(seen), [0.35, 1.25], rtol=1e-6)


def test_finite_differences_need_a_per_image_loss_above_batch_size_one(batch):
    ops = {"fd": Op("fd", no_grad_path(lighter_r), grad_mode="none")}
    with pytest.raises(ProbeError, match="per-image"):
        collect_grads(ScalarLossAdapter(), batch, ops, _fixed(ops, 0.3))
    single = make_batch(n=1)
    row = collect_grads(ScalarLossAdapter(), single, ops, _fixed(ops, 0.3, batch_size=1))
    assert row["fd"].shape == (1,)


# --- guards -------------------------------------------------------------------------


def test_non_finite_gradient_raises(adapter, batch):
    ops = {"explode": Op("explode", lambda x, m: x * (_col(m) / 0.0))}
    with pytest.raises(ProbeError, match="non-finite"):
        collect_grads(adapter, batch, ops, _fixed(ops, 0.5))


def test_an_autograd_op_with_no_gradient_path_is_refused(adapter, batch):
    ops = {"liar": Op("liar", no_grad_path(lighter_r))}  # says autograd, isn't
    with pytest.raises(ProbeError, match="does not depend on the magnitude"):
        collect_grads(adapter, batch, ops, _fixed(ops, 0.5))


def test_magnitudes_are_checked_against_the_ops_range(adapter, batch, ops):
    with pytest.raises(ProbeError, match="outside its mag_range"):
        collect_grads(adapter, batch, ops, _fixed(ops, 1.5))
    with pytest.raises(ProbeError, match="no magnitude"):
        collect_grads(adapter, batch, ops, {"lighter_R": 0.3})


def test_images_must_arrive_in_unit_range(batch, ops):
    class Unnormalized(ToyAdapter):
        def to_unit(self, batch):
            return batch["image"] * 255.0, batch["label"]

    with pytest.raises(ProbeError, match=r"\[0, 1\]"):
        collect_grads(Unnormalized(), batch, ops, _fixed(ops, 0.3))


def test_check_catches_augmentation_left_in_the_preprocessor(batch, ops):
    class Augmenting(ToyAdapter):
        def to_unit(self, batch):
            jitter = 0.1 * torch.rand(len(batch["image"]), 1, 1, 1)
            return (batch["image"] * (0.9 + jitter)).clamp(0, 1), batch["label"]

    collect_grads(Augmenting(), batch, ops, _fixed(ops, 0.3))  # unchecked: runs
    with pytest.raises(ProbeError, match="CLEAN"):
        collect_grads(Augmenting(), batch, ops, _fixed(ops, 0.3), check=True)


def test_check_catches_an_op_that_pools_the_magnitude(adapter, batch, ops):
    """The bug only this catches: a pooled magnitude still yields a (B,) gradient of
    the right shape and finite values."""
    collect_grads(adapter, batch, ops, _fixed(ops, 0.3), check=True)  # clean ops pass
    bad = {"pooled": Op("pooled", pooled)}
    with pytest.raises(ProbeError, match="couples the images"):
        collect_grads(adapter, batch, bad, _fixed(bad, 0.3), check=True)


def _col(m):
    return torch.as_tensor(m).reshape(-1, 1, 1, 1)
