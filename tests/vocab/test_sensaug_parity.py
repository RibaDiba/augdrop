"""The ported ops are Sensaug's ops: bit-identical images, labels and gradients
against third_party/Sensaug itself. Skipped when the submodule is not checked out.

Sensaug's op modules need only torch + kornia (+ torchvision for the label-safe
transforms), so they are imported straight from the submodule without mmseg.
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SENSAUG = os.path.join(ROOT, "third_party", "Sensaug")
if not os.path.isdir(os.path.join(SENSAUG, "sensaug")):
    pytest.skip("third_party/Sensaug is not checked out", allow_module_level=True)
sys.path.insert(0, SENSAUG)

from sensaug.dataset import differentiable_augmentations_aa as sensaug_aa  # noqa: E402
from sensaug.dataset.utils import label_safe_transforms  # noqa: E402

from vocab import FAMILIES, build_vocab, get_op  # noqa: E402

DIFF32 = list(sensaug_aa.DIFF32_OPS)


@pytest.fixture
def images():
    torch.manual_seed(0)
    return torch.rand(3, 3, 24, 40)


def _seeded(fn, *args):
    with torch.random.fork_rng():
        torch.manual_seed(123)
        return fn(*args)


def test_default_vocab_is_diff32_in_order():
    assert list(build_vocab()) == DIFF32


def test_geometric_family_is_sensaugs_geometric_keys():
    assert {s.name for s in FAMILIES["geometric"]} == set(sensaug_aa.GEOMETRIC_OP_KEYS)


@pytest.mark.parametrize("name", DIFF32)
@pytest.mark.parametrize("magnitude", ["scalar", "per_image"])
def test_image_is_bit_identical(images, name, magnitude):
    m = 0.37 if magnitude == "scalar" else torch.tensor([0.1, 0.55, 0.9])
    ours, _ = _seeded(get_op(name).apply, images, m)
    theirs = _seeded(sensaug_aa.DIFF32_OPS[name], images, m)
    assert torch.equal(ours, theirs)


@pytest.mark.parametrize("name", DIFF32)
def test_gradient_is_bit_identical(images, name):
    weights = torch.arange(1.0, 4.0).view(-1, 1, 1, 1)
    grads = []
    for apply in (lambda x, d: get_op(name).apply(x, d)[0], sensaug_aa.DIFF32_OPS[name]):
        delta = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
        out = _seeded(apply, images, delta)
        grads.append(torch.autograd.grad((out * weights).sum(), delta)[0])
    assert torch.equal(grads[0], grads[1])


@pytest.mark.parametrize("name", sorted(sensaug_aa.GEOMETRIC_OP_KEYS))
@pytest.mark.parametrize("label_scale", [1, 2])
def test_geometric_label_path_matches_sensaugs(images, name, label_scale):
    """What Sensaug's CollectGradientHook ran for a geometric key:
    geometric_affine_matrix + warp_image_and_label, including a label at another
    resolution than the image."""
    b, _, h, w = images.shape
    label = torch.randint(0, 19, (b, h * label_scale, w * label_scale))
    delta = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)

    ours_img, ours_label = get_op(name).apply(images, delta, label)
    matrix = sensaug_aa.geometric_affine_matrix(name, images, delta)
    theirs_img, theirs_label = sensaug_aa.warp_image_and_label(images, label, matrix)

    assert torch.equal(ours_img, theirs_img)
    assert torch.equal(ours_label, theirs_label)


LABEL_SAFE = {
    "randaug_default": label_safe_transforms.LabelSafeRandAugment,
    "trivialaug_wide": label_safe_transforms.LabelSafeTrivialAugmentWide,
}


@pytest.mark.parametrize("family", LABEL_SAFE)
@pytest.mark.parametrize("transform_id", sorted(label_safe_transforms.GEOMETRIC_TRANSFORM_IDS))
def test_torchvision_label_replay_matches_sensaugs_label_safe_transforms(family, transform_id):
    """Sensaug's label-safe RandAugment/TrivialAugmentWide replay each geometric
    op on the label; ours must move the label the same way."""
    g = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 256, (3, 40, 56), generator=g, dtype=torch.uint8)
    seg = torch.randint(0, 19, (1, 40, 56), generator=g, dtype=torch.uint8)
    spec = get_op(f"{family}.{transform_id}")
    # At the images' precision, which is how the op takes its magnitude.
    m = float(torch.tensor(spec.mag_range[1] * 0.6, dtype=torch.float32))

    ours_img, ours_label = spec.apply(pixels[None].float() / 255.0, m, seg.long())

    aug = LABEL_SAFE[family]()
    native = m * 56 / 331.0 if (family == "randaug_default" and transform_id == "TranslateX") else m
    if family == "randaug_default" and transform_id == "TranslateY":
        native = m * 40 / 331.0
    aug._seg = seg
    theirs_img = aug._apply_image_or_video_transform(pixels, transform_id, native, aug.interpolation, aug._fill)

    assert torch.equal((ours_img[0] * 255.0).round().to(torch.uint8), theirs_img)
    assert torch.equal(ours_label, aug._seg.long())
