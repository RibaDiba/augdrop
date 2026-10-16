"""dropaug.eval.corruptions: names and mCE, the pre-rendered readers on synthetic trees,
and the on-the-fly perturbation. The real backends (imagecorruptions, cv2) are optional
and their tests skip without them."""

import sys

import numpy as np
import pytest
import torch
from PIL import Image

from dropaug.eval import corruptions as C


def test_names_categories_and_baseline_line_up():
    assert len(C.IMAGENET_C) == 15 and len(C.IMAGENET_C_EXTRA) == 4
    assert tuple(C.ALEXNET_ERR) == C.IMAGENET_C
    categorized = [name for names in C.CATEGORIES.values() for name in names]
    assert sorted(categorized) == sorted(C.ALL_CORRUPTIONS)
    assert set(C.SENSAUG_UNSEEN) <= set(C.IMAGENET_C)
    assert C.category("fog") == "weather" and C.category(C.COMBINATION) == "combination"
    assert C.category("not_a_corruption") is None
    assert len(C.COMBINATION_PARAMETERS) == len(C.COMBINATION_LEVELS) == 6


def test_mce_is_100_at_alexnets_errors_and_needs_the_full_benchmark():
    alexnet = {name: [100.0 * (1 - err)] * 5 for name, err in C.ALEXNET_ERR.items()}
    out = C.mce(alexnet)
    assert out["mCE"] == pytest.approx(100.0)
    assert all(ce == pytest.approx(100.0) for ce in out["CE"].values())
    halved = {name: [100.0 * (1 - err / 2)] * 5 for name, err in C.ALEXNET_ERR.items()}
    assert C.mce(halved)["mCE"] == pytest.approx(50.0)
    with pytest.raises(ValueError, match="missing"):
        C.mce({name: accs for name, accs in alexnet.items() if name != "fog"})
    with pytest.raises(ValueError, match="severity"):
        C.mce({**alexnet, "fog": [50.0] * 4})


# --- pre-rendered readers -------------------------------------------------------------


def test_cifar_c_slices_one_severity(tmp_path):
    n = 4
    images = np.zeros((5 * n, 8, 8, 3), dtype=np.uint8)
    for s in range(5):
        images[s * n : (s + 1) * n] = 10 * (s + 1)
    labels = np.tile(np.arange(n), 5)
    np.save(tmp_path / "fog.npy", images)
    np.save(tmp_path / "labels.npy", labels)

    data = C.CifarC(str(tmp_path), "fog", 3)
    assert len(data) == n
    item = data[2]
    assert item["image"].shape == (3, 8, 8) and item["image"].dtype == torch.float32
    assert torch.allclose(item["image"], torch.full((3, 8, 8), 30 / 255))
    assert item["target"] == 2 and item["id"] == "2"
    with pytest.raises(ValueError):
        C.CifarC(str(tmp_path), "fog", 6)


def test_cifar_c_refuses_a_file_that_is_not_five_severities(tmp_path):
    np.save(tmp_path / "fog.npy", np.zeros((7, 8, 8, 3), dtype=np.uint8))
    np.save(tmp_path / "labels.npy", np.zeros(7, dtype=np.int64))
    with pytest.raises(ValueError, match="divisible"):
        C.CifarC(str(tmp_path), "fog", 1)


def _png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def test_imagenet_c_reads_sorted_wnids(tmp_path):
    for wnid, value in (("n02", 200), ("n01", 100)):
        for k in range(2):
            _png(tmp_path / "fog" / "3" / wnid / f"img{k}.png", np.full((6, 6, 3), value, dtype=np.uint8))
    (tmp_path / "fog" / "3" / "n01" / "notes.txt").write_text("not an image")

    data = C.ImageNetC(str(tmp_path), "fog", 3)
    assert data.classes == ["n01", "n02"] and len(data) == 4
    item = data[3]
    assert item["target"] == 1 and item["id"] == "n02/img1.png"
    assert torch.allclose(item["image"], torch.full((3, 6, 6), 200 / 255))

    subset = C.ImageNetC(str(tmp_path), "fog", 3, classes=["n02"])
    assert len(subset) == 2 and subset[0]["target"] == 0
    with pytest.raises(FileNotFoundError):
        C.ImageNetC(str(tmp_path), "fog", 4)


def _cityscapes_tree(root, image_value):
    _png(
        root / "leftImg8bit" / "val" / "aachen" / "a_000_leftImg8bit.png",
        np.full((4, 6, 3), image_value, dtype=np.uint8),
    )
    label = np.array([[0, 1, 2, 255, 7, 18]] * 4, dtype=np.uint8)
    _png(root / "gtFine" / "val" / "aachen" / "a_000_gtFine_labelTrainIds.png", label)


def test_cityscapes_and_cityscapes_c_share_labels(tmp_path):
    _cityscapes_tree(tmp_path / "cityscapes", 50)
    _png(
        tmp_path / "cityscapes_c" / "fog" / "2" / "aachen" / "a_000_leftImg8bit.png",
        np.full((4, 6, 3), 90, dtype=np.uint8),
    )
    clean = C.Cityscapes(str(tmp_path / "cityscapes"))
    corrupted = C.CityscapesC(str(tmp_path / "cityscapes_c"), "fog", 2, str(tmp_path / "cityscapes"))
    assert len(clean) == len(corrupted) == 1
    a, b = clean[0], corrupted[0]
    assert a["id"] == b["id"] == "aachen/a_000"
    assert torch.equal(a["label"], b["label"]) and a["label"].dtype == torch.int64
    assert a["label"][0].tolist() == [0, 1, 2, 255, 7, 18]
    assert torch.allclose(b["image"], torch.full((3, 4, 6), 90 / 255))


def test_cityscapes_refuses_an_image_without_a_label(tmp_path):
    _png(
        tmp_path / "leftImg8bit" / "val" / "aachen" / "b_000_leftImg8bit.png",
        np.zeros((4, 6, 3), dtype=np.uint8),
    )
    with pytest.raises(FileNotFoundError, match="no label"):
        C.Cityscapes(str(tmp_path))


def test_transform_runs_last(tmp_path):
    _cityscapes_tree(tmp_path, 50)

    def tag(item):
        return {**item, "tagged": True}

    assert C.Cityscapes(str(tmp_path), transform=tag)[0]["tagged"] is True


# --- on the fly -----------------------------------------------------------------------


def unit_batch(n=3, h=5, w=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 256, (n, 3, h, w), generator=g).float() / 255.0)


def test_perturbation_round_trips_8_bit_images_and_passes_the_label():
    seen = []

    def identity(image, name, severity, seed):
        assert image.dtype == np.uint8 and image.shape == (5, 7, 3)
        seen.append(seed)
        return image

    images, label = unit_batch(), torch.zeros(3, 5, 7, dtype=torch.long)
    perturb = C.corruption_perturbation("fog", 2, fn=identity)
    out, label_out = perturb(images, label, 0)
    assert torch.allclose(out, images, atol=1e-6)
    assert label_out is label
    assert len(set(seen)) == 3


def test_perturbation_seeds_by_image_position_not_batch():
    seeds = []

    def record(image, name, severity, seed):
        seeds.append(seed)
        return image

    perturb = C.corruption_perturbation("fog", 2, seed=7, fn=record)
    images = unit_batch(n=4)
    perturb(images, None, 0)
    whole = list(seeds)
    seeds.clear()
    perturb(images[:1], None, 0)
    perturb(images[1:], None, 1)
    assert seeds == whole
    seeds.clear()
    C.corruption_perturbation("fog", 3, seed=7, fn=record)(images, None, 0)
    assert seeds != whole


def test_image_corruption_is_the_perturbation_one_image_at_a_time():
    # What a loader's workers apply per item must be what the unit batch gets.
    def shifted(image, name, severity, seed):
        return ((image.astype(int) + seed) % 256).astype(np.uint8)

    images = unit_batch(n=4)
    batch, _ = C.corruption_perturbation("fog", 2, seed=7, fn=shifted)(images[1:], None, 1)
    one = C.ImageCorruption("fog", 2, seed=7, fn=shifted)
    for j in range(3):
        image = (images[1 + j] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        expected = torch.from_numpy(one(image, 1 + j)).permute(2, 0, 1).float() / 255
        assert torch.equal(batch[j], expected)
    with pytest.raises(KeyError):
        C.ImageCorruption("fogg", 1)


def test_perturbation_refuses_a_resizing_corruption():
    def crop(image, name, severity, seed):
        return image[1:]

    with pytest.raises(ValueError, match="keep the image's size"):
        C.corruption_perturbation("fog", 1, fn=crop)(unit_batch(), None, 0)


def test_unknown_names_and_severities_are_refused():
    with pytest.raises(KeyError):
        C.corruption_perturbation("fogg", 1)
    with pytest.raises(ValueError):
        C.corruption_perturbation("fog", 6)
    with pytest.raises(KeyError):
        C.corrupt(np.zeros((4, 4, 3), dtype=np.uint8), "fogg", 1, 0)
    with pytest.raises(KeyError):
        C.check_backend("fogg")


def test_a_missing_backend_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "imagecorruptions", None)
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError, match="pip install imagecorruptions"):
        C.check_backend("fog")
    with pytest.raises(ImportError, match="pip install opencv-python"):
        C.check_backend(C.COMBINATION)


def test_imagecorruptions_backend_is_seeded_and_leaves_numpy_alone():
    pytest.importorskip("imagecorruptions")
    image = (np.random.default_rng(0).random((48, 64, 3)) * 255).astype(np.uint8)
    np.random.seed(5)
    state = np.random.get_state()
    for name in ("gaussian_noise", "fog", "pixelate", "jpeg_compression"):
        a = C.corrupt(image, name, 3, seed=11)
        b = C.corrupt(image, name, 3, seed=11)
        assert a.shape == image.shape and a.dtype == np.uint8, name
        assert np.array_equal(a, b), name
        assert not np.array_equal(a, image), name
    assert np.array_equal(np.random.get_state()[1], state[1])


def test_combination_is_seeded_and_clips_instead_of_wrapping():
    pytest.importorskip("cv2")
    image = np.full((16, 16, 3), 250, dtype=np.uint8)
    for level in C.COMBINATION_LEVELS:
        a = C.combination(image, level, seed=1)
        assert a.shape == image.shape and a.dtype == np.uint8
        assert np.array_equal(a, C.combination(image, level, seed=1))
    # Setting 1 multiplies blue by 1.118: 250 would wrap to 23 in Sensaug. Setting 1's
    # other steps move a near-white image only mildly, so no pixel should come out dark.
    assert C.combination(image, 1, seed=1).min() > 100
    with pytest.raises(ValueError):
        C.combination(image, 0, seed=1)
