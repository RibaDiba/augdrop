import random

import pytest

from probe.probe_set import ProbeSet

IDS = [f"city_{i % 5}/img_{i:04d}.png" for i in range(40)]


def test_full_set_is_every_image_in_canonical_order():
    shuffled = IDS[:]
    random.Random(0).shuffle(shuffled)
    probe_set = ProbeSet.build(shuffled)

    assert list(probe_set.ids) == sorted(IDS)
    assert probe_set.seed is None
    for index, image_id in zip(probe_set.indices, probe_set.ids):
        assert shuffled[index] == image_id


def test_subset_depends_on_which_images_not_their_order():
    """A dataset listed in a different order (another filesystem, another loader)
    must give the same probe set, or two clusters would probe different images."""
    shuffled = IDS[:]
    random.Random(1).shuffle(shuffled)

    a = ProbeSet.build(IDS, n=10, seed=3)
    b = ProbeSet.build(shuffled, n=10, seed=3)

    assert a.ids == b.ids
    assert a.digest == b.digest
    assert len(a) == 10
    assert list(a.ids) == sorted(a.ids)


def test_seed_changes_the_subset():
    assert ProbeSet.build(IDS, n=10, seed=0).ids != ProbeSet.build(IDS, n=10, seed=1).ids


def test_digest_hashes_the_ids_only():
    a = ProbeSet.build(IDS, n=10, seed=0, source="voc/val")
    b = ProbeSet.build(IDS, n=10, seed=0, source="another label")
    c = ProbeSet.build(IDS, n=11, seed=0)
    assert a.digest == b.digest
    assert a.digest != c.digest


def test_rejects_duplicates_and_bad_sizes():
    with pytest.raises(ValueError, match="unique"):
        ProbeSet.build(["a", "b", "a"])
    with pytest.raises(ValueError, match="n must be"):
        ProbeSet.build(IDS, n=0)
    with pytest.raises(ValueError, match="n must be"):
        ProbeSet.build(IDS, n=len(IDS) + 1)


def test_save_load_round_trip(tmp_path):
    probe_set = ProbeSet.build(IDS, n=12, seed=5, source="voc/val")
    probe_set.save(tmp_path / "probe_set.json")
    assert ProbeSet.load(tmp_path / "probe_set.json") == probe_set


def test_an_edited_file_is_rejected():
    record = ProbeSet.build(IDS, n=12, seed=5).to_dict()
    record["ids"][0] = "somewhere/else.png"
    with pytest.raises(ValueError, match="digest mismatch"):
        ProbeSet.from_dict(record)


def test_loader_yields_the_probe_set_in_order_and_chunks_label_it():
    dataset = [{"id": image_id} for image_id in IDS]
    probe_set = ProbeSet.build(IDS, n=7, seed=2)

    batches = list(probe_set.loader(dataset, 3))
    chunks = probe_set.chunks(3)

    assert [len(c) for c in chunks] == [3, 3, 1]  # the last batch is kept
    assert [list(b["id"]) for b in batches] == [list(c) for c in chunks]


def test_verify_catches_a_changed_dataset():
    probe_set = ProbeSet.build(IDS, n=5, seed=0)
    probe_set.verify(IDS)
    changed = IDS[:]
    changed[probe_set.indices[2]] = "replaced.png"
    with pytest.raises(ValueError, match="changed since"):
        probe_set.verify(changed)


def test_loader_rejects_a_smaller_dataset():
    probe_set = ProbeSet.build(IDS)
    with pytest.raises(ValueError, match="dataset holds"):
        probe_set.loader(list(range(10)), 2)
