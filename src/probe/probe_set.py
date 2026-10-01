"""The fixed, hashed subset of images every G is measured on.

Sensaug swept "the whole val set" and never recorded which images it was. That was
fine while one run produced every R. dropaug compares Gs across checkpoints, magnitudes,
methods and clusters, so each G has to say which images its columns are, and two Gs
have to be checkably on the same ones. A ProbeSet is that record. Its images come in
a canonical order, and its ``digest`` is carried by every G measured on it.

The selection is reproducible from the ids alone. Ids are sorted before the seeded
draw, so the subset depends on WHICH images a dataset holds, not on the order a loader
or a file listing enumerates them in. Use ids that survive moving the data, e.g. paths
relative to the dataset root, not absolute paths on one cluster.

The digest hashes the ids, not the pixels. It pins the subset, not the preprocessing.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
from torch.utils.data import DataLoader, Subset


def _digest(ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


@dataclass(frozen=True)
class ProbeSet:
    """``ids`` in probe order, and ``indices``, the dataset positions they were taken
    from. Build with ``ProbeSet.build``. The constructor trusts its arguments."""

    ids: tuple[str, ...]
    indices: tuple[int, ...]
    source: str = ""
    seed: Optional[int] = None
    digest: str = field(init=False)

    def __post_init__(self):
        if len(self.ids) != len(self.indices):
            raise ValueError(f"{len(self.ids)} ids but {len(self.indices)} indices")
        object.__setattr__(self, "digest", _digest(self.ids))

    @classmethod
    def build(
        cls,
        image_ids: Sequence[Any],
        n: Optional[int] = None,
        *,
        seed: int = 0,
        source: str = "",
    ) -> "ProbeSet":
        """Pick ``n`` of a dataset's images (all of them when ``n`` is None).

        ``image_ids[i]`` is the stable id of dataset item ``i``. ``source`` is a free
        label for the dataset/split, and does not enter the digest.
        """
        ids = [str(i) for i in image_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("image ids must be unique; a duplicate would be probed twice")
        order = sorted(range(len(ids)), key=ids.__getitem__)
        if n is not None:
            if not 0 < n <= len(ids):
                raise ValueError(f"n must be in [1, {len(ids)}], got {n}")
            picked = np.random.default_rng(seed).choice(len(ids), size=n, replace=False)
            order = [order[i] for i in sorted(picked)]
        return cls(
            ids=tuple(ids[i] for i in order),
            indices=tuple(order),
            source=source,
            seed=seed if n is not None else None,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def chunks(self, batch_size: int) -> list[tuple[str, ...]]:
        """The ids of each batch ``loader(dataset, batch_size)`` yields, in order."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        return [self.ids[i : i + batch_size] for i in range(0, len(self.ids), batch_size)]

    def loader(
        self,
        dataset,
        batch_size: int,
        *,
        collate_fn: Optional[Callable] = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """Batches of ``dataset`` restricted to this set, in probe order.

        Never shuffled and never drops the last batch, so batch k holds exactly
        ``chunks(batch_size)[k]``. The sweep relies on that to label its columns.
        """
        if self.indices and max(self.indices) >= len(dataset):
            raise ValueError(
                f"probe set indexes up to {max(self.indices)} but the dataset holds "
                f"{len(dataset)} items. Is this the dataset it was built from?"
            )
        return DataLoader(
            Subset(dataset, list(self.indices)),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def verify(self, image_ids: Sequence[Any]) -> None:
        """Raise unless ``image_ids`` (a dataset's current ids, by position) still hold
        this set's images at this set's indices."""
        current = [str(i) for i in image_ids]
        for index, expected in zip(self.indices, self.ids):
            got = current[index] if index < len(current) else None
            if got != expected:
                raise ValueError(
                    f"dataset item {index} is {got!r}, but probe set {self.digest[:12]} "
                    f"expects {expected!r}. The dataset changed since the set was built."
                )

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "seed": self.seed,
            "digest": self.digest,
            "ids": list(self.ids),
            "indices": list(self.indices),
        }

    @classmethod
    def from_dict(cls, record: dict) -> "ProbeSet":
        probe_set = cls(
            ids=tuple(record["ids"]),
            indices=tuple(int(i) for i in record["indices"]),
            source=record.get("source", ""),
            seed=record.get("seed"),
        )
        if record.get("digest") not in (None, probe_set.digest):
            raise ValueError(
                f"probe set digest mismatch: recorded {record['digest'][:12]}, ids hash "
                f"to {probe_set.digest[:12]}. The file was edited after it was saved."
            )
        return probe_set

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path) -> "ProbeSet":
        with open(path) as f:
            return cls.from_dict(json.load(f))
