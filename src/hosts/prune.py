"""The static prune: which ops a run takes out of training, by name or at random.

    resolve_prune(vocabulary, ops=, random=, seed=, exclude=) -> Prune
    draw_random_prune(pool, count, seed)

Ported from Sensaug's ``train.py``: ``--pruned-augmentations`` (the named ops) and the
``null`` random-prune arm (``--random-prune-method=null --random-prune-count N
--random-prune-seed S``: N ops drawn once, before training, and kept out for the whole
run). The draw is Sensaug's ``draw_random_prune``, unchanged. Its pool is also
Sensaug's: the vocabulary minus the named ops and minus ``exclude`` (the ops the run
already keeps out of training, e.g. lighter_H / darker_H under ``--no-inv-aug``), so a
draw of N always removes N ops the run would otherwise train on. One op name is one
unit: ``lighter_R`` and ``darker_R`` (or ``X_pos`` / ``X_neg``) are drawn independently.

What differs from Sensaug: the seed is a config value (``prune.seed``, by default the
run's seed), never derived from ``SLURM_JOB_ID`` or ``os.urandom``. The draw is then a
pure function of the config, so every rank, the ``--dry-run`` preflight, a resume and
the post-hoc scripts all get the same set, and a run can be repeated from its
``dropaug.yaml`` alone.

The prune is applied by the host (``AugHost.prune``); ``build_host(pruned=...)`` does
it at construction. ``Prune.record()`` is the run's ``prune.json``. Pure numpy.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["PRUNE_FILE", "Prune", "draw_random_prune", "resolve_prune"]

#: The record train.py writes to the run directory.
PRUNE_FILE = "prune.json"


def draw_random_prune(pool: Sequence[str], count: int, seed: int) -> Tuple[str, ...]:
    """``count`` op names drawn from ``pool``, sorted. Sensaug's ``draw_random_prune``.

    A private ``RandomState``, never the global numpy RNG, which the training draws
    consume: drawing from it would shift every later augmentation by how many ops were
    pruned. ``pool`` is sorted first, so the result can't depend on its order."""
    rng = np.random.RandomState(seed)
    return tuple(sorted(rng.choice(sorted(pool), size=count, replace=False).tolist()))


@dataclass(frozen=True)
class Prune:
    """One run's static prune.

    named           the ops pruned by name, in vocabulary order
    random          how many were drawn at random
    seed            the draw's seed (None when nothing is drawn)
    pool            the ops the draw could pick from, sorted (empty when nothing is drawn)
    drawn           the ops drawn, sorted
    ops             every pruned op (named and drawn), in vocabulary order
    hold_none_prob  P(no augmentation) is held fixed (``AugHost.prune``)
    probe           the in-training probe still measures the pruned ops
    """

    named: Tuple[str, ...] = ()
    random: int = 0
    seed: Optional[int] = None
    pool: Tuple[str, ...] = ()
    drawn: Tuple[str, ...] = ()
    ops: Tuple[str, ...] = ()
    hold_none_prob: bool = True
    probe: bool = False

    def __bool__(self) -> bool:
        return bool(self.ops)

    def record(self, vocabulary: Sequence[str], **extra) -> dict:
        """``prune.json``: everything needed to repeat the prune, plus ``kept``."""
        pruned = set(self.ops)
        return {
            "named": list(self.named),
            "random": self.random,
            "seed": self.seed,
            "pool": list(self.pool),
            "drawn": list(self.drawn),
            "pruned": list(self.ops),
            "kept": [n for n in vocabulary if n not in pruned],
            "hold_none_prob": self.hold_none_prob,
            "probe": self.probe,
            **extra,
        }


def _count(value) -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"prune.random must be a count >= 0, got {value!r}")
    return int(value)


def resolve_prune(
    vocabulary: Sequence[str],
    *,
    ops: Iterable[str] = (),
    random: int = 0,
    seed: Optional[int] = None,
    exclude: Iterable[str] = (),
    hold_none_prob: bool = True,
    probe: bool = False,
) -> Prune:
    """The prune ``ops`` (by name) and ``random`` (a count) describe on ``vocabulary``.

    ``exclude`` lists ops the run keeps out of training already; a random draw never
    spends a prune on one. Raises ValueError on a name outside the vocabulary, a
    prune of every op, a count with no seed, or a count that would empty the pool."""
    vocabulary = tuple(str(n) for n in vocabulary)
    named_set = {str(n) for n in ops}
    unknown = sorted(named_set - set(vocabulary))
    if unknown:
        raise ValueError(
            f"prune.ops {unknown} are not in the host's vocabulary ({len(vocabulary)} ops: "
            f"{list(vocabulary)[:4]}...)"
        )
    named = tuple(n for n in vocabulary if n in named_set)
    if len(named) == len(vocabulary):
        raise ValueError("prune.ops names every op of the vocabulary; nothing would be left to train on")

    count = _count(random)
    pool: Tuple[str, ...] = ()
    drawn: Tuple[str, ...] = ()
    if count:
        if seed is None or isinstance(seed, bool) or int(seed) != seed or not 0 <= seed < 2**32:
            raise ValueError(f"prune.seed must be an integer in [0, 2**32) to draw prune.random ops, got {seed!r}")
        seed = int(seed)
        skip = named_set | {str(n) for n in exclude}
        pool = tuple(sorted(n for n in vocabulary if n not in skip))
        if count >= len(pool):
            raise ValueError(
                f"prune.random={count} would leave no op to train on: only {len(pool)} are "
                f"eligible (the vocabulary minus prune.ops {list(named)} and the ops the run "
                f"already excludes {sorted(set(exclude) & set(vocabulary))}): {list(pool)}"
            )
        drawn = draw_random_prune(pool, count, seed)
    else:
        seed = None

    pruned = set(named) | set(drawn)
    return Prune(
        named=named,
        random=count,
        seed=seed,
        pool=pool,
        drawn=drawn,
        ops=tuple(n for n in vocabulary if n in pruned),
        hold_none_prob=bool(hold_none_prob),
        probe=bool(probe),
    )
