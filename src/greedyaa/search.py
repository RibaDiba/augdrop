"""Greedy AutoAugment's search loop (Algorithm 1), framework-free.

The loop over a fixed list of candidate steps (``hosts.greedy_autoaugment.candidates``, the
``(search op, level)`` pairs the prune leaves):

- level 1 tries every candidate alone;
- level c >= 2 takes a base, the best trial so far whose chain has not been a base, and
  tries it with every candidate appended (``Search/main.py``'s ``get_max_idx``: the base
  can come from any earlier level, so a chain need not grow every level);
- after ``levels`` levels the policy is the ``top`` best trials over all of them
  (``saved_policies``, ranked; the released ``found_policies`` hold the top 25).

A trial is scored by whoever runs it (``scripts/gaa_search.py``: a child network trained on
the chain, its mIoU on held-out training images); this module only decides what to try
next. Ties go to the earlier trial (level, then candidate order); the code's
``np.argsort(...)[::-1]`` leaves them to the sort. ``GreedySearch`` is rebuilt from the
records of the trials already scored, so a stopped search resumes where it was.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Step = Tuple[str, int]
Chain = Tuple[Step, ...]

__all__ = ["GreedySearch", "Trial", "trial_counts"]


@dataclass(frozen=True)
class Trial:
    """One candidate: its greedy ``level`` (1-based), its ``index`` among the level's
    candidates, and the chain it trains on."""

    level: int
    index: int
    chain: Chain

    @property
    def key(self) -> str:
        return f"{self.level}-{self.index}"

    def record(self, score: float, **extra) -> dict:
        """The trial's line in ``trials.jsonl``."""
        return {
            "key": self.key,
            "level": self.level,
            "index": self.index,
            "chain": [list(step) for step in self.chain],
            "score": float(score),
            **extra,
        }


def trial_counts(num_candidates: int, levels: int) -> Dict[str, int]:
    """Trials a level and in all: Eq. 1, ``sum_1^k (t_n m_n)``, over the candidates kept."""
    return {"per_level": int(num_candidates), "levels": int(levels), "total": int(num_candidates) * int(levels)}


class GreedySearch:
    """Algorithm 1 over ``candidates`` for ``levels`` levels (module docstring). ``records``
    are the scored trials so far (``Trial.record`` dicts, any order)."""

    def __init__(self, candidates: Sequence[Step], levels: int, records: Iterable[dict] = ()):
        self.candidates: Tuple[Step, ...] = tuple((str(op), int(level)) for op, level in candidates)
        if not self.candidates:
            raise ValueError("no candidate steps to search over (the prune left none)")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("duplicate candidate steps")
        if int(levels) < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.levels = int(levels)
        self._scores: Dict[str, float] = {}
        self._trials: Dict[str, Trial] = {}
        self._bases: Dict[int, Trial] = {}
        for record in sorted(records, key=lambda r: (int(r["level"]), int(r["index"]))):
            self._restore(record)

    # --- state ---------------------------------------------------------------------------

    def _restore(self, record: dict) -> None:
        level, index = int(record["level"]), int(record["index"])
        expected = self.trials_at(level)
        if not 0 <= index < len(expected):
            raise ValueError(f"trial {record['key']}: no candidate {index} at level {level}")
        trial = expected[index]
        chain = tuple((str(op), int(lv)) for op, lv in record["chain"])
        if chain != trial.chain:
            raise ValueError(
                f"trial {trial.key} records chain {list(chain)}, but this search would try "
                f"{list(trial.chain)} there: the records come from another search"
            )
        self.record(trial, float(record["score"]))

    def record(self, trial: Trial, score: float) -> None:
        """Score ``trial``. A non-finite score raises: a child that failed is rerun, not
        ranked."""
        if not math.isfinite(score):
            raise ValueError(f"trial {trial.key}: non-finite score {score}")
        if trial.key in self._scores:
            raise ValueError(f"trial {trial.key} is already scored")
        self._scores[trial.key] = float(score)
        self._trials[trial.key] = trial

    @property
    def level(self) -> int:
        """The level being searched: the first with an unscored trial (``levels + 1`` when
        done)."""
        for level in range(1, self.levels + 1):
            if any(t.key not in self._scores for t in self.trials_at(level)):
                return level
        return self.levels + 1

    @property
    def done(self) -> bool:
        return self.level > self.levels

    def base(self, level: int) -> Optional[Trial]:
        """The chain level ``level`` extends: None at level 1; else the best trial of the
        levels before it whose chain no earlier level used as its base. Needs every trial
        of those levels scored."""
        if level <= 1:
            return None
        if level in self._bases:
            return self._bases[level]
        if level > self.levels:
            raise ValueError(f"level {level} is past the search's {self.levels}")
        before = self.trials_at(level - 1)
        missing = [t.key for t in before if t.key not in self._scores]
        if missing:
            raise ValueError(f"level {level}'s base needs level {level - 1} scored; {len(missing)} trials are not")
        used = {self.base(lv).chain for lv in range(2, level)}
        for trial in self.ranked(upto_level=level - 1):
            if trial.chain not in used:
                self._bases[level] = trial
                return trial
        raise ValueError(f"every trial before level {level} was a base already")

    def trials_at(self, level: int) -> List[Trial]:
        """Level ``level``'s trials, in candidate order."""
        if not 1 <= level <= self.levels:
            raise ValueError(f"level {level} is not in 1..{self.levels}")
        base = self.base(level)
        prefix: Chain = () if base is None else base.chain
        return [Trial(level, i, prefix + (step,)) for i, step in enumerate(self.candidates)]

    def pending(self) -> List[Trial]:
        """The current level's trials not yet scored (empty when the search is done)."""
        if self.done:
            return []
        return [t for t in self.trials_at(self.level) if t.key not in self._scores]

    def score(self, trial: Trial) -> float:
        return self._scores[trial.key]

    # --- the policy --------------------------------------------------------------------------

    def ranked(self, upto_level: Optional[int] = None) -> List[Trial]:
        """The scored trials of levels up to ``upto_level`` (default all), best first, ties
        to the earlier trial."""
        top_level = self.levels if upto_level is None else int(upto_level)
        trials = [t for t in self._trials.values() if t.level <= top_level]
        return sorted(trials, key=lambda t: (-self._scores[t.key], t.level, t.index))

    def policy(self, top: int, upto_level: Optional[int] = None) -> List[dict]:
        """The ``top`` best trials (Algorithm 1, line 24) as the policy file's
        ``sub_policies``, best first."""
        return [
            {"chain": [list(step) for step in t.chain], "score": self._scores[t.key], "trial": t.key}
            for t in self.ranked(upto_level)[: int(top)]
        ]
