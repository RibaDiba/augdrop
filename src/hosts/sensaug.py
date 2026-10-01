"""SensAug as a host: its round schedule decides when to probe, and q is applied to
its (op, level) pdf.

What is ported from Sensaug, and what is not:

- **The round schedule** (``sensaug/round_schedule.py``) is ported, in
  compressed form: one control probe in the last warmup round, then one every
  ``corr_cadence`` rounds, never on the final round. That module's docstrings
  explain why each emission lands one hook point before the SA round that reads
  it.
- **The pdf** is in Sensaug's format: ``(op, level) -> prob`` plus a
  ``("none", 0)`` entry. ``set_weights(q)`` moves perturbation mass BETWEEN ops
  and leaves both each op's level shape and P(none) exactly as the SA pdf had
  them. This is what ``redundancy.reweight`` does for Lever 3.
- **The SA loop** evaluates mIoU per (op, level) every round and generates the pdf
  from it (``sa`` for the math, ``tasks/mmseg/loops.RobustValLoop`` for mmseg). It
  hands each round's pdf to ``set_sa_pdf()``. Its ``warmup_rounds`` and round
  interval must equal this host's, which the loop checks.

Before the first SA pdf the policy is Sensaug's warmup policy: uniform over the
ops plus "none", with the magnitude uniform on [0, 1]. ``pdf()`` returns None then,
and ``set_weights`` raises, because Sensaug never reweights the warmup policy.
The one probe that fires during warmup is the control, whose R must not reach
the pdf anyway.
"""

import math
import warnings
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .base import AugHost

__all__ = [
    "SensAugHost",
    "DIFF32_OPS",
    "NONE_KEY",
    "SA_CURVE_CADENCE",
    "derive_corr_rounds",
    "control_rounds",
    "round_fire_iters",
]

#: Sensaug's 32-op vocabulary (``DIFF32_OPS``), in its registry order: the 14 base
#: ops, then the 18 AutoAugment-family ops. Only the names live here. The
#: implementations are what ``vocab/`` ports.
DIFF32_OPS = (
    "lighter_R", "darker_R", "lighter_G", "darker_G", "lighter_B", "darker_B",
    "lighter_H", "darker_H", "lighter_S", "darker_S", "lighter_V", "darker_V",
    "blur", "noise",
    "rotate_pos", "rotate_neg", "shear_x_pos", "shear_x_neg",
    "shear_y_pos", "shear_y_neg", "translate_x_pos", "translate_x_neg",
    "translate_y_pos", "translate_y_neg",
    "brightness_pos", "brightness_neg", "contrast_pos", "contrast_neg",
    "sharpness_pos", "sharpness_neg", "color_pos", "color_neg",
)  # fmt: skip

#: The pdf's no-op entry. Held fixed by set_weights.
NONE_KEY = ("none", 0)

#: Rounds between SA-curve recomputes (``RobustValLoop.run``'s ``% 6``).
SA_CURVE_CADENCE = 6

#: How far a pdf's total may sit from 1 before set_sa_pdf refuses it.
_PDF_SUM_TOL = 1e-6

Pdf = Dict[Tuple[str, float], float]


# --- the round schedule (ported from sensaug/round_schedule.py) ---------------


def derive_corr_rounds(
    n_rounds: int, warmup_rounds: int, cadence: int = SA_CURVE_CADENCE
) -> list:
    """The default firing rounds: ``warmup_rounds - 1`` (the control probe), then
    every ``cadence`` rounds from ``warmup_rounds``, excluding the final round.
    Nothing is left to read the final round's R once training ends.

    At 20 rounds / 4 warmup: cadence 6 gives [3, 4, 10, 16], cadence 3 gives
    [3, 4, 7, 10, 13, 16]. With no warmup there is no control probe.
    """
    if cadence < 1:
        raise ValueError(f"cadence must be a positive round count, got {cadence}")
    rounds = []
    if warmup_rounds > 0:
        rounds.append(warmup_rounds - 1)
    rounds.extend(range(warmup_rounds, max(n_rounds - 1, 0), cadence))
    return [r for r in rounds if 0 <= r < n_rounds - 1]


def control_rounds(rounds: Iterable[int], warmup_rounds: int) -> list:
    """The firing rounds inside warmup: measured before any pdf existed, so their R
    is a baseline and must not be published."""
    return [r for r in rounds if r < warmup_rounds]


def round_fire_iters(
    rounds: Iterable[int], round_interval: int, pre_train_round: bool = False
) -> Tuple[int, ...]:
    """Round r's val loop runs at iteration count ``(r + 1) * round_interval``, so
    that is the hook point just before it. ``pre_train_round`` covers a val round
    run before training starts (Sensaug's ``init_sa``, on --resume/--no-warmup),
    which shifts every round one interval earlier and drops any that land at 0.
    """
    offset = 1 if pre_train_round else 0
    return tuple(
        (r + 1 - offset) * round_interval for r in rounds if r + 1 - offset >= 1
    )


# --- the host -----------------------------------------------------------------


class SensAugHost(AugHost):
    """``--aug-type=grad_corr``'s policy.

    The round grid is the run's EFFECTIVE one: ``max_iters // round_interval``
    rounds, where ``round_interval`` defaults to ``max_iters // n_rounds``. The
    defaults match the ``configs/rounds.yaml`` Sensaug ships (20 rounds, 4
    warmup, cadence 3). ``corr_rounds`` / ``control_rounds`` pin an explicit
    schedule instead. ``probe_interval`` drops the round grid for a fixed clock
    with no control probe (Sensaug's ``--corr-interval``).
    """

    name = "sensaug"

    def __init__(
        self,
        *,
        max_iters: int,
        round_interval: Optional[int] = None,
        n_rounds: int = 20,
        warmup_rounds: int = 4,
        corr_cadence: int = 3,
        corr_rounds: Optional[Sequence[int]] = None,
        control_rounds: Optional[Sequence[int]] = None,
        pre_train_round: bool = False,
        probe_interval: Optional[int] = None,
        vocabulary: Sequence[str] = DIFF32_OPS,
    ):
        if round_interval is None:
            round_interval = max_iters // n_rounds
        if round_interval < 1:
            raise ValueError(
                f"round_interval must be >= 1, got {round_interval} "
                f"(max_iters={max_iters}, n_rounds={n_rounds})"
            )
        self.round_interval = int(round_interval)
        self.n_rounds = max_iters // self.round_interval
        self.warmup_rounds = int(warmup_rounds)

        if probe_interval is not None:
            self.rounds: Tuple[int, ...] = ()
            probe_iters, control_iters = None, ()
        else:
            self.rounds, probe_iters, control_iters = self._resolve_rounds(
                corr_rounds, control_rounds, corr_cadence, pre_train_round
            )

        self._sa_pdf: Optional[Pdf] = None
        self._tilt: Optional[np.ndarray] = None

        n = len(vocabulary)
        super().__init__(
            vocabulary,
            np.full(n, 1.0 / n) if n else np.zeros(0),
            none_prob=1.0 / (n + 1),
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )

    def _resolve_rounds(self, corr_rounds, controls, cadence, pre_train_round):
        """Sensaug's ``resolve_schedule``, minus ``full_recheck_iters`` (which only
        ``--corr-skip-pruned-eval`` read)."""
        n, warmup = self.n_rounds, self.warmup_rounds
        if n < 1:
            raise ValueError(f"the run has no val round: n_rounds={n} (max_iters // round_interval)")
        if not 0 <= warmup < n:
            raise ValueError(
                f"warmup_rounds={warmup} leaves no post-warmup round of the {n} this run has"
            )

        rounds = (
            derive_corr_rounds(n, warmup, cadence)
            if corr_rounds is None
            else sorted(set(corr_rounds))
        )
        controls = (
            control_rounds(rounds, warmup) if controls is None else sorted(set(controls))
        )
        if not rounds:
            raise ValueError(
                f"the schedule fires nowhere: {n} round(s), {warmup} of them warmup, is "
                f"too few for cadence {cadence}. Lower round_interval, or pass probe_interval."
            )
        for key, values in (("corr_rounds", rounds), ("control_rounds", controls)):
            bad = [r for r in values if not 0 <= r < n]
            if bad:
                raise ValueError(f"{key} {bad} are outside this run's rounds 0..{n - 1}")
        missing = sorted(set(controls) - set(rounds))
        if missing:
            raise ValueError(f"control_rounds {missing} are not in corr_rounds {rounds}")
        if n - 1 in rounds:
            warnings.warn(
                f"round {n - 1} is the final round; the R it emits is measured after the "
                f"last pdf is built and nothing can read it",
                stacklevel=3,
            )

        probe_iters = round_fire_iters(rounds, self.round_interval, pre_train_round)
        if not probe_iters:
            # Every round fell at or before iteration 0 under pre_train_round.
            raise ValueError(f"no firing round of {rounds} lands after iteration 0")
        control_iters = round_fire_iters(controls, self.round_interval, pre_train_round)
        return tuple(rounds), probe_iters, control_iters

    # --- the pdf ---------------------------------------------------------------

    def set_sa_pdf(self, pdf: Pdf) -> None:
        """Install this round's SA pdf, as Sensaug's pdf generators produce it.

        Ops absent from the pdf (e.g. ``lighter_H``/``darker_H`` under
        ``--no-inv-aug``) get base probability 0. A q already set stays in effect.
        It is kept as a ratio to the base it was set against and re-applied to
        the new base, the same way Sensaug re-applies the currently published
        ``red(a)`` to every pdf it generates until a new R replaces it.
        """
        index = {n: i for i, n in enumerate(self._names)}
        mass = np.zeros(len(self._names))
        none = 0.0
        for key, value in pdf.items():
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"pdf[{key!r}] = {value}")
            if tuple(key) == NONE_KEY:
                none = value
                continue
            op = key[0]
            if op not in index:
                raise KeyError(f"pdf op {op!r} is not in this host's vocabulary")
            mass[index[op]] += value
        total = none + float(mass.sum())
        if abs(total - 1.0) > _PDF_SUM_TOL:
            raise ValueError(f"pdf sums to {total}, not 1")
        if mass.sum() <= 0:
            raise ValueError("pdf puts all its mass on ('none', 0); there is nothing to sample")

        base = mass / mass.sum()
        if self._tilt is not None:
            probs = base * self._tilt
            if probs.sum() <= 0:
                raise ValueError(
                    "the q in effect drops every op this SA pdf keeps; call set_weights "
                    "with a q derived from the new base_probs() first"
                )
            probs = probs / probs.sum()
        else:
            probs = base

        self._sa_pdf = dict(pdf)
        self._none_prob = none
        self._base = base
        self._probs = probs

    def pdf(self) -> Optional[Pdf]:
        """The pdf training samples from, with q applied. None before the first SA pdf.

        With no q set it is the SA pdf itself, bit-identical rather than
        renormalized. Sensaug's control arm has to reproduce the baseline exactly,
        so this path does no arithmetic.
        """
        if self._sa_pdf is None:
            return None
        if self._tilt is None:
            return dict(self._sa_pdf)
        index = {n: i for i, n in enumerate(self._names)}
        scale = np.divide(
            self._probs, self._base, out=np.zeros_like(self._probs), where=self._base > 0
        )
        return {
            key: (value if tuple(key) == NONE_KEY else float(value) * float(scale[index[key[0]]]))
            for key, value in self._sa_pdf.items()
        }

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        if self._sa_pdf is None:
            raise RuntimeError(
                "SensAug reweights the SA pdf, and there is none yet (warmup). The "
                "only probe that fires in warmup is the control, whose R must not "
                "reach the pdf."
            )
        resurrected = [n for n, b, v in zip(self._names, self._base, q) if b == 0 and v > 0]
        if resurrected:
            raise ValueError(
                f"q gives weight to {resurrected}, which the SA pdf excludes; set "
                f"them to 0 or derive q from base_probs()"
            )
        self._tilt = np.divide(q, self._base, out=np.ones_like(q), where=self._base > 0)
        return q
