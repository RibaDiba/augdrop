"""Hard: greedy non-maximum suppression over R, with no statistical gate. The NMS arm.

Ops are visited in order of how much they raise the loss, ``mean_i Delta_ai`` (the
probe's ``Sweep.loss_matrix()``, carried as ``stats.Analysis.loss_increase``), highest
first. An op joins the kept set K unless its R with an op already in K exceeds ``tau``.
If it does, it is suppressed and pruned. This is NMS as object detection runs it, with
the loss increase as the box score and R as the overlap. A cluster of three or more
mutually redundant ops keeps exactly one representative, its most damaging member,
where a rule applied pair by pair can keep or drop the cluster unevenly::

    a -> K   if max_{k in K} R_ak <= tau,   otherwise a -> M
    q(a) ~ p(a) w(a),   w(a) = 1 - lam = 0 for a in M, 1 otherwise

R is ``stats.analyze(residual_rank=True)``'s R_residual, taken with no bootstrap, FDR,
n_min or shared-factor gate. ``tau`` is the one threshold. Its default, 0.5, is Cohen's
(1988) "large" correlation: the pair shares a quarter of its rank variance across
images. At A = 32 it sits above ``stats.null_max_bound`` from 57 probe images on
(FACTS D8), so a permutation threshold could not have raised it at the probe's sizes.

Three rules come from ``base.pairwise(R, "signed")``. The two halves of one op
(``mask_within_op``) never suppress each other. An anti-correlated pair never
suppresses: R is signed, and hurting different images is the opposite of redundant.
And a cell R never measured is no evidence.

The prune is latched, unlike ``mrmr``'s. ``pruned`` (M from the last probe) stays
pruned and leaves the ranking, so M only grows over a run, and an op kept at one probe
can still be suppressed at a later one. ``prune_weights`` re-applies M to a new prior,
which is what an SA round does with each fresh pdf.

There is no ``budget``. How many ops go is whatever R says, possibly none.

**The matched null** (``null_arm="random"``, ``random_null``): at each probe, NMS runs as
above, and then as many ops as it newly pruned are drawn at random from the ops it ranked
(a positive prior, a usable row in R, a loss increase, not yet pruned) and pruned instead.
That is the random-K arm of an online prune: the same schedule, the same count at each
probe, the same pool (unprobed and exempt ops are never drawn), only which ops go is
random. Each probe's draw is seeded by ``(null_seed, iteration)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .base import Redundancy, Selection, Selector, as_relevance, declined, pairwise

__all__ = ["NMS", "NULLS", "NmsResult", "TAU", "nms_analysis", "nms_select", "prune_weights", "random_null"]

#: Cohen's (1988) "large" correlation: r^2 = 1/4.
TAU = 0.5

#: The nulls an NMS arm can run as (``NMS(null_arm=...)``, ``random_null``).
NULLS = ("random",)


class NmsResult(NamedTuple):
    """One NMS pass, for the log.

    order          the ranked ops, highest loss increase first
    kept           K, in that order, then the exempt ops
    suppressed     ``{op: (the kept op that suppressed it, their R)}``, this pass only
    exempt         ops with a positive prior but no usable row in R or no score: kept
    pruned_before  M as it came in, whatever the prior now gives its ops
    pruned         M after this pass: ``pruned_before`` plus ``suppressed``, in the
                   order the ops were pruned. Never smaller than ``pruned_before``
    score          ``{op: mean loss increase}`` of the ranked ops
    tau            the cutoff
    """

    order: Tuple[str, ...]
    kept: Tuple[str, ...]
    suppressed: Dict[str, Tuple[str, float]]
    exempt: Tuple[str, ...]
    pruned_before: Tuple[str, ...]
    pruned: Tuple[str, ...]
    score: Dict[str, float]
    tau: float

    @property
    def new(self) -> Tuple[str, ...]:
        """The ops this pass added to M."""
        return tuple(n for n in self.pruned if n not in self.pruned_before)

    def record(self) -> dict:
        """The fields of this pass's corr_redundancy_log.txt record."""
        return {
            "tau": self.tau,
            "order": list(self.order),
            "score": dict(self.score),
            "kept": list(self.kept),
            "suppressed": {op: {"by": by, "r": r} for op, (by, r) in self.suppressed.items()},
            "exempt": list(self.exempt),
            "pruned_before": list(self.pruned_before),
            "pruned_new": list(self.new),
            "pruned": list(self.pruned),
        }


class NMS(Selector):
    """Greedy NMS over the ops with a positive prior.

    tau        an op is suppressed when its R with a kept op is above this
    lam        a suppressed op's weight is ``1 - lam``. It must be 1: this arm prunes. It
               is a parameter only so the run name and the launcher's ``--lam`` mean the
               same thing for every selector.
    null_arm   None, or ``"random"``: the matched null (module docstring), which
               PruneHook applies through ``random_null``
    null_seed  the null's seed
    """

    name = "nms"
    hard = True

    def __init__(self, *, tau: float = TAU, lam: float = 1.0, null_arm: Optional[str] = None, null_seed: int = 0):
        tau = float(tau)
        if not (math.isfinite(tau) and -1.0 < tau <= 1.0):
            raise ValueError(f"tau is a correlation cutoff in (-1, 1], got {tau}")
        if float(lam) != 1.0:
            raise ValueError(
                f"nms prunes: a suppressed op's weight is 1 - lam, so lam must be 1, got {lam}"
            )
        if null_arm is not None and null_arm not in NULLS:
            raise ValueError(f"nms null_arm {null_arm!r} is not one of {NULLS} (or None)")
        if int(null_seed) < 0:
            raise ValueError(f"null_seed must be >= 0, got {null_seed}")
        self.tau = tau
        self.lam = 1.0
        self.null = null_arm
        self.null_seed = int(null_seed)

    def select(
        self,
        R: Redundancy,
        rel: Mapping,
        budget: Optional[int] = None,
        *,
        score: Optional[Mapping] = None,
        pruned: Sequence[str] = (),
    ) -> Selection:
        """``run``'s Selection. ``score`` is required; see ``run``."""
        return self.run(R, rel, budget, score=score, pruned=pruned)[0]

    def run(
        self,
        R: Redundancy,
        rel: Mapping,
        budget: Optional[int] = None,
        *,
        score: Optional[Mapping] = None,
        pruned: Sequence[str] = (),
    ) -> Tuple[Selection, NmsResult]:
        """``(Selection, NmsResult)``. ``score`` is ``{op: mean loss increase}``
        (``dict(zip(analysis.names, analysis.loss_increase))``), and ``pruned`` is M
        from the last probe. The Selection declines when M stays empty."""
        if score is None:
            raise TypeError(
                "nms ranks ops by their loss increase: pass score={op: mean Delta}, e.g. "
                "dict(zip(analysis.names, analysis.loss_increase)). It runs through "
                "PruneHook, not ApplyWeightsHook or selection.decide."
            )
        if budget is not None:
            raise ValueError(f"nms has no budget (tau decides how many ops go), got budget={budget}")
        names, prior = as_relevance(rel)
        prior_of = dict(zip(names, prior))
        work = pairwise(R, "signed")
        index = {name: i for i, name in enumerate(R.names)}
        latched = set(pruned)

        def ranks(name: str) -> bool:
            value = score.get(name)
            return (
                name in index
                and np.isfinite(work[index[name]]).any()
                and value is not None
                and math.isfinite(float(value))
            )

        pool = sorted(n for n in names if prior_of[n] > 0 and n not in latched)
        ranked = sorted((n for n in pool if ranks(n)), key=lambda n: (-float(score[n]), n))
        exempt = [n for n in pool if not ranks(n)]

        kept, suppressed = [], {}
        for name in ranked:
            # The first kept op wins a tie: kept is in rank order.
            best = None
            for k in kept:
                r = work[index[name], index[k]]
                if np.isfinite(r) and (best is None or r > best[1]):
                    best = (k, float(r))
            if best is not None and best[1] > self.tau:
                suppressed[name] = best
            else:
                kept.append(name)

        # M keeps an op whose prior is 0 for now (an SA pdf can leave one out), so it
        # never shrinks.
        before = tuple(dict.fromkeys(str(n) for n in pruned))
        after = before + tuple(n for n in ranked if n in suppressed)
        result = NmsResult(
            order=tuple(ranked),
            kept=tuple(kept) + tuple(exempt),
            suppressed=suppressed,
            exempt=tuple(exempt),
            pruned_before=before,
            pruned=after,
            score={n: float(score[n]) for n in ranked},
            tau=self.tau,
        )
        if not after:
            reason = (
                f"no ranked op has R > tau={self.tau:g} with a kept op, so nothing is pruned"
                if ranked
                else "no op in rel has both a usable row in R and a loss increase; there is nothing to rank"
            )
            return declined(names, prior, reason), result
        selection = prune_weights(rel, after)
        if selection.applied:
            selection = selection._replace(kept=result.kept, exempt=result.exempt)
        return selection, result


def prune_weights(rel: Mapping, pruned: Sequence[str]) -> Selection:
    """The prior with M zeroed, ``q(a) ~ p(a)`` off M and 0 on it: the q an NMS run
    installs, and re-installs on every SA round's fresh pdf.

    Declines when no op with a positive prior is in M (q would be the prior), and when
    M holds every one of them (nothing would be left to train on)."""
    names, prior = as_relevance(rel)
    latched = set(pruned)
    hit = tuple(n for n, p in zip(names, prior) if p > 0 and n in latched)
    if not hit:
        return declined(names, prior, "no op the prior keeps is pruned")
    q = np.where([n in latched for n in names], 0.0, prior)
    if q.sum() <= 0:
        return declined(
            names, prior, f"every op the prior keeps is pruned ({list(hit)}); refusing to train on none"
        )
    kept = tuple(n for n, v in zip(names, q) if v > 0)
    return Selection(names, q / q.sum(), True, None, kept=kept, pruned=hit)


def random_null(result: NmsResult, rel: Mapping, *, seed: int, key: int) -> Tuple[Selection, NmsResult, Tuple[str, ...]]:
    """The matched null of one NMS pass (module docstring): ``len(result.new)`` ops drawn
    from ``result.order`` (the ops NMS ranked: a positive prior, a usable row, a score,
    not yet pruned) with ``hosts.prune.draw_random_prune``, seeded by ``(seed, key)``
    (``key`` the probe's iteration, so each probe draws afresh). Returns ``(Selection,
    NmsResult, drawn)``: the result's ``pruned`` is ``pruned_before`` plus the draw and
    its ``suppressed`` is empty (no op suppressed another); ``result`` itself is NMS's
    own pass, for the log."""
    from hosts.prune import draw_random_prune

    count = len(result.new)
    draw_seed = int(np.random.SeedSequence([int(seed), int(key)]).generate_state(1)[0])
    drawn = draw_random_prune(result.order, count, draw_seed) if count else ()
    pruned = result.pruned_before + tuple(drawn)
    kept = tuple(n for n in result.order if n not in drawn) + result.exempt
    null = result._replace(kept=kept, suppressed={}, pruned=pruned)
    names, prior = as_relevance(rel)
    if not pruned:
        return declined(names, prior, "NMS pruned nothing, so the null draws nothing"), null, drawn
    selection = prune_weights(rel, pruned)
    if selection.applied:
        selection = selection._replace(kept=null.kept, exempt=null.exempt)
    return selection, null, drawn


def nms_analysis(sweep, static_pruned: Sequence[str] = ()):
    """The R this arm reads, from one sweep: ``stats.analyze_sweep`` with
    ``residual_rank=True`` and no bootstrap or n_min. ``static_pruned`` (the run's static
    prune) is set aside even when the probe measured it. PruneHook and
    ``scripts/nms_prune.py`` both take it from here, so the online and post-hoc prunes are
    the same computation."""
    import stats

    return stats.analyze_sweep(sweep, static_pruned=tuple(static_pruned), residual_rank=True, bootstrap=False, n_min=0)


def nms_select(
    analysis,
    rel: Mapping,
    *,
    tau: float = TAU,
    pruned: Sequence[str] = (),
    mask_within_op: bool = True,
) -> Tuple[Selection, NmsResult]:
    """``NMS(tau).run`` on ``nms_analysis``'s result: its primary R (R_residual), ranked
    by its loss increase. ``rel`` is the host's prior and ``pruned`` M so far."""
    return NMS(tau=tau).run(
        Redundancy(analysis.names, analysis.primary, mask_within_op=mask_within_op),
        rel,
        score=dict(zip(analysis.names, analysis.loss_increase)),
        pruned=pruned,
    )
