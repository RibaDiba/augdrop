"""G at several reference magnitudes on one frozen model state (experiment e3).

A probe magnitude is a choice, not a property of the data. R taken at one magnitude
says nothing about R at another, and Sensaug only ever measured one setting per run.
This sweeps the same model and the same probe set once per reference and returns one
G for each, never pooled, so that how R moves with magnitude is measured rather than
assumed.

A reference is either

- a float STRENGTH in [0, 1], applied to every op through its own range
  (``frozen_sweep.strength_to_magnitude``: 0 is the identity, 1 the far end of
  ``mag_range``). One strength works across families with different units, or
- a mapping ``{op name: magnitude}`` in each op's own units, e.g. a KID-calibrated
  set.

Every G is labelled with its reference in ``meta["reference"]``.
"""

from typing import Any, Mapping, Sequence, Union

from .frozen_sweep import Magnitudes, Sweep, strength_to_magnitude, sweep
from .grad_collect import UNPROBED, ModelAdapter, op_method
from .probe_set import ProbeSet

Reference = Union[float, Mapping[str, float]]


def sweep_magnitudes(
    adapter: ModelAdapter,
    probe_set: ProbeSet,
    dataset,
    ops: Mapping[str, Any],
    references: Sequence[Reference],
    **sweep_kwargs,
) -> list[Sweep]:
    """One ``sweep`` per reference, in order, all on the adapter's current weights.

    Every reference is checked against every op before anything runs, so a bad third
    reference fails in seconds instead of after two full sweeps. ``sweep_kwargs`` go to
    ``frozen_sweep.sweep``. A ``meta`` passed there is merged into each G's labels.
    """
    references = list(references)
    if not references:
        raise ValueError("no reference magnitudes to sweep")
    for reference in references:
        _check_reference(reference, ops)

    base_meta = dict(sweep_kwargs.pop("meta", None) or {})
    sweeps = []
    for reference in references:
        if isinstance(reference, Mapping):
            label = {"reference": {n: float(v) for n, v in reference.items()}, "reference_kind": "absolute"}
            magnitudes = Magnitudes.fixed(reference, source="reference")
        else:
            label = {"reference": float(reference), "reference_kind": "strength"}
            magnitudes = Magnitudes.relative(float(reference), source="reference")
        sweeps.append(
            sweep(adapter, probe_set, dataset, ops, magnitudes, meta={**base_meta, **label}, **sweep_kwargs)
        )
    return sweeps


def _check_reference(reference: Reference, ops: Mapping[str, Any]) -> None:
    for name, op in ops.items():
        if op_method(op) == UNPROBED:
            continue
        if not isinstance(reference, Mapping):
            strength_to_magnitude(op, float(reference))  # raises outside [0, 1]
            continue
        if name not in reference:
            raise ValueError(f"reference {dict(reference)} has no magnitude for op {name!r}")
        lo, hi = op.mag_range
        if not lo <= float(reference[name]) <= hi:
            raise ValueError(
                f"reference magnitude {reference[name]} is outside op {name!r}'s "
                f"mag_range [{lo}, {hi}]"
            )
