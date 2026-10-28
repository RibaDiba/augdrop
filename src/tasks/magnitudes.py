"""The probe's per-op magnitudes for a host: each method decides its own intensity.

- ``sensaug``: the SA loop's last snapshot (``sa.probe_magnitudes``), as Sensaug's probe
  did.
- a host with a magnitude policy of its own (``draw_bins``: randaugment,
  trivialaugment, autoaugment): each op at a magnitude the method itself would apply it
  at. RandAugment's bin M for every image, TrivialAugment's uniform bin per image,
  AutoAugment's bins from its policy, signed ops negated at random, read through
  torchvision's own table (``vocab.magnitude_bins``). The ``*_diff`` hosts too
  (smartsampling_diff: one level per image on 5..30, through smartaug_diff's levels).
- a host whose magnitudes come from a policy network (``madaug``): the policy's own
  level for each probe image (``levels``, ``madaug.policy_levels`` over the probe set in
  probe order, which the caller computes with the live policy), perturbed and signed as
  the method draws them (``MADAugHost.method_magnitudes``). Without levels (no policy:
  augment=false, or a checkpoint that holds none) the configured magnitude.
- a host with a searched policy (``draw_magnitudes``: faster_autoaugment): per image, one
  of the magnitudes the searched policy applies each op at, weighted by how often it
  does, in the op's units (``FasterAutoAugmentHost.draw_magnitudes``).
- otherwise (``uniform``), and always with ``magnitude_mode="fixed"``: the configured
  magnitude.

For a faded op (``probe.fade``) this is the fade's intensity ``m``, not ``a``. Shared by
the mmseg hook, the cls callback and scripts/probe.py, so the three agree.
"""

from typing import Mapping, Optional

import numpy as np

import probe
import sa
from vocab import magnitude_bins

__all__ = ["METHOD_SOURCE", "drawn_magnitudes", "method_magnitudes", "policy_magnitudes", "probe_magnitudes"]

#: ``Magnitudes.source`` of a draw from the host's own policy; its mode is the host.
METHOD_SOURCE = "method"


def method_magnitudes(host, fallback: probe.Magnitudes) -> probe.Magnitudes:
    """``probe.Magnitudes`` drawn from ``host.draw_bins``, per batch, with the sweep's
    generator. ``fallback`` covers any op with a magnitude the host's draw leaves out."""
    num_bins = int(host.num_magnitude_bins)

    def draw(ops, batch_size, rng):
        bins, flips = host.draw_bins(batch_size, rng)
        out, missing = {}, {}
        for name, op in ops.items():
            if op.mag_range is None:
                continue
            if name not in bins:
                missing[name] = op
                continue
            values, signed = magnitude_bins(name, num_bins)
            magnitude = np.asarray(values, dtype=np.float64)[np.asarray(bins[name], dtype=int)]
            if signed:
                # Negated, as the method does: mirrored about the identity. The bins are
                # torchvision's float32, so a mirrored end bin can miss the range's end by
                # round-off (2 - 1.99 is 0.0099999905); that is clipped, and more raises.
                magnitude = np.where(flips[name], 2.0 * op.identity_at - magnitude, magnitude)
                lo, hi = op.mag_range
                slack = 1e-6 * (hi - lo)
                if magnitude.min() < lo - slack or magnitude.max() > hi + slack:
                    raise ValueError(
                        f"op {name!r}: mirrored bins span [{magnitude.min()}, {magnitude.max()}], "
                        f"outside its mag_range [{lo}, {hi}]"
                    )
                magnitude = np.clip(magnitude, lo, hi)
            out[name] = magnitude
        if missing:
            out.update(fallback.draw(missing, batch_size, rng))
        return out

    return probe.Magnitudes(draw=draw, source=METHOD_SOURCE, mode=host.name)


def policy_magnitudes(host, levels, fallback: probe.Magnitudes) -> probe.Magnitudes:
    """``probe.Magnitudes`` from a policy host's per-image ``levels`` ((N, outputs), one
    row per probe image, in probe order): each batch takes the next rows, as the sweep
    draws once per batch in probe order, and ``host.method_magnitudes`` turns them into
    the ops' magnitudes with the sweep's generator. One instance serves one sweep: drawing
    past the last row raises. ``fallback`` covers an op the host leaves out."""
    levels = np.asarray(levels, dtype=np.float64)
    taken = [0]

    def draw(ops, batch_size, rng):
        start = taken[0]
        if start + batch_size > len(levels):
            raise ValueError(
                f"the policy's levels cover {len(levels)} probe images; batch at {start} asks "
                f"for {batch_size} more. Compute them over the sweep's whole probe set"
            )
        taken[0] = start + batch_size
        drawn = host.method_magnitudes(levels[start : start + batch_size], rng)
        out, missing = {}, {}
        for name, op in ops.items():
            if op.mag_range is None:
                continue
            if name in drawn:
                out[name] = drawn[name]
            else:
                missing[name] = op
        if missing:
            out.update(fallback.draw(missing, batch_size, rng))
        return out

    return probe.Magnitudes(draw=draw, source=METHOD_SOURCE, mode=host.name)


def drawn_magnitudes(host, fallback: probe.Magnitudes) -> probe.Magnitudes:
    """``probe.Magnitudes`` drawn from ``host.draw_magnitudes`` (op-unit magnitudes per
    image), per batch, with the sweep's generator. ``fallback`` covers any op with a
    magnitude the host's draw leaves out."""

    def draw(ops, batch_size, rng):
        drawn = host.draw_magnitudes(batch_size, rng)
        out, missing = {}, {}
        for name, op in ops.items():
            if op.mag_range is None:
                continue
            if name in drawn:
                out[name] = np.asarray(drawn[name], dtype=np.float64)
            else:
                missing[name] = op
        if missing:
            out.update(fallback.draw(missing, batch_size, rng))
        return out

    return probe.Magnitudes(draw=draw, source=METHOD_SOURCE, mode=host.name)


def probe_magnitudes(
    host, snapshot: Optional[Mapping], mode: str, fallback: probe.Magnitudes, levels=None
) -> probe.Magnitudes:
    """One sweep's magnitudes for ``host`` (see the module docstring). ``snapshot`` is
    the SA loop's ``corr_magnitudes`` (empty or None without one), ``mode`` the probe's
    ``magnitude_mode``, ``fallback`` its configured magnitude, and ``levels`` a policy
    host's levels over the probe set (None without a policy)."""
    if mode not in sa.MAGNITUDE_MODES:
        raise ValueError(f"unknown magnitude mode {mode!r}, expected one of {sa.MAGNITUDE_MODES}")
    if mode == sa.MODE_FIXED:
        return fallback
    if hasattr(host, "method_magnitudes"):
        return fallback if levels is None else policy_magnitudes(host, levels, fallback)
    if hasattr(host, "draw_magnitudes"):
        return drawn_magnitudes(host, fallback)
    if getattr(host, "num_magnitude_bins", None) is not None:
        return method_magnitudes(host, fallback)
    return sa.probe_magnitudes(snapshot or {}, mode, fallback)
