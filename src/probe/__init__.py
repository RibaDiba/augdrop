"""MEASURE: per-image ``d loss / d magnitude`` for every op -> the matrix G.

G is (A ops x N probe images), measured on one frozen model state at one magnitude
setting. Turning G into R (per-image scale normalization, correlation, bootstrap,
FDR) is analysis, not measurement, and is not in this package.

    probe_set.py        ProbeSet: the fixed, hashed image subset every G is measured on
    grad_collect.py     collect_grads(adapter, batch, ops, magnitudes) -> rows of G
    finite_diff.py      fd_grad: the fallback for ops autograd can't differentiate
    fade.py             the other one: d loss / d a of x + a * (op(x, m) - x)
    frozen_sweep.py     sweep / sweep_checkpoints: checkpoints x probe set -> G
    magnitude_sweep.py  sweep_magnitudes: G at several reference magnitudes (e3)

The probe imports neither vocab/ nor hosts/, and knows no training framework. It
talks to two duck-typed things.

THE OPS are a vocab: an ordered ``{name: OpSpec}`` from ``vocab.build_vocab``, whose
order is G's row order. The probe uses ``op.apply(images01, magnitude, label)`` with a
(B,) magnitude tensor, plus ``mag_range``, ``identity_at``, ``grad_mode``,
``warps_label`` and the optional ``tangent``. ``grad_mode="autograd"`` ops are
differentiated directly; ``"none"`` ops through their exact ``tangent`` when they have
one, by finite differences otherwise; ops with no magnitude get a NaN row. A sweep's
``fade`` overrides the last three for the ops it selects: they are faded in at their
magnitude and differentiated along the fade (fade.py), which needs no magnitude. Every
row that isn't faded is d loss / d strength, signed toward a stronger augmentation
(``grad_collect.strength_sign``), so an image a method negated a signed op on enters R
with the same sign as the others.

A MODEL ADAPTER wraps the model being probed (not an ``AugHost``: a host is a
method's sampling policy, and the probe doesn't need one):

    adapter.model
        The bare ``torch.nn.Module``. The sweep puts it in eval() and refuses to probe
        if any submodule is left in train mode.
    adapter.to_unit(batch) -> (images01, label)
        A CLEAN batch as (B, 3, H, W) float RGB in [0, 1] on the model's device, and
        the dense label map the ops carry along ((B, H, W) or (B, 1, H, W) integer ids,
        or None when there is none, e.g. for classification). It must be
        deterministic: no training-time augmentation and no batch-level mixing
        (Mixup/CutMix couple the images).
    adapter.loss(images01, label, batch) -> (B,) per-image loss, or a scalar
        Gets the perturbed images, the label as the op returned it, and the original
        batch (for class ids, metadata). Normalizes internally; the probe only ever
        hands it [0, 1] images. Image i's loss must depend on image i alone. Prefer
        (B,): finite differences and the coupling check need it at batch size > 1.
    adapter.batch_ids(batch) -> ids                             [optional]
        Lets the sweep check each batch against the probe set's ids.
    adapter.collate                                             [optional]
        The probe loader's ``collate_fn``.
"""

from .fade import FADE, FADE_RULES, NON_DIFFERENTIABLE, check_fade_at, faded_ops
from .finite_diff import DEFAULT_STEP, fd_grad
from .frozen_sweep import Magnitudes, Sweep, strength_to_magnitude, sweep, sweep_checkpoints
from .grad_collect import ModelAdapter, ProbeError, collect_grads, preserved_rng, resolve_methods
from .magnitude_sweep import sweep_magnitudes
from .probe_set import ProbeSet

__all__ = [
    "DEFAULT_STEP",
    "FADE",
    "FADE_RULES",
    "NON_DIFFERENTIABLE",
    "Magnitudes",
    "ModelAdapter",
    "ProbeError",
    "ProbeSet",
    "Sweep",
    "check_fade_at",
    "collect_grads",
    "faded_ops",
    "fd_grad",
    "preserved_rng",
    "resolve_methods",
    "strength_to_magnitude",
    "sweep",
    "sweep_checkpoints",
    "sweep_magnitudes",
]
