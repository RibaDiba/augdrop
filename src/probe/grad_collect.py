"""Rows of G for one clean batch: the per-image ``d loss / d magnitude`` of every op.

Ported from Sensaug's ``CollectGradientHook._probe_batch`` and ``_grad_for_op``
(sensaug/hooks/grad_hook.py). Sensaug reached into an mmseg segmentor, its data
preprocessor and ``SegDataSample.gt_sem_seg``, and knew its 32 op names. Here the
model sits behind a ``ModelAdapter`` and the ops are vocab ``OpSpec``s, so the same
probe runs on any model and any vocabulary. Both contracts are in the package
docstring (probe/__init__.py).

Why the probe is shaped this way. Each of these was a real bug in Sensaug:

- The magnitude is a (B,) tensor, one delta PER IMAGE, so one batch yields one
  sensitivity per image per op. A scalar delta gives one number per batch and
  cannot produce R.
- Image i's loss must depend on delta[i] alone. Train-mode BatchNorm breaks that by
  normalizing with batch statistics, so the probe refuses to run on a model with any
  module in train mode. vocab's audit checks each op for the same coupling;
  ``collect_grads(check=True)`` checks the op and the model together, on real data.
- A geometric op moves pixels, so the label must move with it, or its row measures
  image/label misalignment instead of sensitivity. vocab's geometric ops warp the
  label themselves; the probe checks that they did.
- An op's randomness (``noise``) is pinned per op. Every batch then sees the same
  sample, and so do both sides of a finite difference.

An op autograd can't differentiate is measured through its exact ``tangent`` when the
vocab gives one (torchvision's enhance ops), by finite differences otherwise, or, when
``resolve_methods`` selects it, by fading it in (fade.py), which also gives an op with
no magnitude a row.

Every row except a faded one is ``d loss / d strength``: ``d loss / d magnitude`` times
``strength_sign``, the side of ``identity_at`` the image's magnitude is on. A method
that negates a signed op at random (torchvision's) puts some images below the
identity, where a larger magnitude is a weaker augmentation, and their rows would
otherwise enter R with the opposite sign. A fade row is along the augmentation
already. On Sensaug's [0, 1] ops the sign is +1, so their rows are unchanged.

``probe_batch`` also returns each image's loss under each op at its magnitude, and on
the clean image: the loss increase Delta that ``selection.nms`` ranks ops by. Most of
those losses are forwards the rows already ran. The extra cost is one forward per batch
for the clean loss, one per finite-difference op, and one per faded op when ``fade_at``
is not 1. The rows of G are the same with or without them.
"""

import contextlib
import logging
import zlib
from typing import Any, Mapping, NamedTuple, Optional, Protocol, Union

import numpy as np
import torch
from torch import Tensor

from .fade import FADE, Fade, check_fade_at, fade_alpha, fade_images, faded_ops
from .finite_diff import DEFAULT_STEP, fd_grad

logger = logging.getLogger(__name__)

# How each op's row of G is measured. The fifth, FADE, is fade.py's.
AUTOGRAD = "autograd"
TANGENT = "tangent"  # through the vocab op's exact derivative (OpSpec.tangent)
FINITE_DIFF = "finite_diff"
UNPROBED = "unprobed"  # the op has no magnitude and isn't faded; its row is NaN
STATIC_PRUNED = "static_pruned"  # the run prunes the op (AugHost.prune); not measured, NaN row

# to_unit may round-trip through a normalization, so allow float slop around [0, 1].
_UNIT_SLACK = 1e-3
# How far another image's loss may move when only image 0's magnitude does,
# relative to the largest loss in the batch.
_LEAK_RTOL = 1e-5

FdStep = Union[float, Mapping[str, float]]


class ProbeError(RuntimeError):
    """The probe produced something structurally wrong: a bad shape, a non-finite
    gradient, coupled images, an unwarped label.

    Distinct from a transient failure (OOM) so that a sweep lets it propagate
    instead of degrading it into a silently skipped batch.
    """


class ModelAdapter(Protocol):
    """The model side of the probe. See probe/__init__.py for the full contract and
    the optional members (``batch_ids``, ``collate``)."""

    model: torch.nn.Module

    def to_unit(self, batch: Any) -> tuple[Tensor, Optional[Tensor]]: ...

    def loss(self, images01: Tensor, label: Optional[Tensor], batch: Any) -> Tensor: ...


def op_method(op) -> str:
    """How ``op``'s row is measured when it isn't faded: ``"autograd"``, ``"tangent"``
    (vocab ``grad_mode="none"`` with a ``tangent``), ``"finite_diff"`` (the other
    ``grad_mode="none"`` ops), or ``"unprobed"`` (no magnitude to differentiate, or an op
    that ``couples_images``: its value for an image depends on the rest of the batch)."""
    if getattr(op, "mag_range", None) is None or getattr(op, "couples_images", False):
        return UNPROBED
    if op.grad_mode == "autograd":
        return AUTOGRAD
    return TANGENT if getattr(op, "tangent", None) is not None else FINITE_DIFF


def far_end(op) -> float:
    """The end of ``op.mag_range`` farther from ``identity_at``, the direction the op
    strengthens in; ``hi`` on a tie. That matches torchvision's auto-augment bins:
    Solarize and Posterize (identity at the top of the range) strengthen downward, and
    a symmetric signed op like Rotate strengthens in its positive direction."""
    lo, hi = op.mag_range
    anchor = op.identity_at
    # A tie up to float error is a tie: 1.0 - 0.1 exceeds 1.9 - 1.0 by one ulp.
    below_is_farther = (anchor - lo) - (hi - anchor) > 1e-9 * (hi - lo)
    return lo if below_is_farther else hi


def strength_sign(op, magnitude: np.ndarray) -> np.ndarray:
    """Per image, d magnitude / d strength: +1 above ``identity_at``, -1 below, and the
    direction of ``far_end`` at the identity itself."""
    side = np.sign(np.asarray(magnitude, dtype=np.float64) - op.identity_at)
    side[side == 0] = 1.0 if far_end(op) > op.identity_at else -1.0
    return side


def resolve_methods(ops: Mapping[str, Any], fade: Fade = None) -> dict[str, str]:
    """How each op's row is measured: ``"fade"`` for the ops ``fade`` selects (see
    fade.py), ``op_method`` for the rest."""
    faded = set(faded_ops(ops, fade))
    # A coupling op stays unprobed when faded too: its fade target is a blend with
    # another image of the batch.
    return {
        name: FADE if name in faded and not getattr(op, "couples_images", False) else op_method(op)
        for name, op in ops.items()
    }


def op_seed(probe_seed: int, name: str) -> int:
    """The torch seed an op runs under: fixed per (probe_seed, op name).

    Fixed across batches, so ``noise`` projects every batch's pixel gradient onto the
    same sample instead of turning its row into measurement noise. Keyed on the name
    rather than shared by all ops, because one shared seed would give two random ops
    the same sample and correlate their rows by construction. Sensaug seeded once per
    batch, so ``noise``'s sample also depended on what ran before it. crc32 rather
    than ``hash()``, which Python salts per process.
    """
    return zlib.crc32(f"{probe_seed}:{name}".encode())


def fd_step_for(fd_step: FdStep, name: str) -> float:
    """The finite-difference step for one op, as a fraction of its mag_range width."""
    if isinstance(fd_step, Mapping):
        return float(fd_step.get(name, DEFAULT_STEP))
    return float(fd_step)


@contextlib.contextmanager
def preserved_rng():
    """Save the torch RNG state (CPU, every CUDA device, MPS) and restore it on exit.
    Probe randomness must never consume from a training stream."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if mps is not None:
            torch.mps.set_rng_state(mps)


def require_eval(model: torch.nn.Module) -> None:
    """Raise unless every module is in eval mode.

    Checks every submodule, not just ``model.training``. A model whose ``eval()`` is
    overridden, or a submodule someone set back to train mode, couples the images just
    the same.
    """
    training = [name or "<root>" for name, m in model.named_modules() if m.training]
    if training:
        raise ProbeError(
            f"{len(training)} module(s) are in train mode (e.g. {training[:3]}). The "
            f"probe must run in eval(): train-mode BatchNorm couples the images in a "
            f"batch through its batch statistics, and d loss / d delta[i] then absorbs "
            f"the other images' deltas."
        )


def unit_batch(adapter: ModelAdapter, batch: Any) -> tuple[Tensor, Optional[Tensor]]:
    """``adapter.to_unit(batch)``, checked: (B, C, H, W) float in [0, 1], detached."""
    images01, label = adapter.to_unit(batch)
    if not torch.is_tensor(images01) or images01.dim() != 4 or not images01.is_floating_point():
        raise ProbeError(
            f"adapter.to_unit must return a (B, C, H, W) float tensor, got "
            f"{type(images01).__name__} {tuple(getattr(images01, 'shape', ()))}"
        )
    low, high = float(images01.min()), float(images01.max())
    if low < -_UNIT_SLACK or high > 1.0 + _UNIT_SLACK:
        raise ProbeError(
            f"adapter.to_unit must return [0, 1] RGB, got values in [{low:.4g}, {high:.4g}]. "
            f"Are the images still normalized, or still 0-255?"
        )
    return images01.detach().clamp(0.0, 1.0), label


def apply_op(name: str, op, images01: Tensor, label: Optional[Tensor], magnitude: Tensor, seed: int):
    """``op.apply`` under the op's seed, with its output checked.

    Returns ``(perturbed01, label)``. A photometric op hands the caller's label back
    as it was, so one clean batch can be reused across every op.
    """
    torch.manual_seed(seed)
    perturbed, label_out = op.apply(images01, magnitude, label)
    if not torch.is_tensor(perturbed) or perturbed.shape != images01.shape:
        raise ProbeError(
            f"op {name!r} must return images shaped like its input "
            f"{tuple(images01.shape)}, got {tuple(getattr(perturbed, 'shape', ()))}"
        )
    if getattr(op, "warps_label", False) and label is not None and label_out is label:
        raise ProbeError(
            f"op {name!r} is geometric but returned its label unwarped. An image moved "
            f"without its label makes the loss measure misalignment, not sensitivity."
        )
    return perturbed, label_out


class ProbeBatch(NamedTuple):
    """``probe_batch``'s result for one clean batch.

    grads   ``{op: (B,)}``, the batch's segment of each row of G (``collect_grads``)
    losses  ``{op: (B,)}``, each image's loss under the op at its magnitude, the
            augmented image itself (for a faded op, the end of the fade). NaN for an
            unprobed op. None when not asked for.
    clean   ``(B,)``, each image's loss on the clean image. None when not asked for.

    A batch-mean adapter loss can't be split back into images, so at batch size > 1 it
    gives NaN losses (never 0). ``losses[op] - clean`` is the op's loss increase.
    """

    grads: dict
    losses: Optional[dict]
    clean: Optional[np.ndarray]


def collect_grads(
    adapter: ModelAdapter,
    batch: Any,
    ops: Mapping[str, Any],
    magnitudes: Mapping[str, Any],
    *,
    probe_seed: int = 0,
    fd_step: FdStep = DEFAULT_STEP,
    check: bool = False,
    methods: Optional[Mapping[str, str]] = None,
    fade_at: float = 1.0,
) -> dict[str, np.ndarray]:
    """Probe every op on ONE shared clean batch.

    Returns ``{op name: (B,) float64 array}``, the batch's segment of each row of G,
    for every op in ``ops``. Every op sees the same clean images, so the columns line
    up across ops. An unprobed op (no magnitude, not faded) gets a NaN row: it stays
    in G, so G's shape doesn't depend on the vocab, and it can't be mistaken for a
    zero gradient.

    ``magnitudes[name]`` is a scalar or a (B,) array in the op's own units, within its
    ``mag_range``. Ops with no magnitude need no entry. ``fd_step`` is the
    finite-difference step as a fraction of each op's range width, one value for all
    ops or a mapping per op (a quantized op needs a step wider than one bin).

    ``methods`` is ``resolve_methods(ops, fade)``; None is ``op_method`` for every op.
    A ``"fade"`` op's row is ``d loss / d a`` at ``a = fade_at``, toward the op at its
    magnitude (see fade.py). Every other row is ``d loss / d strength``: the
    derivative in the magnitude times ``strength_sign`` (see the module docstring).

    The model must already be in eval mode. The torch RNG state is restored on return.

    ``check=True`` adds two runtime guards, meant for one batch per sweep:
    ``to_unit`` must be deterministic (no training-time augmentation left in the
    preprocessor), and changing image 0's magnitude must leave every other image's
    loss unchanged. The second needs B >= 2 and a per-image loss.
    """
    return probe_batch(
        adapter,
        batch,
        ops,
        magnitudes,
        probe_seed=probe_seed,
        fd_step=fd_step,
        check=check,
        methods=methods,
        fade_at=fade_at,
        losses=False,
    ).grads


def probe_batch(
    adapter: ModelAdapter,
    batch: Any,
    ops: Mapping[str, Any],
    magnitudes: Mapping[str, Any],
    *,
    probe_seed: int = 0,
    fd_step: FdStep = DEFAULT_STEP,
    check: bool = False,
    methods: Optional[Mapping[str, str]] = None,
    fade_at: float = 1.0,
    losses: bool = True,
) -> ProbeBatch:
    """``collect_grads``, plus (``losses=True``) each image's loss under every op and on
    the clean image. The arguments and the rows are ``collect_grads``'s."""
    require_eval(adapter.model)
    images01, label = unit_batch(adapter, batch)
    batch_size = int(images01.shape[0])
    if methods is None:
        methods = {name: op_method(op) for name, op in ops.items()}
    if FADE in methods.values():
        fade_at = check_fade_at(fade_at)

    rows = {}
    op_losses = {} if losses else None
    clean = None
    with preserved_rng():
        per_image = False
        if check:
            _check_deterministic(adapter, batch, images01)
        if check or losses:
            with torch.no_grad():
                clean_loss = _loss(adapter, images01, label, batch, batch_size, "<clean>")
            per_image = clean_loss.dim() == 1
            if losses:
                clean = _per_image(clean_loss, batch_size)
        if check:
            if batch_size < 2 or not per_image:
                logger.info(
                    "per-image independence check skipped: it needs a batch of at "
                    "least 2 and a per-image adapter.loss (batch size %d, per-image %s)",
                    batch_size,
                    per_image,
                )

        for name, op in ops.items():
            method = methods[name]
            if method in (UNPROBED, STATIC_PRUNED):
                rows[name] = np.full(batch_size, np.nan)
                if losses:
                    op_losses[name] = np.full(batch_size, np.nan)
                continue

            seed = op_seed(probe_seed, name)
            if method == FADE:
                magnitude = (
                    _batch_magnitude(name, op, magnitudes, batch_size)
                    if getattr(op, "mag_range", None) is not None
                    else None
                )
                target, label_out = _fade_target(name, op, images01, label, magnitude, seed)
                if check and per_image and batch_size > 1:
                    evaluate = _fade_evaluator(adapter, name, images01, target, label_out, batch)
                    alpha = np.full(batch_size, fade_at)
                    _check_independence(evaluate, name, alpha, fd_step_for(fd_step, name), 0.0, 1.0)
                grad, loss = _fade_row(adapter, name, images01, target, label_out, batch, fade_at)
                rows[name] = _finalize(name, grad, batch_size)
                if losses:
                    if fade_at != 1.0:
                        # The row is taken partway along the fade; the loss is the op's own.
                        with torch.no_grad():
                            loss = _loss(adapter, target, label_out, batch, batch_size, name)
                    op_losses[name] = _per_image(loss, batch_size)
                continue

            magnitude = _batch_magnitude(name, op, magnitudes, batch_size)
            lo, hi = op.mag_range
            step = fd_step_for(fd_step, name) * (hi - lo)
            evaluate = _evaluator(adapter, name, op, images01, label, batch, seed)

            if check and per_image and batch_size > 1:
                _check_independence(evaluate, name, magnitude, step, lo, hi)

            if method == AUTOGRAD:
                grad, loss = _autograd_row(adapter, name, op, images01, label, batch, magnitude, seed)
            elif method == TANGENT:
                grad, loss = _tangent_row(adapter, name, op, images01, label, batch, magnitude, seed)
            else:
                grad = fd_grad(evaluate, magnitude, step=step, lo=lo, hi=hi)
                loss = None
            rows[name] = _finalize(name, grad, batch_size) * strength_sign(op, magnitude)
            if losses:
                # Finite differences never evaluate at the magnitude itself.
                op_losses[name] = evaluate(magnitude) if loss is None else _per_image(loss, batch_size)
    return ProbeBatch(rows, op_losses, clean)


def _batch_magnitude(name, op, magnitudes, batch_size) -> np.ndarray:
    if name not in magnitudes:
        raise ProbeError(f"no magnitude given for op {name!r}")
    value = np.asarray(magnitudes[name], dtype=np.float64)
    if value.ndim == 0:
        value = np.full(batch_size, float(value))
    if value.shape != (batch_size,):
        raise ProbeError(
            f"op {name!r}: magnitude must be a scalar or ({batch_size},), got {value.shape}"
        )
    lo, hi = op.mag_range
    if not np.isfinite(value).all() or value.min() < lo or value.max() > hi:
        raise ProbeError(
            f"op {name!r}: magnitudes span [{value.min():.4g}, {value.max():.4g}], "
            f"outside its mag_range [{lo}, {hi}]"
        )
    return value


def _loss(adapter, images01, label, batch, batch_size, name) -> Tensor:
    loss = adapter.loss(images01, label, batch)
    if not torch.is_tensor(loss) or (loss.dim() != 0 and tuple(loss.shape) != (batch_size,)):
        raise ProbeError(
            f"op {name!r}: adapter.loss must return a ({batch_size},) per-image loss or a "
            f"scalar, got {tuple(getattr(loss, 'shape', ()))}"
        )
    return loss


def _autograd_row(adapter, name, op, images01, label, batch, magnitude, seed):
    """``(d loss_i / d m_i, the losses at m)``."""
    delta = torch.tensor(
        magnitude, dtype=images01.dtype, device=images01.device, requires_grad=True
    )
    perturbed, label_out = apply_op(name, op, images01, label, delta, seed)
    if not perturbed.requires_grad:
        raise ProbeError(
            f"op {name!r} declares grad_mode='autograd' but its output does not depend "
            f"on the magnitude through autograd."
        )
    loss = _loss(adapter, perturbed, label_out, batch, images01.shape[0], name)
    # A per-image loss is summed: in eval mode term i depends on delta[i] alone, so
    # grad[i] = d loss_i / d delta[i]. A scalar loss (e.g. mmseg's batch mean) gives
    # the same up to a per-image weight, which the per-image scale normalization in
    # the analysis divides out.
    (grad,) = torch.autograd.grad(loss.sum(), delta, allow_unused=True)
    if grad is None:
        raise ProbeError(f"op {name!r}: the loss does not depend on the magnitude")
    return grad, loss


def _tangent_row(adapter, name, op, images01, label, batch, magnitude, seed):
    """``(d loss_i / d m_i, the losses at m)``, the derivative through the op's exact derivative: the loss at
    ``op(x, m) + delta * op.tangent(x, m)``, differentiated in ``delta`` at 0. The images
    the model sees are the op's own output; only the direction comes from the tangent."""
    target, label_out = _fade_target(name, op, images01, label, magnitude, seed)
    torch.manual_seed(seed)
    with torch.no_grad():
        direction = op.tangent(
            images01, torch.as_tensor(magnitude, dtype=images01.dtype, device=images01.device)
        )
    if not torch.is_tensor(direction) or direction.shape != images01.shape:
        raise ProbeError(
            f"op {name!r}: its tangent must be shaped like the images "
            f"{tuple(images01.shape)}, got {tuple(getattr(direction, 'shape', ()))}"
        )
    direction = direction.detach().to(images01.device, images01.dtype)
    delta = torch.zeros(
        images01.shape[0], dtype=images01.dtype, device=images01.device, requires_grad=True
    )
    moved = target + delta.view(-1, *([1] * (images01.dim() - 1))) * direction
    loss = _loss(adapter, moved, label_out, batch, images01.shape[0], name)
    # Summed for the same reason as _autograd_row: term i depends on delta[i] alone.
    (grad,) = torch.autograd.grad(loss.sum(), delta, allow_unused=True)
    if grad is None:
        raise ProbeError(f"op {name!r}: the loss does not depend on the tangent")
    return grad, loss


def _fade_target(name, op, images01, label, magnitude, seed):
    """``op(x, m)`` and its label, under no_grad: the end of a fade, the base of a
    tangent. ``magnitude`` is None for an op with no magnitude."""
    if magnitude is not None:
        magnitude = torch.as_tensor(magnitude, dtype=images01.dtype, device=images01.device)
    with torch.no_grad():
        target, label_out = apply_op(name, op, images01, label, magnitude, seed)
    return target.detach(), label_out


def _fade_row(adapter, name, images01, target, label, batch, at):
    """``(d loss_i / d a_i, the losses at a)``, at ``a = at``."""
    alpha = fade_alpha(at, images01, requires_grad=True)
    faded = fade_images(images01, target, alpha)
    loss = _loss(adapter, faded, label, batch, images01.shape[0], name)
    # Summed for the same reason as _autograd_row: term i depends on alpha[i] alone.
    (grad,) = torch.autograd.grad(loss.sum(), alpha, allow_unused=True)
    if grad is None:
        raise ProbeError(f"op {name!r}: the loss does not depend on the fade")
    return grad, loss


def _fade_evaluator(adapter, name, images01, target, label, batch):
    """alpha (B,) -> per-image losses (B,), under no_grad."""

    def perturb(alpha: Tensor):
        return fade_images(images01, target, alpha), label

    return _losses(adapter, name, images01, batch, perturb)


def _evaluator(adapter, name, op, images01, label, batch, seed):
    """magnitude (B,) -> per-image losses (B,), under no_grad. Every call reseeds the
    op, so repeated evaluations share one random sample."""

    def perturb(delta: Tensor):
        return apply_op(name, op, images01, label, delta, seed)

    return _losses(adapter, name, images01, batch, perturb)


def _losses(adapter, name, images01, batch, perturb):
    """values (B,) -> the per-image losses of ``perturb(values)``, under no_grad."""
    batch_size = images01.shape[0]

    def evaluate(values: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(values, dtype=images01.dtype, device=images01.device)
        with torch.no_grad():
            perturbed, label_out = perturb(values)
            loss = _loss(adapter, perturbed, label_out, batch, batch_size, name)
        if loss.dim() == 0:
            if batch_size > 1:
                raise ProbeError(
                    f"op {name!r} is probed by finite differences, which needs a "
                    f"per-image adapter.loss at batch size {batch_size}; this adapter "
                    f"returns a scalar. Return ({batch_size},) losses or use batch size 1."
                )
            loss = loss.reshape(1)
        # To CPU before float64: MPS has no float64.
        return loss.detach().cpu().to(torch.float64).numpy()

    return evaluate


def _check_deterministic(adapter, batch, images01) -> None:
    again, _ = unit_batch(adapter, batch)
    if not torch.equal(images01, again):
        raise ProbeError(
            "adapter.to_unit returned different images for the same batch. Something in "
            "it is random, most likely training-time augmentation left in the "
            "preprocessor. The probe needs CLEAN images: a random per-image "
            "augmentation is a factor shared by every op and inflates all of R."
        )


def _check_independence(evaluate, name, magnitude, step, lo, hi) -> None:
    step = min(step, (hi - lo) / 2)
    bumped = magnitude.copy()
    bumped[0] = magnitude[0] + step if magnitude[0] + step <= hi else magnitude[0] - step
    base, moved = evaluate(magnitude), evaluate(bumped)
    leak = float(np.abs(moved[1:] - base[1:]).max())
    if leak > _LEAK_RTOL * max(1.0, float(np.abs(base).max())):
        raise ProbeError(
            f"op {name!r}: changing image 0's magnitude moved another image's loss by "
            f"{leak:.3g}. Something couples the images in a batch: an op that pools the "
            f"magnitude (e.g. delta.mean()), batch statistics inside the op, or a layer "
            f"that mixes images. The per-image gradients would not be per image."
        )


def _per_image(loss: Tensor, batch_size: int) -> np.ndarray:
    """A loss the probe already ran, as (B,) float64 per image. A batch mean at B > 1
    can't be split back into images: NaN."""
    loss = loss.detach()
    if loss.dim() == 0:
        if batch_size > 1:
            return np.full(batch_size, np.nan)
        loss = loss.reshape(1)
    # To CPU before float64: MPS has no float64.
    return loss.cpu().to(torch.float64).numpy()


def _finalize(name, grad, batch_size) -> np.ndarray:
    if torch.is_tensor(grad):
        grad = grad.detach().cpu().to(torch.float64).numpy()
    grad = np.asarray(grad, dtype=np.float64)
    if grad.shape != (batch_size,):
        raise ProbeError(
            f"op {name!r}: expected a gradient of shape ({batch_size},), got {grad.shape}"
        )
    if not np.isfinite(grad).all():
        raise ProbeError(f"op {name!r}: non-finite gradient {grad.tolist()}")
    return grad
