"""Fade: a differentiable row of G for an op the probe can't differentiate.

An op whose output carries no gradient w.r.t. its magnitude (vocab ``grad_mode="none"``,
e.g. the torchvision families, which run on 8-bit images) has only finite differences,
and an op with no magnitude at all (AutoContrast, Equalize, Invert) has nothing to
differentiate, so its row is NaN. Fading the op in gives both a gradient:

    x(a) = x + a * (op(x, m) - x),    a in [0, 1]

``op(x, m)`` is computed once, under no_grad, at the op's probe magnitude ``m`` (its
intensity; an op with no magnitude has one output). ``x(a)`` is differentiable in ``a``
whatever the op is, and the row is

    d loss_i / d a  =  < grad_x loss_i(x_i(a)), op(x_i, m) - x_i >

At ``a = 1`` (the default) the image is the augmented one, so the row is how image i's
loss moves along the augmentation, measured where the augmentation lands. The row's
magnitude column holds ``m``, and ``a`` is the sweep's ``meta["fade_at"]``.

What a faded row is not:

- **Not d loss / d m.** It is a derivative along the fade at a fixed intensity, so a G
  that fades an op is not comparable with one that took that op by finite differences.
  ``Sweep.methods`` records which ops were faded.
- **Not label-exact between the ends for a geometric op.** For 0 < a < 1 the image is
  a superposition of the clean frame and the warped one, and the label (the op's
  warped label) matches only the warped frame. At a = 1 image and label agree; the
  derivative points toward the ghost.
- **Zero where the op does nothing.** An op that is the identity on an image
  (AutoContrast on an image already spanning 0 to 255) gives an exact 0. That is a
  real zero, not missing data.

The independence check (``collect_grads(check=True)``) bumps one image's ``a``, which
covers the model. The op itself is applied once, outside the fade, so its own
per-image independence is vocab's audit (``per_image``).
"""

from typing import Any, Mapping, Sequence, Union

import torch
from torch import Tensor

FADE = "fade"

#: ``fade="non_differentiable"``: every op whose ``grad_mode`` isn't ``"autograd"``.
NON_DIFFERENTIABLE = "non_differentiable"
FADE_RULES = (NON_DIFFERENTIABLE,)

#: None (no op is faded), a rule from FADE_RULES, or a list of op names.
Fade = Union[None, str, Sequence[str]]


def faded_ops(ops: Mapping[str, Any], fade: Fade) -> list[str]:
    """The names in ``ops`` that ``fade`` selects, in ``ops`` order.

    A list may name any op, a differentiable one included. A name that isn't in
    ``ops`` raises, so a typo can't silently fade nothing.
    """
    if fade is None:
        return []
    if isinstance(fade, str):
        if fade not in FADE_RULES:
            raise ValueError(f"fade rule {fade!r} is not one of {FADE_RULES}, nor a list of ops")
        return [name for name, op in ops.items() if op.grad_mode != "autograd"]
    chosen = [str(name) for name in fade]
    unknown = [name for name in chosen if name not in ops]
    if unknown:
        raise ValueError(f"fade names ops {unknown} that are not in the vocab {list(ops)}")
    chosen = set(chosen)
    return [name for name in ops if name in chosen]


def check_fade_at(at: float) -> float:
    """``at`` as a float, or ValueError outside [0, 1]."""
    at = float(at)
    if not 0.0 <= at <= 1.0:
        raise ValueError(f"fade_at must be in [0, 1], got {at}")
    return at


def fade_images(images01: Tensor, target: Tensor, alpha: Tensor) -> Tensor:
    """``images01 + alpha * (target - images01)``, with ``alpha`` (B,), one per image."""
    return images01 + alpha.view(-1, *([1] * (images01.dim() - 1))) * (target - images01)


def fade_alpha(at: float, images01: Tensor, requires_grad: bool = False) -> Tensor:
    """A (B,) ``a`` at ``at``, on the images' device and dtype."""
    batch_size = images01.shape[0]
    return torch.full(
        (batch_size,), float(at), dtype=images01.dtype, device=images01.device, requires_grad=requires_grad
    )
