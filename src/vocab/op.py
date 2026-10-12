"""What one augmentation op is: `OpSpec`.

Every op in the vocabulary, whichever family declares it, is described by one
OpSpec and applied through one call:

    images_out, label_out = spec.apply(images, magnitude, label=None)

Tensor contract, shared by every family:
    images:     float32, (B, 3, H, W), RGB, values in [0, 1]
    label:      integer class ids, (B, H, W) or (B, 1, H, W), or None
    magnitude:  a scalar or a length-B vector (one magnitude per image), in the
                units `mag_range` is written in. Ops with no magnitude
                (`mag_range is None`) ignore it.

The label comes back unchanged unless `warps_label`, in which case it has been
moved with the image (nearest-neighbour, out-of-frame pixels set to the ignore
index). A geometric op that left the label in place would make the loss measure
image-label misalignment instead of the model's sensitivity, so a geometric
spec that does not warp its label cannot be constructed.

This module only describes ops. The implementations are in `diff_ops` (the
differentiable Sensaug ports) and in the torchvision-backed families, and the
behavioural checks (identity, per-image independence, gradient, label) are in
`audit`.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import torch

Tensor = torch.Tensor
Magnitude = Union[None, float, Tensor]
OpFn = Callable[[Tensor, Magnitude, Optional[Tensor]], Tuple[Tensor, Optional[Tensor]]]
TangentFn = Callable[[Tensor, Magnitude], Tensor]

KINDS = ("photometric", "geometric")

# "autograd": d(output) / d(magnitude) flows through torch autograd, per image,
#             so the gradient probe can differentiate the op directly.
# "none":     the output carries no gradient w.r.t. the magnitude (e.g. the
#             torchvision reference ops, which take the magnitude as a python
#             float and run on 8-bit images).
GRAD_MODES = ("autograd", "none")


@dataclass(frozen=True)
class OpSpec:
    """One augmentation op.

    name         unique across every family; the key R is indexed by
    family       the `families/` module that declares it
    kind         "photometric" (moves no pixels) or "geometric" (moves pixels)
    source       where the implementation and its units come from
    mag_range    (lo, hi) in the op's own units, as its family's paper states
                 them; None for an op with no magnitude
    identity_at  the magnitude at which the op returns its input unchanged;
                 None exactly when mag_range is None
    grad_mode    one of GRAD_MODES
    warps_label  True exactly for geometric ops
    fn           the implementation, (images, magnitude, label) -> (images, label)
    tangent      optional, (images, magnitude) -> (B, C, H, W): the exact derivative
                 of the output w.r.t. the magnitude, per image, for an op autograd
                 can't differentiate (e.g. torchvision's enhance ops, a blend before
                 the 8-bit cast). The probe uses it instead of finite differences.
                 Photometric ops only: along a tangent the label stays where the op
                 put it, and a geometric op's row must move the label with the image.
    """

    name: str
    family: str
    kind: str
    source: str
    mag_range: Optional[Tuple[float, float]]
    identity_at: Optional[float]
    grad_mode: str
    warps_label: bool
    fn: OpFn = field(repr=False, compare=False)
    tangent: Optional[TangentFn] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind {self.kind!r} not in {KINDS}")
        if self.grad_mode not in GRAD_MODES:
            raise ValueError(f"{self.name}: grad_mode {self.grad_mode!r} not in {GRAD_MODES}")
        if self.warps_label != (self.kind == "geometric"):
            raise ValueError(
                f"{self.name}: warps_label={self.warps_label} but kind={self.kind!r}; "
                "geometric ops must warp the label and photometric ops must not"
            )
        if (self.mag_range is None) != (self.identity_at is None):
            raise ValueError(
                f"{self.name}: mag_range and identity_at must both be set or both be None"
            )
        if self.mag_range is not None:
            lo, hi = self.mag_range
            if not lo < hi:
                raise ValueError(f"{self.name}: mag_range {self.mag_range} is not (lo, hi)")
            if not lo <= self.identity_at <= hi:
                raise ValueError(
                    f"{self.name}: identity_at {self.identity_at} is outside "
                    f"mag_range {self.mag_range}"
                )
        if self.tangent is not None and (self.mag_range is None or self.warps_label):
            raise ValueError(
                f"{self.name}: only a photometric op with a magnitude can carry a tangent"
            )

    @property
    def has_magnitude(self) -> bool:
        return self.mag_range is not None

    def apply(
        self,
        images: Tensor,
        magnitude: Magnitude = None,
        label: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply the op to an image batch and, if given, its label."""
        if self.has_magnitude and magnitude is None:
            raise ValueError(f"{self.name} takes a magnitude in {self.mag_range}")
        return self.fn(images, magnitude, label)
