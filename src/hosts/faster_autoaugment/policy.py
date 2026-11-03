"""Faster AutoAugment's policy as a torch module, for the search: dda's ``Policy`` /
``SubPolicy`` / ``SubPolicyStage`` / ``_Operation`` (github.com/moskomule/dda,
``faster_autoaugment/policy.py`` and ``dda/operations.py``), over vocab ``faster_aa``'s 26
ops and carrying a segmentation mask.

Search mode only (dda's ``training`` branch); training on the found policy samples it
through the host (``host.py``), as dda's eval branch does. In search mode:

- an op is applied with a relaxed Bernoulli mask, ``b op(x) + (1 - b) x``, with
  ``b ~ RelaxedBernoulli(temperature, p)`` per image, so its probability p gets a
  gradient (``_Operation.forward``). Its magnitude is ``clamp(mu, 0, 1)`` times dda's
  ``magnitude_scale`` (``faster_aa.POLICY_SCALES``): a vocab op's units
- a stage is the weighted sum of every op's output, the weights
  ``softmax(w / temperature)`` (``SubPolicyStage``), so the op choice gets a gradient
- a sub-policy runs its stages in turn; the policy splits the batch into ``num_chunks``
  chunks and runs one sub-policy, drawn uniformly, on each (``Policy.forward``)

The ops are the vocab's (``vocab.families.faster_aa``). Solarize, Posterize and Cutout
step with the magnitude, so their magnitude gets dda's straight-through gradient: the
forward value is the op's, and ``d out / d mu`` is taken as 1 for every pixel
(``dda.functional.ste``, which also stops the image's gradient through them).
AutoContrast and Equalize are straight-through in the image (``ste(output, img)``). The
rest are differentiated by autograd.

**The mask** (the segmentation adaptation; the paper and dda are classification-only).
It is soft, ``(B, K + 1, H, W)``: one channel per class, the last one the ignore index,
summing to 1 per pixel (``search.soft_mask``). It goes through the same weighted sums as
the image, as AutoAlbument's segmentation search does: a geometric op warps it
bilinearly with the image's matrix (HorizontalFlip flips it), and the pixels the warp
brings in from outside the frame go to the ignore channel; every photometric op,
Cutout and SamplePairing leave it as it is. So the blended mask says, per pixel, how
much of each class the blended image shows there.

dda's ``Policy.dda_operations`` lists TranslateY twice and no TranslateX, and leaves out
Cutout and SamplePairing; this searches the paper's 17 (vocab faster_aa's docstring).
The 9 signed ops are split into _pos/_neg, each with its own weight, p and mu, where dda
learns one mu and p per op and draws its sign at p = 0.5.
"""

import random
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import RelaxedBernoulli

from vocab import get_op
from vocab.diff_ops import _warp
from vocab.families import faster_aa

from .host import VOCAB_NAMES

__all__ = ["Operation", "Policy", "SubPolicy", "SubPolicyStage", "straight_through"]

Batch = Tuple[Tensor, Optional[Tensor]]


def straight_through(forward: Tensor, backward: Tensor) -> Tensor:
    """dda's ``ste``: ``forward``'s value, with ``backward``'s gradient passed through as
    the identity (summed to its shape); ``forward`` itself gets none."""
    return forward.detach() + (backward - backward.detach())


def warp_mask(mask: Tensor, matrix: Tensor) -> Tensor:
    """A soft mask warped by an image's matrix (bilinear, zeros outside), the mass the
    frame lost given to the ignore channel (the last), so every pixel still sums to 1."""
    warped = _warp(mask, matrix)
    lost = (1.0 - warped.sum(dim=1, keepdim=True)).clamp(min=0.0)
    return torch.cat([warped[:, :-1], warped[:, -1:] + lost], dim=1)


class Operation(nn.Module):
    """One vocab op inside one stage: dda's ``_Operation``, with a learnable probability
    and, for an op with a magnitude, a learnable mu in [0, 1]."""

    def __init__(self, name: str, *, temperature: float):
        super().__init__()
        self.name = name
        self.spec = get_op(name)
        self.transform_id, self.sign = faster_aa.TRANSFORMS[name]
        scale = faster_aa.POLICY_SCALES.get(self.transform_id)
        self.scale = None if scale is None else float(scale)
        self._probability = nn.Parameter(torch.empty(1))
        self._magnitude = None if scale is None else nn.Parameter(torch.empty(1))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    @property
    def probability(self) -> Tensor:
        return self._probability.clamp(0.0, 1.0)

    @property
    def magnitude(self) -> Optional[Tensor]:
        """In the op's units: ``clamp(mu, 0, 1) * scale``."""
        if self._magnitude is None:
            return None
        return self._magnitude.clamp(0.0, 1.0) * self.scale

    def _transform(self, images: Tensor, mask: Optional[Tensor]) -> Batch:
        b = images.shape[0]
        t = self.transform_id
        if self._magnitude is not None:
            m = self.magnitude.expand(b)
        if t in faster_aa.GEOMETRIC_IDS:
            matrix = faster_aa.geometric_matrix(t, images, self.sign * m)
            return _warp(images, matrix), None if mask is None else warp_mask(mask, matrix)
        if t == "HorizontalFlip":
            return images.flip(-1), None if mask is None else mask.flip(-1)
        if t in faster_aa.STEPPED_IDS:
            out, _ = self.spec.fn(images, m.detach())
            return straight_through(out, m.view(-1, 1, 1, 1)), mask
        if t in faster_aa.STRAIGHT_THROUGH_IDS:
            out, _ = self.spec.fn(images)
            return straight_through(out, images), mask
        out, _ = self.spec.fn(images, None if self._magnitude is None else m)
        return out, mask

    def forward(self, images: Tensor, mask: Optional[Tensor] = None) -> Batch:
        b = images.shape[0]
        gate = RelaxedBernoulli(self.temperature, probs=self.probability).rsample((b, 1, 1)).view(b, 1, 1, 1)
        out, out_mask = self._transform(images, mask)
        images = (gate * out + (1.0 - gate) * images).clamp(0.0, 1.0)
        if mask is not None and out_mask is not mask:
            mask = gate * out_mask + (1.0 - gate) * mask
        return images, mask


class SubPolicyStage(nn.Module):
    """One stage: every op's output, weighted by ``softmax(w / temperature)``."""

    def __init__(self, names: Sequence[str], *, temperature: float, bernoulli_temperature: float):
        super().__init__()
        self.operations = nn.ModuleList([Operation(n, temperature=bernoulli_temperature) for n in names])
        self._weights = nn.Parameter(torch.empty(len(names)))
        self.temperature = float(temperature)

    @property
    def weights(self) -> Tensor:
        return self._weights.div(self.temperature).softmax(0)

    def forward(self, images: Tensor, mask: Optional[Tensor] = None) -> Batch:
        weights = self.weights
        out_images = torch.zeros_like(images)
        # A photometric op leaves the mask as it is, so its share of the blend is the
        # input mask times its weight: summed once rather than per op.
        out_mask = None if mask is None else torch.zeros_like(mask)
        unmoved = images.new_zeros(())
        for w, op in zip(weights, self.operations):
            op_images, op_mask = op(images, mask)
            out_images = out_images + w * op_images
            if mask is None:
                continue
            if op_mask is mask:
                unmoved = unmoved + w
            else:
                out_mask = out_mask + w * op_mask
        if mask is not None:
            out_mask = out_mask + unmoved * mask
        return out_images, out_mask


class SubPolicy(nn.Module):
    def __init__(self, names: Sequence[str], *, operation_count: int, temperature: float, bernoulli_temperature: float):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SubPolicyStage(names, temperature=temperature, bernoulli_temperature=bernoulli_temperature)
                for _ in range(operation_count)
            ]
        )

    def forward(self, images: Tensor, mask: Optional[Tensor] = None) -> Batch:
        for stage in self.stages:
            images, mask = stage(images, mask)
        return images, mask


class Policy(nn.Module):
    """dda's ``Policy``: ``num_sub_policies`` sub-policies of ``operation_count`` stages,
    over ``names`` (default vocab faster_aa's 26 ops). Every parameter starts uniform on
    [0, 1], as dda initializes it. Images in [0, 1] in, [0, 1] out (dda normalizes inside;
    here the caller does)."""

    def __init__(
        self,
        names: Sequence[str] = VOCAB_NAMES,
        *,
        num_sub_policies: int = 10,
        operation_count: int = 2,
        temperature: float = 0.05,
        bernoulli_temperature: float = 0.05,
        num_chunks: int = 8,
    ):
        super().__init__()
        self.names = tuple(names)
        if self.names != VOCAB_NAMES:
            raise ValueError("the policy file format holds vocab faster_aa's ops in its order; search over those")
        self.num_sub_policies = int(num_sub_policies)
        self.operation_count = int(operation_count)
        self.temperature = float(temperature)
        self.bernoulli_temperature = float(bernoulli_temperature)
        self.num_chunks = int(num_chunks)
        self.sub_policies = nn.ModuleList(
            [
                SubPolicy(
                    self.names,
                    operation_count=self.operation_count,
                    temperature=self.temperature,
                    bernoulli_temperature=self.bernoulli_temperature,
                )
                for _ in range(self.num_sub_policies)
            ]
        )
        for p in self.parameters():
            nn.init.uniform_(p, 0.0, 1.0)

    def forward(self, images: Tensor, mask: Optional[Tensor] = None) -> Batch:
        chunks = max(1, min(self.num_chunks, images.shape[0]))
        image_chunks = images.chunk(chunks)
        mask_chunks = (None,) * len(image_chunks) if mask is None else mask.chunk(chunks)
        outs = [self.sub_policies[random.randrange(self.num_sub_policies)](x, m) for x, m in zip(image_chunks, mask_chunks)]
        out_images = torch.cat([o[0] for o in outs])
        out_mask = None if mask is None else torch.cat([o[1] for o in outs])
        return out_images, out_mask

    @torch.no_grad()
    def export(self) -> Dict[str, object]:
        """The policy as the host reads it (``host.load_policy``): (L, K, A) arrays of
        each stage's op weights (the softmax), probabilities, and magnitudes in the ops'
        units (NaN for an op with none)."""
        shape = (self.num_sub_policies, self.operation_count, len(self.names))
        weights = torch.empty(shape)
        probabilities = torch.empty(shape)
        magnitudes = torch.full(shape, float("nan"))
        for l, sub in enumerate(self.sub_policies):
            for k, stage in enumerate(sub.stages):
                weights[l, k] = stage.weights.detach().cpu()
                for a, op in enumerate(stage.operations):
                    probabilities[l, k, a] = float(op.probability)
                    if op.magnitude is not None:
                        magnitudes[l, k, a] = float(op.magnitude)
        return {
            "ops": self.names,
            "weights": weights.double().numpy(),
            "probabilities": probabilities.double().numpy(),
            "magnitudes": magnitudes.double().numpy(),
        }
