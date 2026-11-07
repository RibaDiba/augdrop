"""Each copy's training loss, L_m (Adversarial AutoAugment's Eq. 5).

The batch the target trains on is M copies of N images, copy-major (copy m is images
``m N .. (m + 1) N - 1``, augmented by policy m), and Eq. 5's ``L_m`` is the mean loss over
copy m's N images. ``copy_losses`` computes it from the step's own logits with the
caller's loss function (the segmentor's decode-head loss, so the same loss and reduction
the target trains on), one slice per copy, with no extra forward pass.
"""

from typing import Callable, Sequence

import torch

Tensor = torch.Tensor


def copy_losses(loss_of: Callable[[Tensor, Sequence], Tensor], logits: Tensor, samples: Sequence, num_copies: int) -> Tensor:
    """(M,): ``loss_of(logits[copy m], samples[copy m])`` for each copy m, no gradient."""
    total = int(logits.shape[0])
    if total % int(num_copies) or len(samples) != total:
        raise ValueError(f"{total} logits and {len(samples)} samples are not {num_copies} equal copies")
    per_copy = total // int(num_copies)
    with torch.no_grad():
        return torch.stack(
            [
                loss_of(logits[m * per_copy : (m + 1) * per_copy], samples[m * per_copy : (m + 1) * per_copy]).float()
                for m in range(int(num_copies))
            ]
        )
