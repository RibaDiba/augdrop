"""DeepAA's policy as it is searched: one categorical per layer over the kept transforms
(``policy.DA_Policy_logits``).

Each layer's logits start at DeepAA's identity init (``policy_init='identity'``: 8 on
Identity, 0 elsewhere, minus their mean), so Identity has about 0.956 of a full layer.
Over a pruned space the logits hold only the kept transforms (``space.kept_transforms``):
the policy is ``K x |kept|`` numbers, and each search step applies ``|kept|`` transforms
per image, not 139. ``export`` writes all 139 columns, the left-out ones at 0, in DeepAA's
own ``.npz`` format (``DeepAA_utils.save_policy``), which ``hosts.deep_autoaugment.
load_policy`` reads.
"""

import os
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from hosts.deep_autoaugment import DEEPAA_OPS, L_MAGS

from .space import LEVELS, NUM_TRANSFORMS, OPS

__all__ = ["LayerPolicy", "export", "save_npz"]


class LayerPolicy(nn.Module):
    """One layer's categorical over the kept transforms ``kept`` (full indices, Identity
    first)."""

    def __init__(self, kept: Sequence[int]):
        super().__init__()
        kept = np.asarray(kept, dtype=np.int64)
        if kept.size == 0 or kept[0] != 0:
            raise ValueError("a layer's kept transforms must start with Identity (0)")
        self.register_buffer("kept", torch.as_tensor(kept))
        init = torch.zeros(len(kept))
        init[0] = 8.0
        self.logits = nn.Parameter(init - init.mean())

    def probs(self) -> torch.Tensor:
        """(|kept|,): the layer's softmax."""
        return torch.softmax(self.logits, dim=-1)

    def sample(self, n: int, generator: torch.Generator) -> np.ndarray:
        """``n`` transforms (full indices), drawn on the CPU from ``generator``."""
        probs = self.probs().detach().double().cpu()
        picks = torch.multinomial(probs, int(n), replacement=True, generator=generator)
        return self.kept.cpu()[picks].numpy()

    def entropy(self) -> float:
        p = self.probs().detach()
        return float(-(p * torch.log(p.clamp_min(1e-30))).sum())


def export(layers: Iterable[LayerPolicy]) -> dict:
    """The searched policy as ``load_policy``'s dict: ``policy_probs`` (K, 139), the
    left-out transforms at 0."""
    rows = []
    for layer in layers:
        row = np.zeros(NUM_TRANSFORMS)
        row[layer.kept.cpu().numpy()] = layer.probs().detach().double().cpu().numpy()
        rows.append(row)
    return {"policy_probs": np.stack(rows), "l_mags": L_MAGS, "ops": OPS.copy(), "mags": LEVELS.copy()}


def save_npz(path: str, policy: dict, pruned_ops: Iterable[str] = ()) -> str:
    """``policy`` in DeepAA's ``save_policy`` format (``policy_probs``, ``l_ops``,
    ``l_mags``, ``ops`` / ``mags`` (T, 1), ``op_names`` ``Op:(low,,high)``), plus
    ``pruned_ops`` (space-separated). Written atomically."""
    names = np.array([f"{n}:({float(lo)},,{float(hi)})" for n, lo, hi in DEEPAA_OPS])
    tmp = f"{path}.tmp.npz"
    np.savez(
        tmp,
        policy_probs=np.asarray(policy["policy_probs"], dtype=np.float64),
        l_ops=np.int64(len(DEEPAA_OPS)),
        l_mags=np.int64(policy["l_mags"]),
        ops=np.asarray(policy["ops"]).reshape(-1, 1).astype(np.int32),
        mags=np.asarray(policy["mags"]).reshape(-1, 1).astype(np.int32),
        op_names=names,
        pruned_ops=np.array(" ".join(sorted(set(pruned_ops)))),
    )
    os.replace(tmp, path)
    return path
