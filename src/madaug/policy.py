"""MADAug's policy network and what is read off it.

``Projection`` is MADAug's ``networks/projection.py``: a linear layer (or an MLP with
``n_layers`` hidden ReLU layers of ``n_hidden``) from the task model's feature to two
values per policy output, the first half levels and the second half op weights
(``MDAAug.predict_aug_params``: ``torch.split(a_params, n_ops)``). The paper's Table 5
found no hidden layer best (h = 0), its default.
"""

import contextlib
from typing import Callable, Iterable, Tuple

import numpy as np
import torch
from torch import nn

from hosts.madaug import OPS_NAMES

__all__ = ["Projection", "eval_mode", "policy_levels", "predict"]


class Projection(nn.Module):
    """MADAug's policy: features (B, in_features) -> (B, 2 * n_outputs)."""

    def __init__(self, in_features: int, n_layers: int = 0, n_hidden: int = 128, n_outputs: int = len(OPS_NAMES)):
        super().__init__()
        self.n_outputs = int(n_outputs)
        if n_layers > 0:
            layers = [nn.Linear(in_features, n_hidden), nn.ReLU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]
            layers.append(nn.Linear(n_hidden, 2 * self.n_outputs))
        else:
            layers = [nn.Linear(in_features, 2 * self.n_outputs)]
        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def predict(policy: Projection, features: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(levels, p)``, each (B, n_outputs): levels = sigmoid, p = softmax(· / T).
    MADAug samples training draws at its temperature and relaxes the policy update at
    T = 1 (``predict_aug_params``' exploit / explore)."""
    levels, weights = torch.split(policy(features), policy.n_outputs, dim=1)
    return torch.sigmoid(levels), torch.softmax(weights / float(temperature), dim=-1)


@contextlib.contextmanager
def eval_mode(model: nn.Module):
    """eval() for the duration, then every module's own flag back (as ``probe.frozen``).
    MADAug reads the policy's features, and runs its policy update, with the task model
    in eval (``predict_aug_params`` calls ``gf_model.eval()``)."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield model
    finally:
        for module, training in modes:
            module.train(training)


def policy_levels(
    policy: Projection,
    features: Callable[[torch.Tensor], torch.Tensor],
    batches: Iterable[torch.Tensor],
) -> np.ndarray:
    """The policy's levels for every image of ``batches`` (each a tensor ``features``
    takes), stacked in order: (N, n_outputs). For the probe, which asks MADAug for its own
    magnitude on each probe image."""
    rows = []
    with torch.no_grad():
        for images in batches:
            levels, _ = predict(policy, features(images))
            rows.append(levels.double().cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, policy.n_outputs))
