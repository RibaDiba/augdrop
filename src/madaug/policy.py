"""MADAug's policy network and what is read off it.

``Projection`` is MADAug's ``networks/projection.py``: a linear layer (or an MLP with
``n_layers`` hidden ReLU layers of ``n_hidden``) from the task model's feature to two
values per policy output, the first half levels and the second half op weights
(``MDAAug.predict_aug_params``: ``torch.split(a_params, n_ops)``). The paper's Table 5
found no hidden layer best (h = 0), its default.

**The prune** (dropaug's, not MADAug's: the probe-and-prune recipe). A pruned op leaves
the network: ``Projection.prune(cols)`` slices its two rows (its level and its weight)
out of the output layer, which becomes ``Linear(h, 2 * live)``, so the policy's trainable
parameters shrink with every prune. It can be called again with more columns (an online
prune grows); a column once pruned stays pruned. The rule every host with a no-op
outcome keeps is that a pruned op's mass goes to the surviving ops in proportion to their
share and the no-op share (here p(Identity)) stays where it was. In a policy that keeps
learning, that takes two pieces:

- The pruned ops' weight logits stay in the softmax as a constant: their rows, as the
  network had them at the prune, are kept as a frozen buffer (no gradient, no Adam, no
  weight decay) and evaluated under no_grad. Dropped outright, the softmax would hand
  Identity part of their mass; left live, they would be a knob on p(Identity) that nothing
  optimizes. Their level rows are dropped (a pruned op is never drawn); ``forward`` gives
  a pruned op's level logit as 0.
- ``predict`` applies the rule: the pruned columns of p are 0, p(Identity) is the full
  softmax's (frozen logits included), and the survivors are rescaled to 1 - p(Identity),
  differentiably.

``forward`` scatters the live outputs and the frozen logits back into MADAug's 2 x 17
layout, so everything downstream (``predict``, ``relax.relaxed_batch(skip=)``,
``search.exploit``, the host's draw) is unchanged. Linear policies only: with hidden
layers, the frozen logits would need the hidden layers as they were at each prune, so
``prune`` refuses them (``dropaug.config.check`` refuses the config first).
``param_counts`` gives (trainable, frozen).
"""

import contextlib
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from hosts.madaug import IDENTITY, OPS_NAMES

__all__ = ["Projection", "eval_mode", "policy_levels", "predict"]


class Projection(nn.Module):
    """MADAug's policy: features (B, in_features) -> (B, 2 * n_outputs)."""

    def __init__(self, in_features: int, n_layers: int = 0, n_hidden: int = 128, n_outputs: int = len(OPS_NAMES)):
        super().__init__()
        self.n_outputs = int(n_outputs)
        self.n_layers = int(n_layers)
        if n_layers > 0:
            layers = [nn.Linear(in_features, n_hidden), nn.ReLU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]
            layers.append(nn.Linear(n_hidden, 2 * self.n_outputs))
        else:
            layers = [nn.Linear(in_features, 2 * self.n_outputs)]
        self.projection = nn.Sequential(*layers)
        #: The pruned outputs (MADAug's indices), sorted; the frozen buffer's rows follow
        #: this order and the output layer's live rows the order of the rest.
        self.frozen_outputs: Tuple[int, ...] = ()

    @property
    def live_outputs(self) -> Tuple[int, ...]:
        """The outputs the output layer still computes, in order."""
        frozen = set(self.frozen_outputs)
        return tuple(c for c in range(self.n_outputs) if c not in frozen)

    def param_counts(self) -> Tuple[int, int]:
        """``(trainable, frozen)``: the network's parameters, and the pruned weight
        logits it keeps as a buffer."""
        trainable = sum(p.numel() for p in self.parameters())
        frozen = sum(b.numel() for name, b in self.named_buffers() if name.startswith("frozen_"))
        return trainable, frozen

    # --- the prune ----------------------------------------------------------------------

    def prune(self, cols: Iterable[int], optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, ...]:
        """Prune outputs ``cols`` (MADAug's indices; the ones pruned already may be
        included, Identity may not) from the network: their rows leave the output layer,
        their weight-logit rows go to the frozen buffer as they are now (module
        docstring). With ``optimizer`` (the policy's), its state follows: the kept rows
        keep their moments, the pruned rows' go. Returns the newly pruned outputs."""
        cols = {int(c) for c in cols}
        if any(not 0 <= c < self.n_outputs for c in cols):
            raise ValueError(f"policy outputs {sorted(cols)} are not all in [0, {self.n_outputs})")
        if self.n_outputs == len(OPS_NAMES) and OPS_NAMES.index(IDENTITY) in cols:
            raise ValueError("Identity is the no-op; it can't be pruned")
        added = tuple(sorted(cols - set(self.frozen_outputs)))
        if not added:
            return ()
        if self.n_layers > 0:
            raise ValueError(
                "a pruned policy must be linear (n_layers=0): with hidden layers its frozen logits "
                "would need the hidden layers as they were at the prune"
            )
        live = self.live_outputs
        n_live = len(live)
        keep = [i for i, c in enumerate(live) if c not in added]
        gone = [i for i, c in enumerate(live) if c in added]
        rows = keep + [n_live + i for i in keep]
        old = self.projection[-1]
        new = nn.Linear(old.in_features, 2 * len(keep)).to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(old.weight[rows])
            new.bias.copy_(old.bias[rows])
            weight = old.weight[[n_live + i for i in gone]].detach().clone()
            bias = old.bias[[n_live + i for i in gone]].detach().clone()
            if self.frozen_outputs:
                weight = torch.cat([self.frozen_weight, weight])
                bias = torch.cat([self.frozen_bias, bias])
        frozen = self.frozen_outputs + added
        order = sorted(range(len(frozen)), key=lambda i: frozen[i])
        self.register_buffer("frozen_weight", weight[order].contiguous())
        self.register_buffer("frozen_bias", bias[order].contiguous())
        self.frozen_outputs = tuple(frozen[i] for i in order)
        self.projection[-1] = new
        if optimizer is not None:
            _move_state(optimizer, (old.weight, new.weight, rows), (old.bias, new.bias, rows))
        return added

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.projection(x)
        if not self.frozen_outputs:
            return out
        live = list(self.live_outputs)
        n_live = len(live)
        full = out.new_zeros(out.shape[0], 2 * self.n_outputs)
        full[:, live] = out[:, :n_live]
        full[:, [self.n_outputs + c for c in live]] = out[:, n_live:]
        with torch.no_grad():
            frozen = nn.functional.linear(x.detach(), self.frozen_weight, self.frozen_bias)
        full[:, [self.n_outputs + c for c in self.frozen_outputs]] = frozen.to(out.dtype)
        return full


def _move_state(optimizer: torch.optim.Optimizer, *moves) -> None:
    """Each ``(old, new, rows)``: ``new`` takes ``old``'s place in ``optimizer``, and
    its per-element state (Adam's moments) is ``old``'s at ``rows``. Scalar state (the
    step count) carries over as it is."""
    for old, new, rows in moves:
        for group in optimizer.param_groups:
            group["params"] = [new if p is old else p for p in group["params"]]
        state = optimizer.state.pop(old, None)
        if state:
            optimizer.state[new] = {
                key: (value[rows].clone() if torch.is_tensor(value) and value.dim() > 0 and value.shape == old.shape else value)
                for key, value in state.items()
            }


def predict(
    policy: Projection, features: torch.Tensor, temperature: float = 1.0, *, unpruned: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(levels, p)``, each (B, n_outputs): levels = sigmoid, p = softmax(· / T).
    MADAug samples training draws at its temperature and relaxes the policy update at
    T = 1 (``predict_aug_params``' exploit / explore).

    With frozen outputs (a prune, module docstring), p's pruned columns are 0, p(Identity)
    is the full softmax's, and the other columns are rescaled to 1 - p(Identity).
    ``unpruned=True`` returns the full softmax instead: the policy as it would draw
    without the prune, for the per-batch check."""
    levels, weights = torch.split(policy(features), policy.n_outputs, dim=1)
    p = torch.softmax(weights / float(temperature), dim=-1)
    if policy.frozen_outputs and not unpruned:
        p = _hold_identity(p, policy.frozen_outputs)
    return torch.sigmoid(levels), p


def _hold_identity(p: torch.Tensor, pruned: Sequence[int]) -> torch.Tensor:
    """p with the ``pruned`` columns 0, Identity's column as it is, and the rest rescaled
    to 1 - p(Identity): the pruned mass goes to the surviving ops by their share."""
    if p.shape[-1] != len(OPS_NAMES):
        raise ValueError(f"the prune rule needs MADAug's {len(OPS_NAMES)} outputs (Identity among them), got {p.shape[-1]}")
    ident = OPS_NAMES.index(IDENTITY)
    if ident in pruned:
        raise ValueError("Identity is the no-op; it can't be pruned")
    keep = torch.ones(p.shape[-1], dtype=p.dtype, device=p.device)
    keep[list(pruned)] = 0.0
    keep[ident] = 0.0
    ops = p * keep
    scale = (1.0 - p[:, ident : ident + 1]) / ops.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    is_ident = torch.zeros_like(keep, dtype=torch.bool)
    is_ident[ident] = True
    return torch.where(is_ident, p, ops * scale)


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
