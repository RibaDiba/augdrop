"""Adversarial AutoAugment's augmentation policy network: an LSTM controller.

§3.3 and §4.1 of the paper: "the augmentation policy network is also implemented as a RNN
... At each time step of the RNN controller, the softmax layer will predict an action
corresponding to a discrete parameter of a sub-policy, and then an embedding of the
predicted action will be fed into the next time step. In our experiments, the RNN
controller will predict 20 discrete parameters to form a whole policy." "The RNN
controller is implemented as a one-layer LSTM. We set the hidden size to 100, and the
embedding size to 32."

So one LSTM step per parameter, 20 of them: for each of the 5 sub-policies, for each of
its 2 operations, its type (a softmax over 16) and then its magnitude (a softmax over 10
bins), the order of the paper's Figure 3 and of AutoAugment's controller. What the paper
leaves open, taken from the controllers it follows (ENAS, Pham et al. 2018, which
AutoAugment's is): a learned first input; one embedding table for types and one for
magnitudes, each sampled action embedded and fed to the next step; every parameter
initialized uniform on [-0.1, 0.1]; the LSTM state starting at 0.

A ``mask`` (16, 10) of allowed (type, bin) pairs (the host's ``action_mask()``, from a
static or an online prune) removes the pruned pairs: a type with no bin left is never
drawn, and a type's disallowed bins are never drawn after it. The log-probabilities and
entropies are those of the masked distributions.

**The prune shrinks the network** (dropaug's, for the probe-and-prune recipe; the paper has
none). ``prune(types)`` removes a type every bin of which is pruned (both signs of an op,
or an unsigned op): its row leaves the type head and the type embedding, so the trainable
parameters shrink, and with the controller's optimizer its Adam state follows the kept
rows (``madaug.policy``'s rule). The type logits are scattered back into the 16 with the
removed types at the dtype's minimum, so the actions keep their indices and a removed type
is never drawn: the softmax hands its mass to the survivors by their share, the rule every
pruned network here keeps (AdvAA has no no-op share to hold). That is the masked
distribution of the unpruned controller exactly. A type with one sign pruned keeps its row,
its pruned bins masked. A prune only grows. The bin head is shared by every type, so it
keeps its 10 rows.
"""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

Tensor = torch.Tensor


class Controller(nn.Module):
    """The policy network A(., theta) (module docstring)."""

    def __init__(
        self,
        num_types: int = 16,
        num_bins: int = 10,
        num_sub_policies: int = 5,
        ops_per_sub_policy: int = 2,
        hidden: int = 100,
        embed: int = 32,
        init_range: float = 0.1,
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.num_bins = int(num_bins)
        self.num_sub_policies = int(num_sub_policies)
        self.ops_per_sub_policy = int(ops_per_sub_policy)
        self.hidden = int(hidden)
        self.start = nn.Parameter(torch.zeros(1, embed))
        self.lstm = nn.LSTMCell(embed, hidden)
        self.type_head = nn.Linear(hidden, num_types)
        self.bin_head = nn.Linear(hidden, num_bins)
        self.type_embed = nn.Embedding(num_types, embed)
        self.bin_embed = nn.Embedding(num_bins, embed)
        for param in self.parameters():
            nn.init.uniform_(param, -init_range, init_range)
        #: Each type's row in the type head and embedding, -1 for a pruned type.
        self.register_buffer("type_rows", torch.arange(self.num_types))

    @property
    def pruned_types(self) -> Tuple[int, ...]:
        """The types ``prune`` removed, ascending."""
        return tuple(int(t) for t in torch.nonzero(self.type_rows < 0).flatten().tolist())

    def param_count(self) -> int:
        """Trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def prune(self, types: Iterable[int], optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, ...]:
        """Remove ``types`` (the ones removed already may be included) from the network
        (module docstring); returns the newly removed. With ``optimizer``, its Adam state
        follows the kept rows."""
        types = {int(t) for t in types}
        if any(not 0 <= t < self.num_types for t in types):
            raise ValueError(f"types {sorted(types)} are not all in [0, {self.num_types})")
        added = tuple(sorted(types - set(self.pruned_types)))
        if not added:
            return ()
        rows = self.type_rows.tolist()
        live = [t for t in range(self.num_types) if rows[t] >= 0 and t not in added]
        if not live:
            raise ValueError("the prune removes every type; nothing would be left to sample")
        keep = [rows[t] for t in live]
        old_head, old_embed = self.type_head, self.type_embed
        head = nn.Linear(old_head.in_features, len(keep)).to(device=old_head.weight.device, dtype=old_head.weight.dtype)
        embed = nn.Embedding(len(keep), old_embed.embedding_dim).to(
            device=old_embed.weight.device, dtype=old_embed.weight.dtype
        )
        with torch.no_grad():
            head.weight.copy_(old_head.weight[keep])
            head.bias.copy_(old_head.bias[keep])
            embed.weight.copy_(old_embed.weight[keep])
        self.type_head, self.type_embed = head, embed
        new_rows = torch.full_like(self.type_rows, -1)
        new_rows[live] = torch.arange(len(live), device=self.type_rows.device)
        self.type_rows.copy_(new_rows)
        if optimizer is not None:
            _move_state(
                optimizer,
                (old_head.weight, head.weight, keep),
                (old_head.bias, head.bias, keep),
                (old_embed.weight, embed.weight, keep),
            )
        return added

    def _type_logits(self, h: Tensor) -> Tensor:
        """(B, 16): the type head's logits in the 16-type layout, a pruned type at the
        dtype's minimum."""
        logits = self.type_head(h)
        if not self.pruned_types:
            return logits
        live = torch.nonzero(self.type_rows >= 0).flatten()
        full = logits.new_full((logits.shape[0], self.num_types), torch.finfo(logits.dtype).min)
        return full.index_copy(1, live, logits)

    @property
    def num_decisions(self) -> int:
        """The parameters one policy is: 2 per operation (20 at the paper's 5 x 2)."""
        return 2 * self.num_sub_policies * self.ops_per_sub_policy

    def _masked(self, logits: Tensor, allowed: Optional[Tensor]) -> Tensor:
        if allowed is None:
            return logits
        return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)

    def _run(
        self,
        num_policies: int,
        actions: Optional[Tensor],
        mask: Optional[Tensor],
        generator: Optional[torch.Generator],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        device = self.start.device
        h = torch.zeros(num_policies, self.hidden, device=device)
        c = torch.zeros(num_policies, self.hidden, device=device)
        x = self.start.expand(num_policies, -1)
        type_ok = None if mask is None else mask.any(dim=1).to(device)
        bins_ok = None if mask is None else mask.to(device)
        if self.pruned_types:
            alive = self.type_rows >= 0
            type_ok = alive if type_ok is None else type_ok & alive
        log_prob = torch.zeros(num_policies, device=device)
        entropy = torch.zeros(num_policies, device=device)
        drawn = []
        for s in range(self.num_sub_policies):
            for j in range(self.ops_per_sub_policy):
                pair = []
                for kind in (0, 1):
                    h, c = self.lstm(x, (h, c))
                    if kind == 0:
                        allowed = None if type_ok is None else type_ok.expand(num_policies, -1)
                        logits = self._type_logits(h)
                    else:
                        allowed = None if bins_ok is None else bins_ok[pair[0]]
                        logits = self.bin_head(h)
                    dist = torch.distributions.Categorical(logits=self._masked(logits, allowed))
                    if actions is None:
                        probs = dist.probs.detach().float().cpu()
                        action = torch.multinomial(probs, 1, generator=generator).squeeze(1).to(device)
                    else:
                        action = actions[:, s, j, kind].to(device)
                    log_prob = log_prob + dist.log_prob(action)
                    entropy = entropy + dist.entropy()
                    pair.append(action)
                    x = self.type_embed(self.type_rows[action]) if kind == 0 else self.bin_embed(action)
                drawn.append(torch.stack(pair, dim=-1))
        out = torch.stack(drawn, dim=1).view(num_policies, self.num_sub_policies, self.ops_per_sub_policy, 2)
        return out, log_prob, entropy

    @torch.no_grad()
    def sample(
        self, num_policies: int, mask: Optional[Tensor] = None, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """``num_policies`` policies, (M, S, J, 2) long: ``[..., 0]`` the type, ``[..., 1]``
        the bin. Drawn on the CPU from ``generator`` (the caller's seed)."""
        actions, _, _ = self._run(int(num_policies), None, mask, generator)
        return actions

    def log_prob_entropy(self, actions: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """``(log p_m, H_m)``, (M,) each: the log-probability of each policy in ``actions``
        (its 20 parameters' sum) and its 20 steps' summed entropy, teacher-forced, with the
        graph to theta."""
        _, log_prob, entropy = self._run(int(actions.shape[0]), actions, mask, None)
        return log_prob, entropy


def _move_state(optimizer: torch.optim.Optimizer, *moves) -> None:
    """Each ``(old, new, rows)``: ``new`` takes ``old``'s place in ``optimizer``, its
    per-element state (Adam's moments) ``old``'s at ``rows``; scalar state (the step) as
    it is. ``madaug.policy._move_state``'s rule."""
    for old, new, rows in moves:
        for group in optimizer.param_groups:
            group["params"] = [new if p is old else p for p in group["params"]]
        state = optimizer.state.pop(old, None)
        if state:
            optimizer.state[new] = {
                key: (value[rows].clone() if torch.is_tensor(value) and value.dim() > 0 and value.shape == old.shape else value)
                for key, value in state.items()
            }
