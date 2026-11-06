"""Adversarial AutoAugment's controller update, REINFORCE on the target's training loss.

§3.3: the policy network maximizes the target network's training loss,
``J(theta) = E_x E_{tau ~ A(theta)} L[F(tau(x), w), y]`` (Eq. 7), estimated over the
epoch's M policies as ``grad J ~ (1/M) sum_m L_m grad log p_m`` (Eq. 8). "To reduce the
variance ..., we replace the training loss of a mini-batch L_m with L^_m a moving
average over a certain mini-batches [footnote 2: the length of the moving average is
fixed to an epoch], and then normalize it among M instances as L~_m", and the update is
``theta <- theta + beta (1/M) sum_m L~_m grad log p_m`` (Eq. 9), with Adam at an initial
learning rate of 0.00035 and "an entropy penalty of a weight of 0.00001 ... to avoid
unexpected rapid convergence" (§4.1).

Algorithm 1 resets L^_m to 0 at each epoch's start and updates it every step of that
epoch, so L^_m is the mean of policy m's per-step losses over the epoch (``EpochMeans``).

Two choices the paper leaves open (flagged in CLAUDE.md):

- **The normalization** is a z-score across the M policies, ``(L^ - mean) / (std + eps)``
  (population std). It centres the reward, which is the baseline REINFORCE needs here.
- **The entropy term** is a bonus on the maximized objective, ``J + 1e-5 H`` (ENAS's
  and AutoAugment's sign), since it is there to slow convergence.

So the loss minimized is ``-(1/M) sum_m L~_m log p_m - 1e-5 (1/M) sum_m H_m``.
"""

from typing import Optional

import torch

Tensor = torch.Tensor

#: The paper's controller learning rate and entropy weight (§4.1).
LR = 0.00035
ENTROPY_WEIGHT = 0.00001


def normalize(losses: Tensor, eps: float = 1e-8) -> Tensor:
    """L~_m: the epoch means z-scored across the M policies."""
    losses = losses.float()
    return (losses - losses.mean()) / (losses.std(unbiased=False) + eps)


def controller_loss(log_prob: Tensor, entropy: Tensor, rewards: Tensor, entropy_weight: float = ENTROPY_WEIGHT) -> Tensor:
    """The loss whose gradient is minus Eq. 9's ascent direction, entropy bonus included."""
    rewards = rewards.detach().to(log_prob.dtype)
    return -(rewards * log_prob).mean() - float(entropy_weight) * entropy.mean()


class EpochMeans:
    """L^_m: each policy's training loss, averaged over the epoch's steps (Algorithm 1,
    lines 2 and 7). ``sums`` and ``count`` are plain tensors, for all-reducing across
    ranks and for checkpoints."""

    def __init__(self, num_policies: int, device: Optional[torch.device] = None):
        self.sums = torch.zeros(int(num_policies), dtype=torch.float64, device=device)
        self.count = torch.zeros((), dtype=torch.float64, device=device)

    def add(self, losses: Tensor) -> None:
        self.sums += losses.detach().to(self.sums)
        self.count += 1

    def means(self) -> Tensor:
        if float(self.count) <= 0:
            raise RuntimeError("no step was recorded this epoch")
        return self.sums / self.count

    def reset(self) -> None:
        self.sums.zero_()
        self.count.zero_()
