"""Adversarial AutoAugment's controller, with no training framework (Zhang, Wang, Zhang &
Zhong, ICLR 2020, arXiv 1912.11188; no code was released).

The target network and an augmentation policy network are trained jointly and
adversarially: the target minimizes its training loss on augmented data, and the policy
network (an LSTM controller) maximizes it, by REINFORCE (§3.3, Algorithm 1). Each epoch
the controller samples M policies; every training image is repeated M times, copy m
augmented by policy m; each policy's reward is its copy's training loss averaged over
the epoch and normalized across the M; and the controller takes one Adam step. These are
the parts of that which are math, in torch:

    controller.py  Controller: the 1-layer LSTM (hidden 100, embedding 32) that emits a
                   policy's 20 parameters, with sample and log_prob_entropy
    reinforce.py   normalize (L~_m), controller_loss (Eq. 9 with the entropy bonus),
                   EpochMeans (L^_m)
    reward.py      copy_losses (Eq. 5's L_m, from the step's own logits)

The sampling (the policies' ops and magnitudes, the copies, q, the prune) is the host's,
``hosts/adversarial_autoaugment.py``, pure numpy. The loop is glue:
``tasks/mmseg/advaa_hooks.py``.
"""

from .controller import Controller
from .reinforce import ENTROPY_WEIGHT, LR, EpochMeans, controller_loss, normalize
from .reward import copy_losses

__all__ = ["Controller", "ENTROPY_WEIGHT", "EpochMeans", "LR", "controller_loss", "copy_losses", "normalize"]
