"""MADAug's policy, with no training framework (Hou, Zhang & Zhou, ICCV 2023).

"When to Learn What: Model-Adaptive Data Augmentation Curriculum", arXiv 2309.04747,
code github.com/JackHck/MADAug. A policy network reads each training image's feature
and picks, per image, which ops to apply and how strongly. It is trained online,
alternating with the task model, to minimize the task model's validation loss after a
virtual step on the policy's augmentations (a bi-level problem, §3.3). These are the
parts of that which are math, in torch:

    policy.py     Projection (MADAug's policy net), predict (levels, p), eval_mode,
                  policy_levels (the policy on every probe image, for the probe)
    relax.py      the relaxed batch the policy update differentiates (explore), and its
                  soft-label cross-entropy
    hypergrad.py  the policy's gradient through one virtual SGD step: exact
                  (Hessian-vector product) or finite differences (DARTS)
    search.py     policy_step (the update) and exploit (the training draw)

The sampling itself (the curriculum, k ops from p, δ, signs, q, the level tables) is
the host's, ``hosts/madaug.py``, pure numpy. The loop that alternates the two updates
is glue: ``tasks/mmseg/madaug.py``.

What this ports, and what it doesn't. The paper, not its released script, is followed
where they differ on whether the policy learns at all: ``main.sh`` runs
``main_higher.py`` without ``--bi_epochs`` (default 300) for 240 or 200 epochs, and the
update is gated on ``epoch > bi_epochs``, so the released runs never update the policy.
Here the policy updates every ``search_freq`` iterations from the first. Other
differences, each documented where it lives:

- the relaxation blends single ops (the code), not pairs (the paper's Eq. 6)
- the virtual step is plain SGD at the task's learning rate (Eq. 5), not ``higher``
  around the task optimizer
- segmentation: the policy reads the backbone's last stage, average-pooled, for
  MADAug's penultimate classifier feature; the relaxed label is blended too
- the policy reads the current model's features; the code reads a ``deepcopy`` of the
  model refreshed only at each policy update
- the curriculum draws each image (the paper's §3.1), not the first rows of the batch
  with a +0.01 offset (the code)
- MADAug's Cutout(16) base augmentation after the policy (``--cutout``) is the
  dataset's training pipeline here, which has none
"""

from .hypergrad import MODES, policy_gradient
from .policy import Projection, eval_mode, policy_levels, predict
from .relax import relaxed_batch, soft_cross_entropy
from .search import exploit, policy_step

__all__ = [
    "MODES",
    "Projection",
    "eval_mode",
    "exploit",
    "policy_gradient",
    "policy_levels",
    "policy_step",
    "predict",
    "relaxed_batch",
    "soft_cross_entropy",
]
