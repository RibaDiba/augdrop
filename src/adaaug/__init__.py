"""AdaAug's search, with no training framework (Cheung & Yeung, ICLR 2022).

"AdaAug: Learning Class- and Instance-adaptive Data Augmentation Policies", code
github.com/jamestszhim/adaptive_augment (MIT; the paper is only on OpenReview, so the
released code is the reference). A search model gf and a policy network h train
together: gf on images h augments (exploit), h on the loss of gf's head on the p-mixed
features of every op's copy of held-out images (explore). Then a new model trains on
the frozen h, which reads the frozen gf's features. These are the parts of that which
are math, in torch:

    explore.py   the explore loss: one copy per op, passed straight through in λ, the
                 features mixed by p, the head on the mix against the blended label
    search.py    the search loop (exploit, explore, h's Adam, gf's schedule), its saves
                 (``final/``: gf.pth, h.pth, meta.json) and its cost (``pass_counts``)

h is MADAug's ``Projection`` (MADAug's code is built on AdaAug's: the same 17 outputs,
the same network, the same straight-through estimator), and ``madaug.predict`` reads it.
The sampling (k ops from p, δ, the mirror coin, the level map onto the split vocab ops,
the prune) is the host's, ``hosts/adaaug.py``, pure numpy. The glue is
``scripts/adaaug_search.py`` (the search's model and data) and
``tasks/mmseg/adaaug_hooks.py`` (h in training).

What differs from the code, each documented where it lives: segmentation (f is the
segmentor's ``extract_feat`` and g its decode head, every level mixed; the label blended
over the copies, since geometric ops move it), each held-out image backpropagated on its
own (the same gradient), gf's optimizer the recipe's (SGD at 0.1 on CIFAR there), the
held-out images from the val split, no Cutout after the ops.
"""

from .explore import copies, explore_loss, mix_features, pooled, straight_through
from .search import SearchConfig, exploit_step, explore_step, pass_counts, policy_params, save_search, search, segmentor_logits

__all__ = [
    "SearchConfig",
    "copies",
    "exploit_step",
    "explore_loss",
    "explore_step",
    "mix_features",
    "pass_counts",
    "policy_params",
    "pooled",
    "save_search",
    "search",
    "segmentor_logits",
    "straight_through",
]
