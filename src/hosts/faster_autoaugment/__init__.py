"""Faster AutoAugment (Hataya et al., ECCV 2020): its host, and its policy search.

``host.py`` is the host (numpy, like every host), and the only module this package
imports. ``policy.py`` and ``search.py`` are the search (torch): the policy and the
WGAN-GP loop that trains it, run before training by ``scripts/faa_search.py``. They are
the one place under ``hosts/`` with torch, and they are imported only by the search.
"""

from .host import FasterAutoAugmentHost, initial_policy, load_policy

__all__ = ["FasterAutoAugmentHost", "initial_policy", "load_policy"]
