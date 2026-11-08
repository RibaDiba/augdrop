"""Greedy AutoAugment's search (Naghizadeh, Abavisani & Metaxas, 2020; arXiv 1908.00704),
framework-free: which chains to try, in which order, and the policy they give. The
children that score them are trained by ``tasks/mmseg/greedyaa_trial.py`` through
``scripts/gaa_search.py``; the host is ``hosts/greedy_autoaugment.py``."""

from .search import GreedySearch, Trial, trial_counts

__all__ = ["GreedySearch", "Trial", "trial_counts"]
