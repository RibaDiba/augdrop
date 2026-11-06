"""DeepAA's search space as the search applies it: the 139 (op, level) transforms of the
ImageNet list, each as one vocab ``deepaa`` op at its magnitude.

The search applies a transform at its own signed value: DeepAA's search ops
(``augmentation.py``) never mirror, only the trainer does (``random_mirror``), so here a
signed transform is one op, ``_pos`` or ``_neg``. RandFlip is its coin (Flip or nothing,
half each, drawn per image); RandCutout and RandomResizedCrop draw their own randomness
inside the op (the torch RNG).

``kept_transforms`` is DeepAA's ``available_policies`` after a prune: a transform is left
out when the op it applies in the search is pruned (Identity always stays, and stays
first, where DeepAA's identity init puts its logit). Under the trainer's mirror a kept
signed transform also applies its other sign; when that sign is pruned, the host's prune
zeroes it there (``DeepAutoAugmentHost.install_policy``).
"""

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from hosts.deep_autoaugment import DEEPAA_OPS, L_MAGS, NO_OP, _value, transform_atoms, unique_policy

__all__ = ["NAMES", "NUM_TRANSFORMS", "OPS", "LEVELS", "draws", "kept_transforms", "transform_label"]

#: DeepAA's op names, in its order.
NAMES: Tuple[str, ...] = tuple(n for n, _, _ in DEEPAA_OPS)

#: (T,) each: transform t's op index and level (``_get_unique_policy``).
OPS, LEVELS = unique_policy(NAMES, L_MAGS)
NUM_TRANSFORMS = int(len(OPS))

#: Each transform's atoms in the search: ``(vocab op or NO_OP, magnitude, share)``.
ATOMS: List[list] = [
    transform_atoms(NAMES[int(op)], _value(int(op), int(level), L_MAGS), random_mirror=False)
    for op, level in zip(OPS, LEVELS)
]


def transform_label(t: int) -> str:
    """``Op@level``, for logs."""
    return f"{NAMES[int(OPS[t])]}@{int(LEVELS[t])}"


def kept_transforms(pruned: Iterable[str]) -> np.ndarray:
    """The transforms left after pruning the vocab ops ``pruned``, ascending (DeepAA's
    ``available_policies``)."""
    pruned = set(pruned)
    kept = [t for t, atoms in enumerate(ATOMS) if not any(name in pruned for name, _, _ in atoms if name != NO_OP)]
    if not kept or kept[0] != 0:
        raise ValueError("Identity must stay in the search space")
    return np.array(kept, dtype=np.int64)


def draws(ids: Sequence[int], rng: np.random.Generator) -> List[Tuple[str, float]]:
    """``(op or NO_OP, magnitude)`` per image for transform ids ``ids``, a random op's
    coin drawn from ``rng`` (``tasks.augment.apply_draws``'s input)."""
    out = []
    for t in ids:
        atoms = ATOMS[int(t)]
        if len(atoms) == 1:
            name, magnitude, _ = atoms[0]
        else:
            name, magnitude, _ = atoms[int(rng.choice(len(atoms), p=[share for _, _, share in atoms]))]
        out.append((name, float(magnitude)))
    return out
