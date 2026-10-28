"""MADAug (Hou, Zhang & Zhou, ICCV 2023) as a host: the numpy half of its policy.

"When to Learn What: Model-Adaptive Data Augmentation Curriculum" (arXiv 2309.04747,
code github.com/JackHck/MADAug). A policy network reads each training image's feature
and gives, per image, a probability p over 17 entries (16 ops and Identity) and a level
λ in [0, 1] for each. The network is torch and lives in ``src/madaug/``; the loop that
trains it is ``tasks/mmseg/madaug.py``. This host holds what is left, the sampling:

- **the curriculum** (paper §3.1, Eq. 1): an image is augmented with probability
  ``p(t) = tanh(t / τ)``, else left alone. t is the training step as a fraction of
  ``max_iters`` (the host is told the step by ``HostProgressHook``, ``set_step``), and
  ``tau`` is τ in the same units: MADAug's best τ = 40 epochs of its 240 (Table 5) is
  ``40 / 240``. The released code augments the first ``int(rate * B)`` rows of each batch
  with ``rate = min(tanh(epoch / 40) + 0.01, 1)``; at a segmentation batch of 2 that rounds
  to 0, 1 or 2 images, so each image is drawn instead, as the paper describes it
- **the draw** (``draw``, MADAug's ``get_training_aug_images``): ``k_ops`` (2) entries
  drawn from the image's p without replacement (``torch.multinomial`` there), applied in
  the order drawn; Identity applies nothing. Each op's λ is perturbed by up to ``delta``
  (0.3, ``perturb_param``), mapped to ``low + λ (high - low)`` in MADAug's ranges
  (``RANGES``, its ``AUGMENT_LIST``), and for the 5 geometric ops negated with p = 0.5
  (``random_mirror``), then put in the vocab op's units (``level_to_magnitude``)

The vocabulary is configs/vocab/madaug.yaml: the 16 ops, Identity out (the no-op stays
out of every host's vocabulary). Hardcoded here so the host stays free of torch;
tests/hosts would pin the two (no test yet).

**base_probs and q.** MADAug has no fixed distribution: p is per image and moves with the
model. ``observe(p)`` keeps an exponential average of the policy's mean p over training
batches (``prior_momentum``), and ``base_probs()`` is its 16 ops, renormalized (uniform
before the first batch). That is the prior a selector tilts against. A q is held as the
ratio ``q / base`` at the time it was set, and each image's p over the 16 ops is
multiplied by it and renormalized to ``1 - p(Identity)``: q changes which op an image
draws, never how likely it is to draw Identity, nor the curriculum. A zero in q (or a
static prune) is never drawn. ``sampling_probs()`` is base times that ratio, the mean
of what is drawn. The policy itself is trained on its own p, never the tilted one.

``none_prob`` is ``1 - p(t)``, the share of images the curriculum leaves alone.

Pure numpy, like every host.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import AugHost
from .sensaug import NONE_KEY

__all__ = [
    "IDENTITY",
    "MADAugHost",
    "OPS_NAMES",
    "RANGES",
    "SIGNED",
    "VOCAB_NAMES",
    "level_to_magnitude",
    "perturb_level",
]

#: MADAug's ``config.OPS_NAMES``: the policy's 17 outputs, in its order.
OPS_NAMES = (
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Rotate",
    "AutoContrast",
    "Invert",
    "Equalize",
    "Solarize",
    "Posterize",
    "Contrast",
    "Color",
    "Brightness",
    "Sharpness",
    "Cutout",
    "Flip",
    "Identity",
)
IDENTITY = "Identity"

#: Each MADAug op's vocab op (configs/vocab/madaug.yaml).
VOCAB_NAMES: Dict[str, str] = {
    **{
        op: f"randaug_default.{op}"
        for op in OPS_NAMES
        if op not in ("Invert", "Cutout", "Flip", IDENTITY)
    },
    "Invert": "autoaugment.Invert",
    "Cutout": "madaug.Cutout",
    "Flip": "madaug.Flip",
}

#: MADAug's ``AUGMENT_LIST``: (low, high) of each op's argument, None for an op that
#: ignores it. Translate is a fraction of the image's extent, Cutout of its width.
RANGES: Dict[str, Optional[Tuple[float, float]]] = {
    "ShearX": (-0.3, 0.3),
    "ShearY": (-0.3, 0.3),
    "TranslateX": (-0.45, 0.45),
    "TranslateY": (-0.45, 0.45),
    "Rotate": (-30.0, 30.0),
    "AutoContrast": None,
    "Invert": None,
    "Equalize": None,
    "Solarize": (0.0, 256.0),
    "Posterize": (4.0, 8.0),
    "Contrast": (0.1, 1.9),
    "Color": (0.1, 1.9),
    "Brightness": (0.1, 1.9),
    "Sharpness": (0.1, 1.9),
    "Cutout": (0.0, 0.2),
    "Flip": None,
    IDENTITY: None,
}

#: The ops MADAug negates with p = 0.5 (``random_mirror``).
SIGNED = frozenset({"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"})

#: randaug_default states translate in pixels at 331x331 (AutoAugment's Table 6) and
#: scales it to the image; MADAug states it as a fraction of the image. 0.45 is 148.95.
#: The sign flips too: PIL's affine data maps output to input, so MADAug's +v moves the
#: image by -v, where torchvision's translate moves it by +v. (MADAug mirrors the sign at
#: random anyway; this keeps one level the same picture.)
_TRANSLATE_REF = -331.0

#: What an image left alone draws: tasks.augment's NO_OP.
NO_OP = NONE_KEY[0]

#: One slot of a batch's draw: per image, ``(vocab op name or NO_OP, magnitude)``.
Slot = List[Tuple[str, float]]


def perturb_level(level: np.ndarray, delta: float, rng: np.random.Generator) -> np.ndarray:
    """MADAug's ``perturb_param``, per element: an amount uniform on [0, delta], taken
    off (clipped at 0) or added (clipped at 1) with p = 0.5. ``delta <= 0`` returns it
    unchanged."""
    level = np.asarray(level, dtype=np.float64)
    if delta <= 0:
        return level
    amount = rng.uniform(0.0, delta, size=level.shape)
    down = rng.random(size=level.shape) < 0.5
    return np.where(down, np.maximum(0.0, level - amount), np.minimum(1.0, level + amount))


def level_to_magnitude(op: str, level, negate=False) -> np.ndarray:
    """MADAug op ``op`` at ``level`` in [0, 1], as the vocab op's magnitude:
    ``low + level (high - low)`` in MADAug's range, negated where ``negate`` (a signed
    op only), then in the vocab op's units, so the vocab op gives the image MADAug's PIL
    op does (up to 8-bit rounding: see CLAUDE.md). 0 for an op with no magnitude."""
    level = np.asarray(level, dtype=np.float64)
    if RANGES[op] is None:
        return np.zeros_like(level)
    low, high = RANGES[op]
    value = low + level * (high - low)
    if op in SIGNED:
        value = np.where(negate, -value, value)
    if op in ("TranslateX", "TranslateY"):
        value = value * _TRANSLATE_REF
    return value


class MADAugHost(AugHost):
    """MADAug's sampling over its 16 ops (module docstring)."""

    name = "madaug"

    #: The policy's output width: the 16 ops and Identity.
    num_policy_outputs = len(OPS_NAMES)

    def __init__(
        self,
        *,
        max_iters: int,
        k_ops: int = 2,
        delta: float = 0.3,
        temperature: float = 3.0,
        tau: float = 40.0 / 240.0,
        prior_momentum: float = 0.99,
        probe_interval: Optional[int] = None,
        probe_iters: Optional[Iterable[int]] = None,
        control_iters: Iterable[int] = (),
    ):
        if int(max_iters) < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        if not 1 <= int(k_ops) <= len(OPS_NAMES):
            raise ValueError(f"k_ops must be in [1, {len(OPS_NAMES)}], got {k_ops}")
        if not 0.0 <= float(delta) <= 1.0:
            raise ValueError(f"delta must be in [0, 1], got {delta}")
        if float(temperature) <= 0 or float(tau) <= 0:
            raise ValueError(f"temperature and tau must be positive, got {temperature}, {tau}")
        if not 0.0 <= float(prior_momentum) < 1.0:
            raise ValueError(f"prior_momentum must be in [0, 1), got {prior_momentum}")
        self.max_iters = int(max_iters)
        self.k_ops = int(k_ops)
        self.delta = float(delta)
        self.temperature = float(temperature)
        self.tau = float(tau)
        self.prior_momentum = float(prior_momentum)
        self._step = 1

        ops = [op for op in OPS_NAMES if op != IDENTITY]
        self._ops = tuple(ops)
        names = [VOCAB_NAMES[op] for op in ops]
        super().__init__(
            names,
            np.full(len(names), 1.0 / len(names)),
            none_prob=0.0,
            probe_interval=probe_interval,
            probe_iters=probe_iters,
            control_iters=control_iters,
        )
        self._tilt = np.ones(len(names))
        self._mean_p: Optional[np.ndarray] = None

    # --- the curriculum -----------------------------------------------------------

    def set_step(self, step: int) -> None:
        """The 1-based training step about to run (``runner.iter + 1`` in
        ``before_train_iter``): sets p(t) for the draws that follow."""
        if int(step) < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        self._step = int(step)

    @property
    def augment_prob(self) -> float:
        """p(t) = tanh(t / tau), t the step as a fraction of training."""
        return float(np.tanh((self._step / self.max_iters) / self.tau))

    @property
    def none_prob(self) -> float:
        """1 - p(t): the share of images the curriculum leaves alone. Never moved by q."""
        return 1.0 - self.augment_prob

    # --- the prior and q -----------------------------------------------------------

    def observe(self, mean_p: Sequence[float]) -> None:
        """Fold one training batch's mean policy p (17 entries, MADAug's order) into
        the running prior ``base_probs()``."""
        mean_p = np.asarray(mean_p, dtype=np.float64)
        if mean_p.shape != (len(OPS_NAMES),) or not np.all(np.isfinite(mean_p)):
            raise ValueError(f"mean p must be {len(OPS_NAMES)} finite values, got {mean_p}")
        m = self.prior_momentum
        self._mean_p = mean_p if self._mean_p is None else m * self._mean_p + (1.0 - m) * mean_p
        base = np.array([self._mean_p[OPS_NAMES.index(op)] for op in self._ops])
        base[[n in self._pruned for n in self._names]] = 0.0
        if base.sum() > 0:
            self._base = base / base.sum()

    @property
    def identity_prob(self) -> Optional[float]:
        """The policy's running mean p(Identity), None before the first batch."""
        return None if self._mean_p is None else float(self._mean_p[OPS_NAMES.index(IDENTITY)])

    def _apply_weights(self, q: np.ndarray) -> np.ndarray:
        """Hold q as its ratio to the base it was derived from."""
        self._tilt = np.divide(q, self._base, out=np.zeros_like(q), where=self._base > 0)
        return q

    def clear_weights(self) -> None:
        """Back to the policy's own p (a static prune still holds)."""
        self._tilt = np.array([0.0 if n in self._pruned else 1.0 for n in self._names])

    def sampling_probs(self) -> np.ndarray:
        """The prior times q's ratio, renormalized: the mean of what is drawn."""
        probs = self._base * self._tilt
        total = probs.sum()
        return probs / total if total > 0 else self._base.copy()

    def prune(self, names: Iterable[str], *, hold_none_prob: bool = True) -> Tuple[str, ...]:
        """``AugHost.prune``, with the pruned ops' ratio set to 0. The no-op mass is each
        image's p(Identity), the policy's, so ``hold_none_prob=False`` has nothing to
        renormalize and raises."""
        names = tuple(names)
        if names and not hold_none_prob:
            raise ValueError(
                f"{type(self).__name__}: the no-op is the policy's per-image p(Identity), "
                f"so hold_none_prob=False has nothing to renormalize"
            )
        implied = super().prune(names, hold_none_prob=True)
        self.clear_weights()
        return implied

    # --- the draws -------------------------------------------------------------------

    def tilted(self, p: np.ndarray) -> np.ndarray:
        """Per image (rows of ``p``, 17 wide), the policy's p with q's ratio applied to
        the 16 ops and p(Identity) held: the distribution ``draw`` samples."""
        p = np.asarray(p, dtype=np.float64)
        ident = OPS_NAMES.index(IDENTITY)
        cols = [OPS_NAMES.index(op) for op in self._ops]
        ops = p[:, cols] * self._tilt
        total = ops.sum(axis=1, keepdims=True)
        keep = 1.0 - p[:, ident : ident + 1]
        ops = np.divide(ops * keep, total, out=np.zeros_like(ops), where=total > 0)
        out = np.zeros_like(p)
        out[:, cols] = ops
        out[:, ident] = p[:, ident]
        return out / out.sum(axis=1, keepdims=True)

    def draw(self, p, levels, rng: np.random.Generator) -> List[Slot]:
        """One batch's draw: ``k_ops`` slots, slot k holding each image's k-th
        ``(vocab op, magnitude)``, ``NO_OP`` where the image is left alone (the
        curriculum, Identity, or fewer than k drawable entries). ``p`` and ``levels`` are
        the policy's (B, 17) outputs for the batch, in MADAug's order."""
        p = self.tilted(p)
        levels = np.asarray(levels, dtype=np.float64)
        batch_size = p.shape[0]
        if levels.shape != p.shape:
            raise ValueError(f"levels {levels.shape} and p {p.shape} differ")
        p_aug = self.augment_prob
        slots: List[Slot] = [[(NO_OP, 0.0)] * batch_size for _ in range(self.k_ops)]
        for i in range(batch_size):
            if not rng.random() < p_aug:
                continue
            k = min(self.k_ops, int(np.count_nonzero(p[i])))
            picks = rng.choice(len(OPS_NAMES), size=k, replace=False, p=p[i])
            for slot, j in enumerate(picks):
                op = OPS_NAMES[int(j)]
                if op == IDENTITY:
                    continue
                slots[slot][i] = (VOCAB_NAMES[op], float(self._magnitude(op, levels[i, j : j + 1], rng)[0]))
        return slots

    def method_magnitudes(self, levels, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """For the probe: each op's magnitude at the policy's ``levels`` ((B, 17), one
        row per image), perturbed and signed as ``draw`` does. Ops with no magnitude are
        left out."""
        levels = np.asarray(levels, dtype=np.float64)
        return {
            VOCAB_NAMES[op]: self._magnitude(op, levels[:, OPS_NAMES.index(op)], rng)
            for op in self._ops
            if RANGES[op] is not None
        }

    def _magnitude(self, op: str, level: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        level = perturb_level(level, self.delta, rng)
        negate = rng.random(size=level.shape) < 0.5 if op in SIGNED else False
        return level_to_magnitude(op, level, negate)
