"""mmengine glue for MADAug's policy. Nothing is computed here.

    MADAugPolicyHook   before every training step (after HostProgressHook has set the
                       host's step): every ``search_freq`` steps, the policy's update on the
                       step's clean batch and one val batch (``madaug.policy_step``), its
                       gradient averaged over ranks and stepped with Adam; then the
                       policy's draws for the batch (``madaug.exploit``), installed on the
                       preprocessor (``set_policy_slots``), which applies them inside
                       ``train_step``. The policy and its optimizer go into every checkpoint
                       (``madaug_policy``) and come back on resume or load.

The policy is MADAug's (``madaug.Projection``) on the segmentor's pooled deepest feature
(``backend.pooled_features``); the host is the one on the preprocessor
(``hosts.madaug.MADAugHost``). The policy's validation batches come from the val
dataloader config (the split Sensaug's SA loop scores ops on, and the one mIoU is
reported on), shuffled and endless, one per update. The update runs the segmentor in
eval, functionally (``torch.func.functional_call``) on the unwrapped module, so it
touches neither the weights nor their ``.grad`` and no DDP hook fires; each rank updates
on its own batches and the gradient is averaged, so every rank holds the same policy.
The live policy is ``runner.madaug_policy``, which ``CollectGradientHook`` reads for the
probe's magnitudes.

Defaults are MADAug's: Adam at lr 1e-3, weight decay 1e-3 (``--proj_learning_rate`` in
main.sh, ``--proj_weight_decay``), no hidden layer, an update every 3 steps (main.sh's
``--search_freq 3`` on CIFAR-10; the paper's Table 5 found every step best). The virtual
step's learning rate is the task optimizer's current one (Eq. 5's α).

Checked on stand-ins for mmengine and mmseg only (no mmengine in ``sensAug``).
"""

import copy
from typing import Optional

import numpy as np
import torch
from mmengine.dist import all_reduce, broadcast
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS

import madaug

from .backend import MMSegBackend, functional_logits, pooled_features, unwrap
from .hooks import host_of
from .transforms import _gather_labels

__all__ = ["CHECKPOINT_KEY", "MADAugPolicyHook", "policy_from_checkpoint", "probe_set_levels"]

#: Where the policy lives in a checkpoint.
CHECKPOINT_KEY = "madaug_policy"


def probe_set_levels(backend, policy, probe_set, dataset, batch_size: int = 1, num_workers: int = 0) -> np.ndarray:
    """The policy's levels on every image of ``probe_set``, in probe order, with the
    backend's model in eval: what ``tasks.magnitudes.policy_magnitudes`` reads for a
    sweep over the same probe set. Shared by CollectGradientHook and scripts/probe.py."""
    loader = probe_set.loader(dataset, batch_size, collate_fn=backend.collate, num_workers=num_workers)
    with madaug.eval_mode(backend.model):
        return madaug.policy_levels(policy, backend.features, (backend.to_unit(batch)[0] for batch in loader))


def policy_from_checkpoint(path, device=None) -> Optional[madaug.Projection]:
    """The MADAug policy saved in an mmengine checkpoint (``CHECKPOINT_KEY``, written by
    MADAugPolicyHook), or None when it holds none (a run of another method)."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = checkpoint.get(CHECKPOINT_KEY) if isinstance(checkpoint, dict) else None
    if state is None:
        return None
    weights = state["state_dict"]
    first = min((k for k in weights if k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    last = max((k for k in weights if k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    policy = madaug.Projection(
        int(weights[first].shape[1]),
        n_layers=int(state["n_layers"]),
        n_hidden=int(state["n_hidden"]),
        n_outputs=int(weights[last].shape[0]) // 2,
    )
    policy.load_state_dict(weights)
    return policy.to(device) if device is not None else policy


def _rng() -> np.random.Generator:
    """A generator for one step's draws, seeded from numpy's global RNG, so the
    framework's seeding governs it (as ``tasks.augment`` seeds its hosts' draws)."""
    return np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))


@HOOKS.register_module()
class MADAugPolicyHook(Hook):
    """MADAug's policy, trained and applied alongside the segmentor (module docstring).

    search_freq   a policy update every this many steps (the first step included)
    lr, weight_decay
                  the policy's Adam
    n_layers, n_hidden
                  the policy's hidden layers (MADAug's ``--n_proj_layer``; 0 = linear)
    hypergrad     ``"exact"`` or ``"finite_diff"`` (``madaug.hypergrad``)
    fd_eps        finite_diff's ε numerator (DARTS' 0.01)
    split         the dataloader config the policy's validation batches come from
    """

    priority = "ABOVE_NORMAL"

    def __init__(
        self,
        search_freq: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        n_layers: int = 0,
        n_hidden: int = 128,
        hypergrad: str = "exact",
        fd_eps: float = 0.01,
        split: str = "val",
    ):
        if int(search_freq) < 1:
            raise ValueError(f"search_freq must be >= 1, got {search_freq}")
        if hypergrad not in madaug.MODES:
            raise ValueError(f"hypergrad {hypergrad!r} is not one of {madaug.MODES}")
        self.search_freq = int(search_freq)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_layers = int(n_layers)
        self.n_hidden = int(n_hidden)
        self.hypergrad = hypergrad
        self.fd_eps = float(fd_eps)
        self.split = split
        self.policy: Optional[madaug.Projection] = None

    # --- building and checkpoints -----------------------------------------------------

    def before_run(self, runner) -> None:
        """Build the policy before ``load_or_resume``, so a checkpoint can fill it."""
        self.host = host_of(runner)
        if not hasattr(self.host, "draw") or not hasattr(self.host, "observe"):
            raise TypeError(f"host {self.host.name!r} is not a policy host; MADAugPolicyHook drives madaug")
        model = unwrap(runner.model)
        self.dp = model.data_preprocessor
        if not getattr(self.dp, "augment", True):
            raise ValueError("augment=False trains clean; MADAugPolicyHook would train a policy nothing uses")
        device = next(model.parameters()).device
        size = tuple(getattr(self.dp, "size", None) or (512, 512))
        with torch.no_grad(), madaug.eval_mode(model):
            width = pooled_features(model, torch.zeros(1, 3, *size, device=device)).shape[1]
        self.policy = madaug.Projection(
            width, n_layers=self.n_layers, n_hidden=self.n_hidden, n_outputs=self.host.num_policy_outputs
        ).to(device)
        for param in self.policy.parameters():
            broadcast(param.data, src=0)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay
        )
        self.backend = MMSegBackend.from_runner(runner, split=self.split)
        self._val_iter = None
        runner.madaug_policy = self.policy
        print_log(
            f"[madaug] policy {width} -> {2 * self.host.num_policy_outputs} ({self.n_layers} hidden), "
            f"update every {self.search_freq} step(s), {self.hypergrad} hypergradient, "
            f"validation batches from the {self.split} dataloader",
            logger="current",
        )

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        checkpoint[CHECKPOINT_KEY] = {
            "state_dict": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "n_layers": self.n_layers,
            "n_hidden": self.n_hidden,
        }

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        state = checkpoint.get(CHECKPOINT_KEY)
        if state is None:
            print_log("[madaug] the checkpoint holds no policy; it starts fresh", logger="current", level=30)
            return
        self.policy.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])

    # --- every step ----------------------------------------------------------------------

    def _val_batch(self, runner):
        if self._val_iter is None:
            cfg = copy.deepcopy(runner.cfg.get(f"{self.split}_dataloader"))
            cfg["sampler"] = dict(type="InfiniteSampler", shuffle=True)
            cfg["persistent_workers"] = False
            loader = runner.build_dataloader(cfg, seed=runner.seed)
            self._val_iter = iter(loader)
        return next(self._val_iter)

    def before_train_iter(self, runner, batch_idx, data_batch=None) -> None:
        model = unwrap(runner.model)
        with self.dp.suspended():
            data = self.dp(data_batch, training=True)
        inputs = data["inputs"]
        images01 = ((inputs * self.dp.std.view(1, -1, 1, 1) + self.dp.mean.view(1, -1, 1, 1)) / 255.0).clamp(0.0, 1.0)
        rng = _rng()

        if runner.iter % self.search_freq == 0:
            labels = _gather_labels(data.get("data_samples") or [])
            val_batch = self._val_batch(runner)
            lr = float(next(iter(runner.optim_wrapper.get_lr().values()))[0])
            with madaug.eval_mode(model):
                grads, stats = madaug.policy_step(
                    model=model,
                    policy=self.policy,
                    images01=images01,
                    label=labels,
                    ops=self.dp.ops,
                    normalize=self.backend._normalize,
                    features=lambda x: pooled_features(model, x),
                    logits=lambda weights, x: functional_logits(model, weights, x, labels.shape[-2:]),
                    val_loss=lambda weights: self.backend.functional_loss(weights, val_batch),
                    lr=lr,
                    rng=rng,
                    mode=self.hypergrad,
                    eps=self.fd_eps,
                )
            for param, grad in zip(self.policy.parameters(), grads):
                all_reduce(grad, op="mean")
                param.grad = grad.detach()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            for key, value in stats.items():
                runner.message_hub.update_scalar(f"train/madaug_{key}", value)

        with madaug.eval_mode(model):
            slots = madaug.exploit(self.policy, lambda x: pooled_features(model, x), inputs, self.host, rng)
        self.dp.set_policy_slots(slots)
        runner.message_hub.update_scalar("train/madaug_augment_prob", self.host.augment_prob)
