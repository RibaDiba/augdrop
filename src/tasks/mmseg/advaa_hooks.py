"""mmengine glue for Adversarial AutoAugment's controller. Nothing is computed here.

    AdvAAControllerHook  before training: the controller (``advaa.Controller``) and its
                         Adam, the first epoch's M policies sampled on rank 0, broadcast,
                         and installed on the host (``host.set_policies``). After every
                         step: each copy's training loss L_m (``advaa.copy_losses``, the
                         decode head's own loss on its slice of the step's logits),
                         accumulated over the epoch (``advaa.EpochMeans``). At every epoch's
                         end: the means all-reduced over ranks, normalized
                         (``advaa.normalize``), one controller step
                         (``advaa.controller_loss``), and the next epoch's policies sampled
                         and installed (Algorithm 1). The controller, its optimizer, the
                         installed policies and the epoch's sums go into every checkpoint
                         (``advaa_controller``) and come back on resume or load.

The batch is the preprocessor's: M copies of each image, copy-major
(``GpuAugSegDataPreProcessor``, ``num_copies``), augmented by the host's ``draw_stages``.
L_m is read without another forward pass: a forward hook on the decode head keeps the
training step's logits (detached), and the preprocessor keeps the step's augmented data
samples (``last_train_samples``); copy m's loss is the decode head's ``loss_by_feat`` on
its slice, the sum of its ``loss`` terms, so the same loss and reduction the target trains
on. An auxiliary head's loss is not in L_m.

An "epoch" is ``epoch_iters`` steps from the epoch's first step (an online prune starts
a new one): by default one pass over the training set,
``ceil(len(dataset) / (batch size x world size))``, as Algorithm 1 has it (mmseg trains by
iterations). The controller samples on rank 0 from a generator seeded off numpy's global
RNG (so the framework's seed governs it), and every rank installs the broadcast policies;
every rank then takes the same step on the same all-reduced rewards, and rank 0's weights
are broadcast after it, so every rank holds one controller.

A prune shrinks the controller (``advaa.Controller.prune``: a type every bin of which is
pruned leaves its type head and embedding; one with a sign pruned has those bins masked)
and masks it (``host.action_mask``), so it never samples a pruned (type, bin). Two prunes
reach it. The static prune (the ``prune:`` block) at its build, and selector nms's latched
M (PruneHook publishes it as ``runner.latched_prune``) before the first step after each
probe that grows it, the probe-and-prune recipe's online kind (the user's choice,
2026-11-05: probe and prune every max_iters/8). At that step the epoch so far is closed
early: if it has seen a step, the controller takes its REINFORCE step on the partial
epoch's means, under the old mask (the actions it sampled keep finite log-probabilities);
then the prune, M new policies over what is left, and a new epoch's clock from that step.
Loaded policies that the run's prune forbids are resampled; a checkpoint's controller
comes back pruned as it was saved (``pruned_types``), then pruned to the run.

Every epoch's policies, the copies' mean losses and the rewards are appended to
``<work_dir>/advaa_policies.jsonl`` (rank 0), the record of the dynamic policy the
paper's Figure 6 plots.

Defaults are the paper's (§4.1): hidden 100, embedding 32, Adam at lr 0.00035, entropy
weight 0.00001.

Checked on stand-ins for mmengine and mmseg only (no mmengine in ``sensAug``).
"""

import json
import math
import os
from typing import Optional

import numpy as np
import torch
from mmengine.dist import all_reduce, broadcast, get_world_size, is_main_process
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS

import advaa
from hosts.adversarial_autoaugment import NUM_BINS, TYPES

from .backend import unwrap
from .hooks import host_of

__all__ = ["AdvAAControllerHook", "CHECKPOINT_KEY", "decode_loss"]

#: Where the controller lives in a checkpoint.
CHECKPOINT_KEY = "advaa_controller"

#: The policy record, in the work dir.
POLICY_LOG = "advaa_policies.jsonl"


def decode_loss(head, logits, samples) -> torch.Tensor:
    """The decode head's training loss on ``logits`` against ``samples``: the sum of
    ``loss_by_feat``'s ``loss`` terms (mmengine's ``parse_losses`` rule)."""
    total = 0.0
    for key, value in head.loss_by_feat(logits, samples).items():
        if "loss" not in key:
            continue
        total = total + (sum(v.mean() for v in value) if isinstance(value, (list, tuple)) else value.mean())
    return torch.as_tensor(total)


@HOOKS.register_module()
class AdvAAControllerHook(Hook):
    """Adversarial AutoAugment's controller, trained alongside the segmentor (module
    docstring).

    lr, entropy_weight  the controller's Adam learning rate and its entropy bonus
    hidden, embed       the LSTM's hidden and embedding sizes
    epoch_iters         steps per controller update; None is one pass over the train set
    """

    priority = "ABOVE_NORMAL"

    def __init__(
        self,
        lr: float = advaa.LR,
        entropy_weight: float = advaa.ENTROPY_WEIGHT,
        hidden: int = 100,
        embed: int = 32,
        epoch_iters: Optional[int] = None,
    ):
        if epoch_iters is not None and int(epoch_iters) < 1:
            raise ValueError(f"epoch_iters must be >= 1, got {epoch_iters}")
        self.lr = float(lr)
        self.entropy_weight = float(entropy_weight)
        self.hidden = int(hidden)
        self.embed = int(embed)
        self.epoch_iters = None if epoch_iters is None else int(epoch_iters)
        self.controller: Optional[advaa.Controller] = None
        self._logits = None
        self._epoch_start = 0
        self._pruned: tuple = ()

    # --- building and checkpoints -----------------------------------------------------

    def before_run(self, runner) -> None:
        """Build the controller and the first policies before ``load_or_resume``, so a
        checkpoint can replace them."""
        self.host = host_of(runner)
        if not hasattr(self.host, "set_policies"):
            raise TypeError(f"host {self.host.name!r} takes no policies; AdvAAControllerHook drives adversarial_autoaugment")
        model = unwrap(runner.model)
        self.dp = model.data_preprocessor
        if not getattr(self.dp, "augment", True):
            raise ValueError("augment=False trains clean; AdvAAControllerHook would train a controller nothing uses")
        self.device = next(model.parameters()).device
        self.controller = self._new_controller()
        for param in self.controller.parameters():
            broadcast(param.data, src=0)
        # The static prune, at the controller's build (every rank holds rank 0's weights,
        # so every rank removes the same rows).
        self._set_prune(tuple(self.host.pruned()))
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.lr)
        if self.epoch_iters is None:
            loader = runner.train_loop.dataloader
            self.epoch_iters = max(1, math.ceil(len(loader.dataset) / (int(loader.batch_size) * get_world_size())))
        self.means = advaa.EpochMeans(self.host.num_copies, self.device)
        self.epoch = 0
        self._handle = model.decode_head.register_forward_hook(self._stash)
        self._log_path = os.path.join(runner.work_dir, POLICY_LOG)
        self._resample()
        print_log(
            f"[advaa] controller LSTM({self.embed} -> {self.hidden}), {self.controller.num_decisions} "
            f"parameters a policy, {self.controller.param_count()} trainable parameters, "
            f"{self.host.num_copies} policies an epoch of {self.epoch_iters} steps, "
            f"Adam lr {self.lr}, entropy weight {self.entropy_weight}",
            logger="current",
        )
        if self.controller.pruned_types:
            self._log_pruned("the static prune", self.controller.pruned_types)

    def _new_controller(self) -> advaa.Controller:
        return advaa.Controller(
            num_types=len(TYPES),
            num_bins=NUM_BINS,
            num_sub_policies=self.host.num_sub_policies,
            ops_per_sub_policy=self.host.ops_per_sub_policy,
            hidden=self.hidden,
            embed=self.embed,
        ).to(self.device)

    def _run_prune(self, runner) -> tuple:
        """The ops the run prunes now: the static prune plus PruneHook's latched M."""
        latched = getattr(runner, "latched_prune", ()) or ()
        names = set(self.host.pruned()) | set(latched)
        return tuple(n for n in self.host.vocabulary() if n in names)

    def _set_prune(self, names: tuple, optimizer=None) -> tuple:
        """Mask the controller to ``names`` and remove the types left with no bin; returns
        the newly removed types."""
        self._pruned = tuple(names)
        self.mask = torch.as_tensor(self.host.action_mask(extra=names))
        dead = [t for t in range(len(TYPES)) if not bool(self.mask[t].any())]
        return self.controller.prune(dead, optimizer)

    def _log_pruned(self, why: str, types) -> None:
        print_log(
            f"[advaa] {why} removes type(s) {[TYPES[t] for t in types]} from the controller: "
            f"{len(self.controller.pruned_types)} removed in all, {self.controller.param_count()} trainable "
            f"parameters; {int((~self.mask).sum())} of {self.mask.numel()} (type, bin) pairs masked",
            logger="current",
        )

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        checkpoint[CHECKPOINT_KEY] = {
            "state_dict": self.controller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "actions": self.actions.cpu(),
            "sums": self.means.sums.cpu(),
            "count": self.means.count.cpu(),
            "epoch": self.epoch,
            "epoch_start": self._epoch_start,
            "pruned_types": list(self.controller.pruned_types),
            "pruned_ops": list(self._pruned),
            "hidden": self.hidden,
            "embed": self.embed,
        }

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        state = checkpoint.get(CHECKPOINT_KEY)
        if state is None:
            print_log("[advaa] the checkpoint holds no controller; it starts fresh", logger="current", level=30)
            return
        saved = tuple(int(t) for t in state.get("pruned_types") or ())
        run = self._run_prune(runner)
        grown = sorted(set(state.get("pruned_ops") or ()) - set(run))
        if grown:
            raise ValueError(
                f"the checkpoint's controller was pruned of {grown}, which this run doesn't prune; "
                f"a pruned controller can't grow its rows back"
            )
        # A fresh controller pruned as the checkpoint's was, so its shapes match, then
        # pruned to the run (a fork of a less pruned run prunes here).
        controller = self._new_controller()
        controller.prune(saved)
        controller.load_state_dict(state["state_dict"])
        optimizer = torch.optim.Adam(controller.parameters(), lr=self.lr)
        optimizer.load_state_dict(state["optimizer"])
        self.controller, self.optimizer = controller, optimizer
        before = set(saved)
        self._set_prune(run, self.optimizer)
        added = tuple(t for t in self.controller.pruned_types if t not in before)
        if added:
            self._log_pruned(f"at the loaded controller (iter {runner.iter}),", added)
        self.epoch = int(state["epoch"])
        self._epoch_start = int(state.get("epoch_start", 0))
        actions = state["actions"].to(self.device)
        allowed = self.mask.to(self.device)[actions[..., 0], actions[..., 1]]
        if bool(allowed.all()):
            self.actions = actions
            self.host.set_policies(actions.cpu().numpy())
            self.means.sums.copy_(state["sums"])
            self.means.count.copy_(state["count"])
        else:
            print_log("[advaa] the loaded policies apply pruned ops; sampling new ones", logger="current", level=30)
            self.means.reset()
            self._resample()

    # --- the controller -------------------------------------------------------------------

    def _stash(self, module, args, output) -> None:
        if module.training:
            self._logits = output.detach()

    def _resample(self) -> None:
        generator = torch.Generator().manual_seed(int(np.random.randint(0, 2**31 - 1)))
        actions = self.controller.sample(self.host.num_copies, self.mask, generator).to(self.device)
        broadcast(actions, src=0)
        self.actions = actions
        self.host.set_policies(actions.cpu().numpy())

    def _record(self, runner, means, rewards, early: bool = False) -> None:
        if not is_main_process():
            return
        record = {
            "epoch": self.epoch,
            "iter": int(runner.iter) + (0 if early else 1),
            "early": bool(early),
            "pruned_ops": list(self._pruned),
            "policies": self.actions.cpu().tolist(),
            "described": self.host.describe(),
            "copy_losses": means.tolist(),
            "rewards": rewards.tolist(),
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def before_train_iter(self, runner, batch_idx, data_batch=None) -> None:
        # An online prune at the last probe (PruneHook, after_train_iter) takes effect
        # here, before this step's draw.
        names = self._run_prune(runner)
        if set(names) <= set(self._pruned):
            return
        new = sorted(set(names) - set(self._pruned))
        if float(self.means.count) > 0:
            self._step(runner, early=True)
        before = set(self.controller.pruned_types)
        added = self._set_prune(names, self.optimizer)
        runner.message_hub.update_scalar("train/advaa_controller_params", self.controller.param_count())
        print_log(f"[advaa] the online prune (iter {runner.iter}) prunes {new}", logger="current")
        self._log_pruned(f"the online prune (iter {runner.iter})", tuple(t for t in added if t not in before))
        self.means.reset()
        self._epoch_start = int(runner.iter)
        self._resample()
        print_log("[advaa] new policies " + " || ".join(self.host.describe()), logger="current")

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        samples = self.dp.last_train_samples
        if self._logits is None or samples is None:
            raise RuntimeError("[advaa] no training logits or samples were kept for this step")
        head = unwrap(runner.model).decode_head
        losses = advaa.copy_losses(
            lambda logits, part: decode_loss(head, logits, part), self._logits, samples, self.host.num_copies
        )
        self._logits = None
        self.means.add(losses)
        runner.message_hub.update_scalar("train/advaa_copy_loss_mean", float(losses.mean()))
        runner.message_hub.update_scalar("train/advaa_copy_loss_spread", float(losses.max() - losses.min()))
        if (int(runner.iter) + 1 - self._epoch_start) % self.epoch_iters == 0:
            self._update(runner)

    def _update(self, runner) -> None:
        """The epoch's end: the controller's step, then the next epoch's policies."""
        self._step(runner)
        self.means.reset()
        self._resample()
        print_log(f"[advaa] epoch {self.epoch}: controller stepped, new policies " + " || ".join(self.host.describe()), logger="current")

    def _step(self, runner, early: bool = False) -> None:
        """One REINFORCE step on the epoch's means so far (Algorithm 1), under the current
        mask; ``early`` marks an epoch an online prune closed."""
        sums, count = self.means.sums.clone(), self.means.count.clone()
        all_reduce(sums, op="sum")
        all_reduce(count, op="sum")
        means = sums / count
        rewards = advaa.normalize(means)
        self._record(runner, means, rewards, early)

        self.optimizer.zero_grad(set_to_none=True)
        log_prob, entropy = self.controller.log_prob_entropy(self.actions, self.mask)
        loss = advaa.controller_loss(log_prob, entropy, rewards.to(self.device), self.entropy_weight)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        for param in self.controller.parameters():
            broadcast(param.data, src=0)

        for key, value in (
            ("controller_loss", float(loss.detach())),
            ("entropy", float(entropy.detach().mean())),
            ("copy_loss_epoch_mean", float(means.mean())),
            ("copy_loss_epoch_spread", float(means.max() - means.min())),
        ):
            runner.message_hub.update_scalar(f"train/advaa_{key}", value)
        runner.message_hub.update_scalar("train/advaa_controller_params", self.controller.param_count())
        self.epoch += 1
