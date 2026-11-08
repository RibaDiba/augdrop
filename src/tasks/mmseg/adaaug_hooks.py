"""mmengine glue for AdaAug's policy in training. Nothing is computed here.

    AdaAugPolicyHook   before every training step: h on each clean image's feature
                       (``madaug.predict`` at the host's temperature), folded into the
                       host's prior (``observe``) and drawn by the host (``draw``), the
                       slots installed on the preprocessor (``set_policy_slots``), which
                       applies them inside ``train_step``. h never trains here.

Two policies, by the host's ``policy_file`` (``hosts/adaaug.py``):

- ``initial``, the search's starting policy (the search-then-train recipe's trunk): an
  untrained h (``madaug.Projection``, seeded by ``seed``, the same on every rank) on the
  live segmentor's pooled deepest feature (``backend.pooled_features``, the model in
  eval), drawn as the search's exploit draws (k = 1, δ = 0, T = 1). Its state goes into
  every checkpoint (``adaaug_policy``)
- a search's directory (``scripts/adaaug_search.py``): the searched h and the search
  model gf, both frozen and in eval, from ``final/`` (``gf.pth``, ``h.pth``; gf rebuilt
  from ``gf_config.py``, the search's segmentor config with a plain preprocessor). h
  reads gf's feature, as AdaAug's ``train.py`` reads its frozen search model's
  (``gf_model.f``), normalized as gf's preprocessor normalizes. A fork of the trunk
  (``FORK_FROM``) loads a checkpoint holding the starting h, and ignores it

Every step logs h's size (``train/adaaug_policy_params``) and the batch's mean p(Identity)
(``train/adaaug_identity_share``).

Checked on stand-ins for mmengine and mmseg only (no mmengine in ``sensAug``).
"""

import os
from typing import Optional

import numpy as np
import torch
from mmengine.dist import broadcast
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS

import madaug
from adaaug.explore import pooled
from hosts.adaaug import FINAL_DIR, IDENTITY

from .backend import MMSegBackend, pooled_features, unwrap
from .hooks import host_of

__all__ = ["CHECKPOINT_KEY", "GF_CONFIG", "AdaAugPolicyHook", "load_search_model"]


def _rng() -> np.random.Generator:
    """A generator for one step's draws, seeded from numpy's global RNG, so the
    framework's seeding governs it (as ``madaug_hooks._rng``)."""
    return np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))

#: Where the starting policy lives in a checkpoint.
CHECKPOINT_KEY = "adaaug_policy"

#: The search segmentor's config in a search's ``final/`` (scripts/adaaug_search.py).
GF_CONFIG = "gf_config.py"


def load_search_model(directory: str, device=None) -> MMSegBackend:
    """The frozen search model gf from a search's ``final/`` (or its directory): an
    ``MMSegBackend`` on the rebuilt segmentor, ``gf.pth`` loaded strictly, in eval,
    with no gradient. Its ``_normalize`` is gf's own."""
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings

    directory = os.path.abspath(directory)
    if os.path.basename(directory.rstrip(os.sep)) != FINAL_DIR:
        directory = os.path.join(directory, FINAL_DIR)
    cfg = Config.fromfile(os.path.join(directory, GF_CONFIG))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    backend = MMSegBackend(cfg, device=device)
    model = backend.build_model()
    state = torch.load(os.path.join(directory, "gf.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return backend


@HOOKS.register_module()
class AdaAugPolicyHook(Hook):
    """AdaAug's policy applied to each training batch (module docstring).

    n_layers, n_hidden
              the starting h's hidden layers (AdaAug's ``--n_proj_layer``, 0 = linear),
              the shape the search's h starts from
    seed      the starting h's initialization
    """

    priority = "ABOVE_NORMAL"

    def __init__(self, n_layers: int = 0, n_hidden: int = 128, seed: int = 0):
        self.n_layers = int(n_layers)
        self.n_hidden = int(n_hidden)
        self.seed = int(seed)
        self.policy: Optional[madaug.Projection] = None
        self.gf: Optional[MMSegBackend] = None

    def before_run(self, runner) -> None:
        self.host = host_of(runner)
        if getattr(self.host, "name", None) != "adaaug":
            raise TypeError(f"host {getattr(self.host, 'name', None)!r} is not adaaug; AdaAugPolicyHook drives hosts.adaaug")
        model = unwrap(runner.model)
        self.dp = model.data_preprocessor
        if not getattr(self.dp, "augment", True):
            raise ValueError("augment=False trains clean; AdaAugPolicyHook would draw for nothing")
        device = next(model.parameters()).device
        n_outputs = len(self.host.outputs)
        if self.host.searched:
            directory = self.host.policy_meta["directory"]
            self.gf = load_search_model(directory, device=device)
            meta = self.host.policy_meta
            self.policy = madaug.Projection(
                int(meta["feature_dim"]), n_layers=int(meta["n_layers"]), n_hidden=int(meta.get("n_hidden", 128)), n_outputs=n_outputs
            ).to(device)
            self.policy.load_state_dict(torch.load(os.path.join(directory, "h.pth"), map_location="cpu", weights_only=False))
            source = f"the searched policy at {directory} (h and gf frozen)"
        else:
            size = tuple(getattr(self.dp, "size", None) or (512, 512))
            with torch.no_grad(), madaug.eval_mode(model):
                width = pooled_features(model, torch.zeros(1, 3, *size, device=device)).shape[1]
            generator_state = torch.random.get_rng_state()
            torch.manual_seed(self.seed)
            try:
                self.policy = madaug.Projection(int(width), n_layers=self.n_layers, n_hidden=self.n_hidden, n_outputs=n_outputs)
            finally:
                torch.random.set_rng_state(generator_state)
            self.policy = self.policy.to(device)
            for param in self.policy.parameters():
                broadcast(param.data, src=0)
            source = f"the starting policy (untrained, seed {self.seed}) on the live segmentor's feature"
        self.policy.requires_grad_(False)
        self.policy.eval()
        self._params = sum(p.numel() for p in self.policy.parameters())
        print_log(
            f"[adaaug] {source}: {n_outputs} outputs, {self._params} parameters, draws at {self.host.settings}"
            + (f"; the search left out {self.host.policy_meta['pruned_ops'].split()}" if self.host.searched else ""),
            logger="current",
        )

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        checkpoint[CHECKPOINT_KEY] = {
            "policy_file": self.host.policy_file,
            "outputs": list(self.host.outputs),
            "state_dict": None if self.host.searched else self.policy.state_dict(),
        }

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        state = checkpoint.get(CHECKPOINT_KEY)
        if state is None:
            return
        if state.get("policy_file") != self.host.policy_file or state.get("state_dict") is None:
            # A fork of the trunk onto a searched policy: the trunk's starting h is not it.
            print_log(
                f"[adaaug] the checkpoint's policy ({state.get('policy_file')}) is not this run's "
                f"({self.host.policy_file}); the run's is kept",
                logger="current",
            )
            return
        if list(state.get("outputs") or ()) != list(self.host.outputs):
            raise ValueError(
                f"the checkpoint's starting policy has outputs {state.get('outputs')}, this run's {list(self.host.outputs)}"
            )
        self.policy.load_state_dict(state["state_dict"])

    def before_train_iter(self, runner, batch_idx, data_batch=None) -> None:
        model = unwrap(runner.model)
        with self.dp.suspended():
            data = self.dp(data_batch, training=True)
        inputs = data["inputs"]
        with torch.no_grad():
            if self.gf is None:
                with madaug.eval_mode(model):
                    features = pooled_features(model, inputs)
            else:
                images01 = ((inputs * self.dp.std.view(1, -1, 1, 1) + self.dp.mean.view(1, -1, 1, 1)) / 255.0).clamp(0.0, 1.0)
                features = pooled(self.gf.model, self.gf._normalize(images01))
            levels, p = madaug.predict(self.policy, features, self.host.temperature)
        p = p.double().cpu().numpy()
        levels = levels.double().cpu().numpy()
        self.host.observe(p, levels)
        self.dp.set_policy_slots(self.host.draw(p, levels, _rng()))
        hub = runner.message_hub
        hub.update_scalar("train/adaaug_policy_params", self._params)
        hub.update_scalar("train/adaaug_identity_share", float(p[:, self.host.outputs.index(IDENTITY)].mean()))
