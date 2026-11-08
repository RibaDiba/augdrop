"""Experiment configs: compose ``configs/`` with Hydra, check that the groups fit
together, and build what each group describes.

    compose(overrides)       configs/train.yaml + Hydra overrides -> a resolved DictConfig
    check(cfg, training=)    cross-group checks; raises ConfigError, returns warnings
    save(cfg, work_dir)      <work_dir>/dropaug.yaml, the file probe.py / eval.py rebuild from
    load(work_dir)           ... and back, with dotted overrides on top
    vocab_of / host_of / fade_of / prune_of / selector_spec / magnitudes_of / run_name

One experiment is one choice per group (``GROUPS``). What a group's keys mean is in
its yaml files. What this module adds is the rules that span groups, checked before
anything is built, so a bad combination fails in seconds rather than hours into a run:

- the dataset's ``kind`` is one the task's backend serves (seg for mmseg, cls for cls),
  and an ``eval_only`` dataset (IDD) is never trained on
- the vocab is the host's vocabulary. The sensaug and uniform hosts take it as theirs;
  the torchvision hosts have a fixed one, and the vocab must equal it
- training augments only with the hosts and ops ``tasks.augment`` can apply
- the selector builds with its keys, and the control (``selector=none``) keeps lam 0
- the probe magnitude is valid for every op of the vocab, and ``probe.magnitude_mode``
  is one of ``sa.MAGNITUDE_MODES``
- the host's ``fade:`` block (which ops the probe fades in, ``probe.fade``) names a rule
  or ops of the vocab, and its ``at`` is in [0, 1]
- the ``prune:`` block (``prune_of``: the static prune, ``hosts.prune``) names ops of the
  vocab and leaves something to train on; a prune with no effect (``augment=false``)
  is warned about, and so are ``hold_none_prob`` / ``probe`` with nothing pruned. The
  old ``sa.pruned_augmentations`` / ``sa.hold_none_prob`` are refused: they moved there
- the ``sa`` group (the SA loop's knobs) matters only with the sensaug host: elsewhere a
  changed value is warned about. With sensaug, the host's ``pre_train_round`` must equal
  ``sa.init_sa`` (an error: the probe rounds shift otherwise), and ``task.val_interval``
  should be the host's ``round_interval`` (a warning: the mmseg builder sets it)

``vocab`` is a top-level group like the others: each host config overrides its choice
(``- override /vocab: ...``), and ``vocab=<name>`` on the command line wins over that.

The composed config is resolved (no interpolations left) before it is saved, so a
run rebuilt from its ``dropaug.yaml`` doesn't depend on the configs/ it was composed
from, except for an mmseg task's .py file, which train.py snapshots into the run.
"""

import hashlib
import inspect
import os
import re
from typing import Iterable, List, Optional, Sequence

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = [
    "CONFIG_DIR",
    "ConfigError",
    "GROUPS",
    "RUN_FILE",
    "check",
    "compose",
    "checkpoint_interval",
    "fade_of",
    "fork_iter",
    "fork_tag",
    "group_options",
    "host_of",
    "load",
    "magnitudes_of",
    "policy_of",
    "deepaa_search_of",
    "search_of",
    "prune_of",
    "prune_tag",
    "run_name",
    "unprobed_ops",
    "save",
    "selector_spec",
    "task_for_backbone",
    "to_python",
    "vocab_of",
]

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")

#: The config groups one experiment chooses from, in configs/train.yaml's order.
#: ``vocab`` is chosen by the host's own defaults unless overridden. ``sa`` is read only
#: by the sensaug host. ``recipe`` is not chosen: it is ``<task>_<dataset>``, Sensaug's
#: recipe for that pair, and is recorded only when configs/recipe/ has one.
GROUPS = ("cluster", "task", "dataset", "host", "vocab", "selector", "probe", "sa", "recipe")

#: The resolved config a run is rebuilt from.
RUN_FILE = "dropaug.yaml"

#: Which dataset kind each task backend serves.
BACKEND_KINDS = {"mmseg": "seg", "cls": "cls"}


class ConfigError(ValueError):
    """The composed groups don't fit together."""


def _idiv(a, b) -> int:
    return int(a) // int(b)


# ${idiv:a,b}: integer division, for intervals derived from max_iters.
if not OmegaConf.has_resolver("idiv"):
    OmegaConf.register_new_resolver("idiv", _idiv)


# --- composing ------------------------------------------------------------------------


def compose(overrides: Sequence[str] = (), config_name: str = "train") -> DictConfig:
    """configs/<config_name>.yaml with Hydra ``overrides`` (``host=uniform``,
    ``selector.lam=0.5``, ``+experiment=...``), resolved. The group choices are kept
    under ``choices``, so a saved run still says which files it came from."""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = hydra_compose(config_name, overrides=list(overrides), return_hydra_config=True)
    choices = {group: cfg.hydra.runtime.choices.get(group) for group in (*GROUPS, "experiment")}
    # Hydra reports the optional recipe's interpolated name whether or not the file exists.
    if choices.get("recipe") and choices["recipe"] not in group_options("recipe"):
        choices["recipe"] = None
    OmegaConf.set_struct(cfg, False)
    del cfg["hydra"]
    cfg.choices = {group: choice for group, choice in choices.items() if choice is not None}
    OmegaConf.resolve(cfg)
    return cfg


def group_options(group: str) -> List[str]:
    """The option names a group offers: its yaml files."""
    directory = os.path.join(CONFIG_DIR, group)
    return sorted(f[: -len(".yaml")] for f in os.listdir(directory) if f.endswith(".yaml"))


def task_for_backbone(name: str) -> str:
    """The ``task`` option whose yaml says ``backbone: <name>`` (train.py --backbone)."""
    found, known = [], set()
    for option in group_options("task"):
        node = OmegaConf.load(os.path.join(CONFIG_DIR, "task", option + ".yaml"))
        backbone = node.get("backbone")
        if backbone is None:
            continue
        known.add(str(backbone))
        if backbone == name:
            found.append(option)
    if len(found) != 1:
        problem = "no task config has" if not found else f"several task configs ({found}) have"
        raise ConfigError(f"{problem} backbone {name!r}; known backbones: {sorted(known)}")
    return found[0]


def save(cfg: DictConfig, work_dir: str) -> str:
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, RUN_FILE)
    OmegaConf.save(cfg, path, resolve=True)
    return path


def load(work_dir: str, overrides: Iterable[str] = ()) -> DictConfig:
    """A run's saved config, with dotted ``key=value`` overrides merged on top (e.g.
    ``probe.magnitude=0.3``). Groups can't be switched here: the run was trained with
    the ones it has."""
    path = os.path.join(work_dir, RUN_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist; {work_dir} was not trained by train.py, or is not a run directory"
        )
    cfg = OmegaConf.load(path)
    overrides = list(overrides)
    if overrides:
        switched = [o for o in overrides if o.split("=", 1)[0].lstrip("+~") in GROUPS]
        if switched:
            raise ConfigError(
                f"{switched} would switch a whole group of a trained run; only its keys can "
                f"be overridden here (e.g. probe.magnitude=0.3)"
            )
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def to_python(node):
    """An OmegaConf node as plain python (dicts, lists, scalars)."""
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True)
    return node


def run_name(cfg: DictConfig) -> str:
    """``<host>_<selector>[-l<lam>]_<dataset>_<backbone>_s<seed>[_prune-...][_fork-i<N>]
    [_deepaa-search]``, the name slurm/train.sbatch gives a run directory (``prune_tag`` and
    ``fork_tag`` for the suffixes; the last for a deep_autoaugment run that searches in the
    run, ``DEEPAA_POLICY=search``)."""
    selector = cfg.choices.get("selector", cfg.selector.name)
    null = cfg.selector.get("null_arm")
    if null:
        # NMS's matched null is its own arm: nms-random<seed>-l1.
        selector = f"{selector}-{null}{int(cfg.selector.get('null_seed', 0))}"
    arm = "none" if cfg.selector.get("control") else f"{selector}-l{float(cfg.selector.lam):g}"
    host = cfg.choices.get("host", cfg.host.name)
    search = "_deepaa-search" if cfg.host.get("name") == "deep_autoaugment" and cfg.host.get("policy_file") == "search" else ""
    return f"{host}_{arm}_{cfg.dataset.name}_{cfg.task.backbone}_s{cfg.seed}{prune_tag(cfg)}{fork_tag(cfg)}{search}"


#: mmengine's checkpoint names (``iter_<N>.pth``, ``best_mIoU_iter_<N>.pth``): N is the
#: completed iteration the weights are from.
_CHECKPOINT_ITER = re.compile(r"iter_(\d+)\.pth$")


def fork_iter(path: str) -> int:
    """The completed iteration of the checkpoint ``path`` names (mmengine's
    ``iter_<N>.pth``). Raises ConfigError on any other name: a fork's run name and its
    resumed iteration both come from it."""
    match = _CHECKPOINT_ITER.search(os.path.basename(str(path)))
    if not match:
        raise ConfigError(f"fork_from {path!r} is not an mmengine iter_<N>.pth checkpoint")
    return int(match.group(1))


def fork_tag(cfg: DictConfig) -> str:
    """The run name's fork suffix: ``_fork-i<N>`` for a run forked from another run's
    ``iter_<N>.pth`` (``fork_from``), else empty."""
    path = cfg.get("fork_from")
    return f"_fork-i{fork_iter(path)}" if path else ""


def checkpoint_interval(cfg: DictConfig) -> int:
    """Iterations between the checkpoints a run saves: the task's ``checkpoint_interval``,
    else max_iters // 20. The mmseg builder sets CheckpointHook's interval from it."""
    task = cfg.task
    return max(1, int(task.get("checkpoint_interval") or int(task.max_iters) // 20))


def _names_hash(names: Iterable[str]) -> str:
    """6 hex digits of sha1 over the sorted, distinct names, one per line. train.sbatch
    computes the same: ``printf '%s\\n' ... | LC_ALL=C sort -u | sha1sum | cut -c1-6``."""
    text = "".join(f"{name}\n" for name in sorted(set(names)))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]


def prune_tag(cfg: DictConfig, prune=None) -> str:
    """The run name's prune suffix: empty with nothing pruned, else ``_prune-`` and
    ``o<N>.<hash>`` for N named ops, ``r<K>`` for K drawn (``r<K>s<seed>`` when
    ``prune.seed`` is not the run's seed, then ``p`` when the draw's pool is
    ``prune.pool=probed``), joined by ``-``. ``hold_none_prob`` and
    ``probe`` are not in it: a run differing only there needs a RUN_TAG."""
    prune = prune_of(cfg) if prune is None else prune
    if not prune:
        return ""
    parts = []
    if prune.named:
        parts.append(f"o{len(prune.named)}.{_names_hash(prune.named)}")
    if prune.random:
        parts.append(
            f"r{prune.random}"
            + ("" if prune.seed == int(cfg.seed) else f"s{prune.seed}")
            + ("p" if prune.pool_rule == POOL_PROBED else "")
        )
    return "_prune-" + "-".join(parts)


# --- building ---------------------------------------------------------------------------


def vocab_of(cfg: DictConfig, vocab: Optional[DictConfig] = None) -> dict:
    """The vocab (``vocab.build_vocab``): the ops the host samples, the probe measures
    and the eval scores."""
    from vocab import build_vocab

    return build_vocab(to_python(cfg.vocab if vocab is None else vocab))


def _takes_vocabulary(name: str) -> bool:
    from hosts import HOSTS

    return "vocabulary" in inspect.signature(HOSTS[name].__init__).parameters


def host_kwargs(cfg: DictConfig, prune=None) -> dict:
    """``build_host`` kwargs, with the vocab passed in to the hosts that take one, and
    the static prune (``prune_of``, or ``prune`` when given) as ``pruned`` /
    ``hold_none_prob`` when it prunes anything."""
    kwargs = to_python(cfg.host)
    kwargs.pop("vocab", None)  # the vocab is passed as `vocabulary`, never as a host key
    kwargs.pop("fade", None)  # the probe's, not the host's: see fade_of
    kwargs.pop("policy", None)  # the policy hook's, not the host's: see policy_of
    kwargs.pop("search", None)  # Faster AutoAugment's search, not the host's: see search_of
    kwargs.pop("controller", None)  # Adversarial AutoAugment's controller hook's: see controller_of
    name = kwargs.pop("name")
    if _takes_vocabulary(name):
        kwargs["vocabulary"] = list(vocab_of(cfg))
    prune = prune_of(cfg) if prune is None else prune
    if prune:
        kwargs["pruned"] = list(prune.ops)
        kwargs["hold_none_prob"] = prune.hold_none_prob
    return {"name": name, **kwargs}


def host_of(cfg: DictConfig, prune=None):
    """The host the config describes (checked against the vocab by ``check``), with its
    static prune applied."""
    from hosts import build_host

    kwargs = host_kwargs(cfg, prune)
    return build_host(kwargs.pop("name"), **kwargs)


#: ``prune.pool``: the random draw's pool. ``all`` is Sensaug's (the vocabulary minus the
#: named ops); ``probed`` also leaves out the ops the probe gives no row under the host's
#: fade (no R-based selector can prune those), for a random-K null matched to NMS.
POOL_ALL, POOL_PROBED = "all", "probed"

#: The sa keys the prune: block replaced, and where each went.
_MOVED_SA_KEYS = {"pruned_augmentations": "prune.ops", "hold_none_prob": "prune.hold_none_prob"}


def prune_of(cfg: DictConfig, vocabulary: Optional[Sequence[str]] = None):
    """The run's static prune (``hosts.Prune``) from its ``prune:`` block, on the vocab
    (or ``vocabulary``). A pure function of the config: the random draw is seeded by
    ``prune.seed``, so every rank and every rebuild of the run gets the same ops. Its
    pool leaves out ``sa.REMOVE_H_NAMES`` when the sensaug host runs with
    ``sa.remove_H``, as Sensaug's did. A config with no ``prune:`` block (saved before
    it existed) prunes nothing. Raises ConfigError on a bad block, and on the
    ``sa.pruned_augmentations`` / ``sa.hold_none_prob`` it replaced."""
    from hosts import resolve_prune

    sa_cfg = cfg.get("sa") or {}
    moved = [
        f"sa.{key} -> {target}"
        for key, target in _MOVED_SA_KEYS.items()
        if to_python(sa_cfg.get(key)) not in (None, [], False)
    ]
    if moved:
        raise ConfigError(f"these keys moved to the prune: block (configs/train.yaml): {', '.join(moved)}")

    block = to_python(cfg.get("prune")) or {}
    unknown = sorted(set(block) - {"ops", "random", "seed", "hold_none_prob", "probe", "pool"})
    if unknown:
        raise ConfigError(f"prune: unknown key(s) {unknown}; expected ops, random, seed, hold_none_prob, probe, pool")
    pool_rule = block.get("pool") or POOL_ALL
    if pool_rule not in (POOL_ALL, POOL_PROBED):
        raise ConfigError(f"prune.pool must be {POOL_ALL!r} or {POOL_PROBED!r}, got {pool_rule!r}")
    exclude = ()
    if cfg.host.get("name") == "sensaug" and sa_cfg.get("remove_H"):
        from sa import REMOVE_H_NAMES

        exclude = tuple(REMOVE_H_NAMES)
    if pool_rule == POOL_PROBED:
        exclude = exclude + unprobed_ops(cfg)
    ops = block.get("ops") or []
    if isinstance(ops, str):
        raise ConfigError(f"prune.ops must be a list of op names, got the string {ops!r}")
    try:
        return resolve_prune(
            list(vocab_of(cfg)) if vocabulary is None else vocabulary,
            ops=ops,
            random=block.get("random") or 0,
            seed=block.get("seed"),
            exclude=exclude,
            hold_none_prob=block.get("hold_none_prob", True),
            probe=block.get("probe", False),
            pool_rule=pool_rule,
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from None


def unprobed_ops(cfg: DictConfig) -> tuple:
    """The vocab ops the probe gives no row under the host's fade (``probe.resolve_methods``
    says ``unprobed``: no magnitude and not faded, or ``couples_images``), in vocab order.
    R sets them aside, so no R-based selector can prune them."""
    from probe import resolve_methods
    from probe.grad_collect import UNPROBED

    ops = vocab_of(cfg)
    fade, _ = fade_of(cfg)
    methods = resolve_methods(ops, fade)
    return tuple(name for name in ops if methods[name] == UNPROBED)


def fade_of(cfg: DictConfig) -> tuple:
    """``(fade, fade_at)`` from the host's ``fade:`` block: which ops the probe fades
    in (None, a ``probe.FADE_RULES`` rule, or a list of op names) and the ``a`` it
    differentiates at. No block, or ``ops: null``, fades nothing."""
    block = to_python(cfg.host.get("fade")) or {}
    return block.get("ops"), float(block.get("at", 1.0))


def policy_of(cfg: DictConfig) -> Optional[dict]:
    """The host's ``policy:`` block (MADAug's policy training, MADAugPolicyHook's
    kwargs), or None for a host without one."""
    block = to_python(cfg.host.get("policy"))
    return None if block is None else dict(block)


def controller_of(cfg: DictConfig) -> Optional[dict]:
    """The host's ``controller:`` block (Adversarial AutoAugment's controller,
    AdvAAControllerHook's kwargs), or None for a host without one."""
    block = to_python(cfg.host.get("controller"))
    return None if block is None else dict(block)


def search_of(cfg: DictConfig) -> Optional[dict]:
    """The host's ``search:`` block (Faster AutoAugment's policy search,
    scripts/faa_search.py), or None for a host without one."""
    block = to_python(cfg.host.get("search"))
    return None if block is None else dict(block)


#: DeepAASearchHook's own keys in Deep AutoAugment's ``search:`` block; the rest are
#: ``deepaa.Settings``.
_DEEPAA_HOOK_KEYS = ("at", "save_every", "seed")


def deepaa_search_of(cfg: DictConfig) -> Optional[dict]:
    """DeepAASearchHook's kwargs for a deep_autoaugment run that searches in the run
    (``host.policy_file: search``), from the host's ``search:`` block, ``at`` resolved to
    a completed iteration (a fraction of max_iters below 1, else the iteration); None for
    any other run."""
    if cfg.host.get("name") != "deep_autoaugment" or cfg.host.get("policy_file") != "search":
        return None
    block = search_of(cfg)
    if block is None:
        return None
    at = float(block.get("at", 0.25))
    block["at"] = int(round(at * int(cfg.task.max_iters))) if at < 1 else int(at)
    return block


def selector_spec(cfg: DictConfig) -> dict:
    """ApplyWeightsHook's ``selector`` dict (``type``, target ``lam``, the selector's
    own kwargs) plus ``lambda_ramp``, which the hook takes separately."""
    spec = to_python(cfg.selector)
    spec.pop("control", None)
    name = spec.pop("name")
    ramp = spec.pop("lambda_ramp", "linear")
    return {"selector": {"type": name, **spec}, "lambda_ramp": ramp}


def magnitudes_of(magnitude):
    """``probe.magnitude`` as a ``probe.Magnitudes``: a strength for every op, or
    ``{op: magnitude}`` in each op's own units. The same rule as CollectGradientHook."""
    import probe

    magnitude = to_python(magnitude)
    if isinstance(magnitude, dict):
        return probe.Magnitudes.fixed(magnitude)
    return probe.Magnitudes.relative(float(magnitude))


# --- checking -----------------------------------------------------------------------------


def _check_deepaa_search(cfg: DictConfig, host, backend: str, selector_name: str) -> List[str]:
    """Deep AutoAugment's search in the run (``policy_file: search``): the errors."""
    import deepaa

    errors: List[str] = []
    block = search_of(cfg)
    if block is None:
        return ["host.policy_file=search needs the host's search: block (DeepAASearchHook's settings)"]
    if backend != "mmseg":
        errors.append(f"host.policy_file=search runs DeepAASearchHook, which is mmseg's; the task is {backend}")
    hook = deepaa_search_of(cfg)
    try:
        deepaa.Settings(**{k: v for k, v in hook.items() if k not in _DEEPAA_HOOK_KEYS})
    except (TypeError, ValueError) as error:
        errors.append(f"host.search: {error}")
    at, max_iters = hook["at"], int(cfg.task.max_iters)
    interval = checkpoint_interval(cfg)
    if not 0 < at < max_iters:
        errors.append(f"host.search.at is iteration {at}; the search needs one inside (0, {max_iters})")
    elif at % interval:
        errors.append(
            f"host.search.at is iteration {at}, off the checkpoint grid (every {interval}): a job stopped "
            f"in the search could not resume at the pause (set task.checkpoint_interval or host.search.at)"
        )
    if selector_name == "nms" and host.probe_role(at) != "active":
        errors.append(
            f"selector nms prunes the search space at the pause, but the probe has no active probe at "
            f"iteration {at} (probe.schedule.iters / interval)"
        )
    return errors



def check(cfg: DictConfig, *, training: bool) -> List[str]:
    """Raise ConfigError on groups that don't fit together; return the warnings (things
    that run, but not the way the config's names suggest). ``training`` adds the
    checks only a training run needs."""
    import probe
    from hosts import HOSTS
    from sa import MAGNITUDE_MODES
    from selection import SELECTORS, build_selector

    errors: List[str] = []
    warnings: List[str] = []
    choices = cfg.get("choices", {})

    backend = cfg.task.backend
    if backend not in BACKEND_KINDS:
        errors.append(f"task {choices.get('task')!r} names backend {backend!r}; known: {sorted(BACKEND_KINDS)}")
    elif cfg.dataset.kind != BACKEND_KINDS[backend]:
        errors.append(
            f"dataset {cfg.dataset.name!r} is a {cfg.dataset.kind} dataset and task "
            f"{choices.get('task')!r} ({backend}) trains {BACKEND_KINDS[backend]} models"
        )
    if training and cfg.dataset.role == "eval_only":
        errors.append(
            f"dataset {cfg.dataset.name!r} is eval-only (a shift target for "
            f"{cfg.dataset.get('eval_for')!r}); train on that and evaluate with "
            f"scripts/eval.py --dataset {cfg.dataset.name}"
        )

    # The static prune. On failure the host below is checked without it.
    from hosts import Prune

    try:
        pruning = prune_of(cfg)
    except (ConfigError, KeyError) as error:
        errors.append(f"prune: {error}")
        pruning = Prune()

    # The host, and its vocabulary against the vocab.
    host = None
    name = cfg.host.name
    if name not in HOSTS:
        errors.append(f"host {name!r} is not one of {sorted(HOSTS)}")
    else:
        try:
            ops = vocab_of(cfg)
            host = host_of(cfg, pruning)
        except (KeyError, ValueError, TypeError, OSError) as error:
            errors.append(f"host {name!r} with vocab {choices.get('vocab')!r}: {error}")
        else:
            if tuple(ops) != tuple(host.vocabulary()):
                errors.append(
                    f"host {name!r} samples a fixed vocabulary of {len(host.vocabulary())} ops "
                    f"{list(host.vocabulary())[:3]}..., but vocab {choices.get('vocab')!r} is "
                    f"{list(ops)[:3]}... ({len(ops)} ops). Use the vocab its host config names, "
                    f"or a host that takes any vocab (sensaug, uniform)."
                )
                host = None
    if host is not None and pruning:
        implied = [n for n in host.pruned() if n not in pruning.ops]
        if implied:
            warnings.append(
                f"pruning {list(pruning.ops)} on host {name!r} also prunes {implied}: every "
                f"sub-policy that applied them applies a pruned op too"
            )
        if pruning.pool_rule != POOL_ALL and not pruning.random:
            warnings.append(f"prune.pool={pruning.pool_rule} set, but nothing is drawn at random: no effect")
        if not cfg.augment:
            warnings.append(
                f"prune.* prunes {list(pruning.ops)}, but augment=false trains on clean images: "
                f"the prune only keeps those ops out of the probe and the selector's prior"
            )
    elif not pruning:
        block = to_python(cfg.get("prune")) or {}
        inert = [
            f"prune.{key}={block[key]}"
            for key, default in (("hold_none_prob", True), ("probe", False), ("pool", POOL_ALL))
            if key in block and block[key] != default
        ]
        if inert:
            warnings.append(f"{', '.join(inert)} set, but nothing is pruned: no effect")

    # The selector.
    spec = selector_spec(cfg)["selector"]
    selector_name = spec.pop("type")
    if cfg.selector.get("control") and float(cfg.selector.lam) != 0.0:
        errors.append(
            f"selector 'none' is the lambda = 0 control, but lam is {cfg.selector.lam}; pick a "
            f"selector (e.g. selector=kl_reweight) to give lam a value"
        )
    if selector_name not in SELECTORS:
        errors.append(f"selector {selector_name!r} is not one of {sorted(SELECTORS)}")
    else:
        try:
            build_selector(selector_name, **spec)
        except (TypeError, ValueError) as error:
            errors.append(f"selector {selector_name!r} doesn't build from {spec}: {error}")
    if selector_name in ("cover_single", "cover_even") and name not in ("trivialaugment", "trivialaugment_diff", "sensaug"):
        errors.append(
            f"selector {selector_name!r} builds a per-op q that is only checked on a TrivialAugment "
            f"host (one shared bin per image, so cover_single's q means 'apply the cover "
            f"instead') and on sensaug (q scales each op's levels by one factor); host {name!r} "
            f"is neither"
        )
    if cfg.selector.get("lambda_ramp") not in ("linear", "constant"):
        errors.append(f"selector.lambda_ramp must be linear or constant, got {cfg.selector.get('lambda_ramp')!r}")

    # The probe.
    if cfg.probe.get("magnitude_mode") not in MAGNITUDE_MODES:
        errors.append(f"probe.magnitude_mode must be one of {MAGNITUDE_MODES}, got {cfg.probe.get('magnitude_mode')!r}")
    if host is not None:
        try:
            magnitudes_of(cfg.probe.magnitude).draw(vocab_of(cfg), 1, np.random.default_rng(0))
        except (KeyError, ValueError) as error:
            errors.append(f"probe.magnitude={to_python(cfg.probe.magnitude)!r}: {error}")
        try:
            fade, fade_at = fade_of(cfg)
            probe.faded_ops(vocab_of(cfg), fade)
            probe.check_fade_at(fade_at)
        except (TypeError, ValueError) as error:
            errors.append(f"host.fade={to_python(cfg.host.get('fade'))!r}: {error}")

    if training and host is not None:
        if cfg.augment:
            from tasks.augment import check_trainable

            try:
                check_trainable(host, vocab_of(cfg))
            except NotImplementedError as error:
                errors.append(f"{error} (set augment=false to train clean and probe anyway)")
            policy = policy_of(cfg)
            if policy is not None and name == "adaaug":
                if backend != "mmseg":
                    errors.append(
                        f"host {name!r} applies its policy with AdaAugPolicyHook, which is mmseg's; "
                        f"task {choices.get('task')!r} is {backend} (set augment=false to probe only)"
                    )
            elif policy is not None:
                import madaug

                if backend != "mmseg":
                    errors.append(
                        f"host {name!r} trains its policy with MADAugPolicyHook, which is mmseg's; "
                        f"task {choices.get('task')!r} is {backend} (set augment=false to probe only)"
                    )
                if policy.get("hypergrad", "exact") not in madaug.MODES:
                    errors.append(f"host.policy.hypergrad must be one of {madaug.MODES}, got {policy['hypergrad']!r}")
                if int(policy.get("n_layers", 0)) > 0 and (host.pruned() or selector_name == "nms"):
                    errors.append(
                        f"host.policy.n_layers={policy['n_layers']}: a pruned MADAug policy must be linear "
                        f"(n_layers=0), since its frozen logits would need the hidden layers as they were "
                        f"at each prune (madaug.Projection.prune)"
                    )
            if name == "deep_autoaugment" and cfg.host.get("policy_file") == "search":
                errors.extend(_check_deepaa_search(cfg, host, backend, selector_name))
            if controller_of(cfg) is not None and backend != "mmseg":
                errors.append(
                    f"host {name!r} trains its controller with AdvAAControllerHook, which is mmseg's; "
                    f"task {choices.get('task')!r} is {backend} (set augment=false to probe only)"
                )
            # A policy searched over a subset (scripts/faa_search.py --prune-ops / --random-prune)
            # never applies the ops it left out; the run's prune should say so, or its name and
            # prune.json don't.
            searched_out = set(str((getattr(host, "policy_meta", None) or {}).get("pruned_ops") or "").split())
            unlisted = sorted(searched_out - set(host.pruned()))
            if unlisted:
                warnings.append(
                    f"host.policy_file was searched without {unlisted}, which the run's prune doesn't list: "
                    f"they are never applied, but the run name and prune.json won't say so (PRUNE_OPS=...)"
                )
            copies = int(getattr(host, "num_copies", 1))
            if copies > 1 and backend != "mmseg":
                errors.append(
                    f"host {name!r} trains on {copies} copies of each image (num_copies), which only "
                    f"the mmseg preprocessor makes; task {choices.get('task')!r} is {backend}"
                )
        max_iters = int(cfg.task.max_iters)
        search = deepaa_search_of(cfg) if cfg.augment else None
        if search is not None and selector_name == "nms":
            later = [i for i in range(search["at"] + 1, max_iters + 1) if host.probe_role(i) == "active"]
            if later:
                warnings.append(
                    f"active probes after the search (iterations {later[:3]}{'...' if len(later) > 3 else ''}) "
                    f"prune the searched policy online; the recipe's later probes are control_iters"
                )
        if not any(host.should_probe(i) for i in range(1, max_iters + 1)):
            warnings.append(
                f"host {name!r} never probes within {max_iters} iterations: no R, and the "
                f"selector never runs. Set probe.schedule.interval or probe.schedule.iters."
            )
        if selector_name == "relevance_only" and name not in ("sensaug", "smartsampling_diff"):
            warnings.append(
                f"relevance_only ranks by the host's prior, and host {name!r} has a flat one: it "
                "will decline at every probe"
            )

    # The SA loop's knobs, which only the sensaug host reads.
    sa_cfg = cfg.get("sa")
    if sa_cfg is not None:
        if name != "sensaug":
            defaults = OmegaConf.load(os.path.join(CONFIG_DIR, "sa", "default.yaml"))
            changed = sorted(k for k in defaults if to_python(sa_cfg.get(k)) != to_python(defaults[k]))
            if changed:
                warnings.append(f"sa.{', sa.'.join(changed)} changed, but host {name!r} has no SA loop: ignored")
        elif host is not None:
            if bool(cfg.host.get("pre_train_round")) != bool(sa_cfg.init_sa):
                errors.append(
                    f"host.pre_train_round ({cfg.host.get('pre_train_round')}) and sa.init_sa ({sa_cfg.init_sa}) "
                    f"must be equal: the val round before training shifts every probe round by one. "
                    f"Set sa.init_sa; host.pre_train_round follows it"
                )
            if training and cfg.task.get("val_interval") is not None and int(cfg.task.val_interval) != int(host.round_interval):
                warnings.append(
                    f"task.val_interval is {cfg.task.val_interval}, but host 'sensaug' runs a round every "
                    f"{host.round_interval} iterations; the mmseg builder sets val_interval to the "
                    f"round interval for this host"
                )

    # Checkpoints kept past max_keep_ckpts, and the checkpoint a fork starts from.
    keep = to_python(cfg.get("keep_checkpoints")) or []
    if keep:
        interval, max_iters = checkpoint_interval(cfg), int(cfg.task.max_iters)
        bad = [n for n in keep if isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= max_iters]
        off_grid = [n for n in keep if n not in bad and n % interval and n != max_iters]
        if bad:
            errors.append(f"keep_checkpoints {bad} are not iterations in (0, {max_iters}]")
        if off_grid:
            errors.append(
                f"keep_checkpoints {off_grid} are not on the checkpoint grid (every {interval} "
                f"iterations): no checkpoint is saved there to keep"
            )
    if cfg.get("fork_from"):
        block = to_python(cfg.get("prune")) or {}
        if not block.get("hold_none_prob", True) and (block.get("ops") or block.get("random")):
            # The owner's rule for every host with a no-op outcome: a fork is a matched
            # comparison with the run it forks, and a growing no-op share would train the
            # pruned branch on less augmentation than its control.
            errors.append(
                "a fork's prune must hold the no-op share (prune.hold_none_prob=true): the pruned "
                "mass goes to the surviving ops, so the fork and its source augment as often"
            )
        try:
            at = fork_iter(cfg.fork_from)
        except ConfigError as error:
            errors.append(str(error))
        else:
            if not 0 < at < int(cfg.task.max_iters):
                errors.append(f"fork_from is at iteration {at}; a fork needs one inside (0, {cfg.task.max_iters})")

    if errors:
        raise ConfigError("the config doesn't fit together:\n  - " + "\n  - ".join(errors))
    return warnings
