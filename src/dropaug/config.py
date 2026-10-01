"""Experiment configs: compose ``configs/`` with Hydra, check that the groups fit
together, and build what each group describes.

    compose(overrides)       configs/train.yaml + Hydra overrides -> a resolved DictConfig
    check(cfg, training=)    cross-group checks; raises ConfigError, returns warnings
    save(cfg, work_dir)      <work_dir>/dropaug.yaml, the file probe.py / eval.py rebuild from
    load(work_dir)           ... and back, with dotted overrides on top
    vocab_of / host_of / selector_spec / magnitudes_of / run_name

One experiment is one choice per group (``GROUPS``). What a group's keys mean is in
its yaml files. What this module adds is the rules that span groups, checked before
anything is built, so a bad combination fails in seconds rather than hours into a run:

- the dataset's ``kind`` is one the task's backend serves (seg for mmseg, cls for cls),
  and an ``eval_only`` dataset (ACDC, IDD) is never trained on
- the vocab is the host's vocabulary. The sensaug and uniform hosts take it as theirs;
  the torchvision hosts have a fixed one, and the vocab must equal it
- training augments only with the hosts and ops ``tasks.augment`` can apply
- the selector builds with its keys, and the control (``selector=none``) keeps lam 0
- the probe magnitude is valid for every op of the vocab, and ``probe.magnitude_mode``
  is one of ``sa.MAGNITUDE_MODES``
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

import inspect
import os
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
    "group_options",
    "host_of",
    "load",
    "magnitudes_of",
    "run_name",
    "save",
    "selector_spec",
    "task_for_backbone",
    "to_python",
    "vocab_of",
]

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")

#: The config groups one experiment chooses from, in configs/train.yaml's order.
#: ``vocab`` is chosen by the host's own defaults unless overridden. ``sa`` is read only
#: by the sensaug host.
GROUPS = ("cluster", "task", "dataset", "host", "vocab", "selector", "probe", "sa")

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
    """``<host>_<selector>[-l<lam>]_<dataset>_<backbone>_s<seed>``, the name
    slurm/train.sbatch gives a run directory."""
    selector = cfg.choices.get("selector", cfg.selector.name)
    arm = "none" if cfg.selector.get("control") else f"{selector}-l{float(cfg.selector.lam):g}"
    host = cfg.choices.get("host", cfg.host.name)
    return f"{host}_{arm}_{cfg.dataset.name}_{cfg.task.backbone}_s{cfg.seed}"


# --- building ---------------------------------------------------------------------------


def vocab_of(cfg: DictConfig, vocab: Optional[DictConfig] = None) -> dict:
    """The vocab (``vocab.build_vocab``): the ops the host samples, the probe measures
    and the eval scores."""
    from vocab import build_vocab

    return build_vocab(to_python(cfg.vocab if vocab is None else vocab))


def _takes_vocabulary(name: str) -> bool:
    from hosts import HOSTS

    return "vocabulary" in inspect.signature(HOSTS[name].__init__).parameters


def host_kwargs(cfg: DictConfig) -> dict:
    """``build_host`` kwargs, with the vocab passed in to the hosts that take one."""
    kwargs = to_python(cfg.host)
    kwargs.pop("vocab", None)  # the vocab is passed as `vocabulary`, never as a host key
    name = kwargs.pop("name")
    if _takes_vocabulary(name):
        kwargs["vocabulary"] = list(vocab_of(cfg))
    return {"name": name, **kwargs}


def host_of(cfg: DictConfig):
    """The host the config describes (checked against the vocab by ``check``)."""
    from hosts import build_host

    kwargs = host_kwargs(cfg)
    return build_host(kwargs.pop("name"), **kwargs)


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


def check(cfg: DictConfig, *, training: bool) -> List[str]:
    """Raise ConfigError on groups that don't fit together; return the warnings (things
    that run, but not the way the config's names suggest). ``training`` adds the
    checks only a training run needs."""
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

    # The host, and its vocabulary against the vocab.
    host = None
    name = cfg.host.name
    if name not in HOSTS:
        errors.append(f"host {name!r} is not one of {sorted(HOSTS)}")
    else:
        try:
            ops = vocab_of(cfg)
            host = host_of(cfg)
        except (KeyError, ValueError, TypeError) as error:
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

    if training and host is not None:
        if cfg.augment:
            from tasks.augment import check_trainable

            try:
                check_trainable(host, vocab_of(cfg))
            except NotImplementedError as error:
                errors.append(f"{error} (set augment=false to train clean and probe anyway)")
        max_iters = int(cfg.task.max_iters)
        if not any(host.should_probe(i) for i in range(1, max_iters + 1)):
            warnings.append(
                f"host {name!r} never probes within {max_iters} iterations: no R, and the "
                f"selector never runs. Set probe.schedule.interval or probe.schedule.iters."
            )
        if selector_name == "relevance_only" and name != "sensaug":
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

    if errors:
        raise ConfigError("the config doesn't fit together:\n  - " + "\n  - ".join(errors))
    return warnings
