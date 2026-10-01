"""name -> OpSpec across every family, and building a vocab from config.

A vocab is an ordered `{name: OpSpec}` dict. Its order is the order of R's axes,
so it is fixed by the config, never by set iteration.

Config (a plain mapping, e.g. loaded from YAML by the caller):

    families: [legacy_hsv, geometric, pil_autoaug]   # default: DEFAULT_FAMILIES
    ops: [lighter_R, rotate_pos, ...]                # instead of families, in this order
    exclude: [noise]                                 # dropped from either

`families` and `ops` are alternatives; give at most one. Every name in `ops` or
`exclude` must be a real op, so a typo raises rather than silently selecting or
dropping nothing. Excluding a real op that the selection does not contain is
fine.
"""

from typing import Dict, Iterable, Mapping, Optional

from .families import FAMILIES
from .op import OpSpec

# Sensaug's 32-op DIFF32_OPS, in its order: the vocabulary every Sensaug R was
# measured over, so a dropaug R built from the default vocab lines up with it.
DEFAULT_FAMILIES = ("legacy_hsv", "geometric", "pil_autoaug")

CONFIG_KEYS = frozenset({"families", "ops", "exclude"})


def _index(families: Mapping[str, Iterable[OpSpec]]) -> Dict[str, OpSpec]:
    ops: Dict[str, OpSpec] = {}
    for family, specs in families.items():
        for spec in specs:
            if spec.family != family:
                raise ValueError(f"{spec.name} is declared in {family} but says family={spec.family}")
            if spec.name in ops:
                raise ValueError(f"op name {spec.name!r} is declared twice")
            ops[spec.name] = spec
    return ops


OPS: Dict[str, OpSpec] = _index(FAMILIES)


def get_op(name: str) -> OpSpec:
    try:
        return OPS[name]
    except KeyError:
        raise KeyError(f"unknown op {name!r}; known ops: {sorted(OPS)}") from None


def _check_names(names: Iterable[str], key: str) -> None:
    unknown = [n for n in names if n not in OPS]
    if unknown:
        raise KeyError(f"config {key!r} names unknown ops {unknown}; known ops: {sorted(OPS)}")


def build_vocab(config: Optional[Mapping] = None) -> Dict[str, OpSpec]:
    """Build an ordered vocab from `config` (see module docstring). None gives
    the default: every op of DEFAULT_FAMILIES."""
    config = dict(config or {})
    unknown_keys = set(config) - CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"unknown config keys {sorted(unknown_keys)}; expected {sorted(CONFIG_KEYS)}")
    if "families" in config and "ops" in config:
        raise ValueError("give either 'families' or 'ops', not both")

    if "ops" in config:
        names = list(config["ops"])
        _check_names(names, "ops")
        if len(set(names)) != len(names):
            raise ValueError(f"config 'ops' repeats a name: {names}")
    else:
        families = list(config.get("families", DEFAULT_FAMILIES))
        unknown_families = [f for f in families if f not in FAMILIES]
        if unknown_families:
            raise KeyError(f"unknown families {unknown_families}; known: {list(FAMILIES)}")
        if len(set(families)) != len(families):
            raise ValueError(f"config 'families' repeats a family: {families}")
        names = [spec.name for family in families for spec in FAMILIES[family]]

    exclude = list(config.get("exclude", ()))
    _check_names(exclude, "exclude")
    dropped = set(exclude)

    vocab = {name: OPS[name] for name in names if name not in dropped}
    if not vocab:
        raise ValueError(f"config selects no ops: {config}")
    return vocab
