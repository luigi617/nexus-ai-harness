from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass
class BenchmarkConfig:
    """Base for a benchmark's run configuration.

    Subclass it with typed fields and defaults — the fields are the suite's
    tuning knobs (dataset split, step budget, timeout, ...). Defaults must make
    the suite runnable with no configuration; a run overrides individual fields
    via the CLI ``--set key=value`` (see :func:`build_config`). Credentials and
    cache locations stay in the environment, not here.
    """


def build_config(
    config_type: type[BenchmarkConfig], overrides: dict[str, Any]
) -> BenchmarkConfig:
    """Build a config instance from its defaults plus overrides.

    A string override is coerced to the annotated type of its field, so ``"40"``
    becomes ``int`` ``40`` and ``"false"`` becomes ``bool`` ``False`` (the
    ``--set`` path). An already-typed value — an ``int``, ``bool``, ``list``, or
    ``None`` from a parsed YAML file — is applied as-is. Fields not overridden
    keep their dataclass default.

    Args:
        config_type: The :class:`BenchmarkConfig` subclass to instantiate.
        overrides: Field-name to value pairs; string values are coerced, native
            values are applied directly.

    Returns:
        A ``config_type`` instance with the overrides applied.

    Raises:
        ValueError: If an override names an unknown field, targets an
            unsupported field type, or holds a value that can't be coerced.
    """
    if not overrides:
        return config_type()
    hints = get_type_hints(config_type)
    valid = {f.name for f in fields(config_type)}
    kwargs: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in valid:
            raise ValueError(f"unknown config key {key!r}; valid keys: {sorted(valid)}")
        # --set gives strings (coerce); YAML gives native types (apply as-is).
        kwargs[key] = (
            _coerce(value, hints[key], key) if isinstance(value, str) else value
        )
    return config_type(**kwargs)


def load_config_file(path: str | Path, benchmark: str) -> dict[str, Any]:
    """Load the ``benchmark`` section from a YAML config file.

    The file maps each benchmark name to its config fields, so one file can
    configure every suite::

        tau-bench:
          env: airline
          max_steps: 40
        swe-bench:
          dataset: princeton-nlp/SWE-bench_Lite

    Values are native YAML types and are applied by :func:`build_config` without
    string coercion.

    Args:
        path: The YAML file to read.
        benchmark: The benchmark whose section to return.

    Returns:
        The benchmark's field-to-value mapping, or ``{}`` if it has no section.

    Raises:
        ValueError: If the file's top level, or the benchmark's section, is not
            a mapping.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must map benchmark -> settings")
    section = data.get(benchmark, {})
    if not isinstance(section, dict):
        raise ValueError(f"{path}: section {benchmark!r} must be a mapping")
    return section


def _coerce(raw: str, hint: Any, key: str) -> Any:
    """Coerce ``raw`` to the field type ``hint`` (unwrapping ``X | None``)."""
    if get_origin(hint) in (Union, UnionType):
        if raw.strip().lower() in {"", "none", "null"}:
            return None
        # Coerce to the first non-None member of the union (e.g. str in str|None).
        hint = next(a for a in get_args(hint) if a is not type(None))
    origin = get_origin(hint)
    if origin in (list, tuple):
        args = get_args(hint)
        elem_type = args[0] if args else str
        parts = [p.strip() for p in raw.split(",")]
        items = [_coerce(p, elem_type, key) for p in parts if p]
        return items if origin is list else tuple(items)
    if hint is bool:
        low = raw.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{key}: expected a boolean, got {raw!r}")
    if hint is int:
        return int(raw)
    if hint is float:
        return float(raw)
    if hint is str:
        return raw
    raise ValueError(f"{key}: cannot set a field of type {hint!r} via --set")
