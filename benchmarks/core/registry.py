from __future__ import annotations

from benchmarks.core.benchmark import Benchmark
from benchmarks.core.config import build_config

_REGISTRY: dict[str, type[Benchmark]] = {}


def register(cls: type[Benchmark]) -> type[Benchmark]:
    """Class decorator that registers a benchmark under its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty class-level 'name'")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"benchmark name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_benchmark(name: str, overrides: dict[str, str] | None = None) -> Benchmark:
    """Instantiate the benchmark registered under ``name``.

    Args:
        name: The registered benchmark name.
        overrides: Optional ``field=value`` config overrides (from ``--set``),
            coerced onto the suite's config defaults.

    Returns:
        The configured benchmark instance.

    Raises:
        KeyError: If no benchmark is registered under ``name``.
        ValueError: If an override names an unknown field or an uncoercible value.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown benchmark {name!r}; registered: {known}") from None
    return cls(build_config(cls.config_type, overrides or {}))


def registered_benchmarks() -> dict[str, type[Benchmark]]:
    """A copy of the name -> class mapping, for listing."""
    return dict(_REGISTRY)
