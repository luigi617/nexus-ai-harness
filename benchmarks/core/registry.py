from __future__ import annotations

from benchmarks.core.benchmark import Benchmark

_REGISTRY: dict[str, type[Benchmark]] = {}


def register(cls: type[Benchmark]) -> type[Benchmark]:
    """Class decorator that registers a benchmark under its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty class-level 'name'")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"benchmark name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_benchmark(name: str) -> Benchmark:
    """Instantiate the benchmark registered under ``name``."""
    try:
        return _REGISTRY[name]()
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown benchmark {name!r}; registered: {known}") from None


def registered_benchmarks() -> dict[str, type[Benchmark]]:
    """A copy of the name -> class mapping, for listing."""
    return dict(_REGISTRY)
