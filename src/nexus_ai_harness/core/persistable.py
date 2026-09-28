from __future__ import annotations

import dataclasses
import json
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_BY_NAME: dict[str, type] = {}
_NAME_OF: dict[type, str] = {}


class PersistenceWarning(UserWarning):
    """Session data that could not be saved or restored and was skipped."""


def persistable(name: str) -> Callable[[type[T]], type[T]]:
    """Opt a class into session persistence under a stable ``name``.

    Session state (``ctx.state(cls)``) and pending interventions are saved with
    a session snapshot only when their class carries this decorator; anything
    else is deliberately left out. The ``name`` is what lands on disk, so it
    must stay the same across releases even if the class is renamed or moved.

    A decorated class is serialized in one of two ways:

    * If it defines ``to_dict(self) -> dict`` and a ``from_dict(cls, data)``
      classmethod, those are used.
    * Otherwise it must be a dataclass whose fields hold JSON-native values
      (``str``, ``int``, ``float``, ``bool``, ``None``, lists, and dicts); it is
      saved with ``dataclasses.asdict`` and restored by passing the saved
      fields, ignoring any the class no longer declares, back to the class.

    Example::

        @persistable("my_plugin.counter")
        @dataclass
        class CounterState:
            calls: int = 0

    Args:
        name: The stable identifier stored in snapshots.

    Returns:
        A class decorator that registers the class and returns it unchanged.

    Raises:
        TypeError: If the class is neither a dataclass nor defines
            ``to_dict``/``from_dict``.
        ValueError: If ``name`` is empty or already taken by another class.
    """
    if not name or not name.strip():
        raise ValueError("persistable name must be a non-empty string")

    def register(cls: type[T]) -> type[T]:
        custom = hasattr(cls, "to_dict") and hasattr(cls, "from_dict")
        if not custom and not dataclasses.is_dataclass(cls):
            raise TypeError(
                f"{cls.__qualname__} must be a dataclass or define to_dict/from_dict "
                "to be persistable"
            )
        existing = _BY_NAME.get(name)
        # A module reload re-registers the "same" class; only a real clash is an error.
        if existing is not None and _qualified(existing) != _qualified(cls):
            raise ValueError(
                f"persistable name {name!r} is already used by {_qualified(existing)}"
            )
        if existing is not None:
            _NAME_OF.pop(existing, None)
        _BY_NAME[name] = cls
        _NAME_OF[cls] = name
        return cls

    return register


def persisted_name(cls: type) -> str | None:
    """Return the stable name ``cls`` was registered under, or ``None``."""
    return _NAME_OF.get(cls)


def persistable_class(name: str) -> type | None:
    """Return the class registered under ``name``, or ``None`` if unknown."""
    return _BY_NAME.get(name)


def dump(obj: object) -> dict | None:
    """Serialize a persistable instance to a JSON-safe dict.

    Args:
        obj: An instance of a class registered with :func:`persistable`.

    Returns:
        The instance's JSON-safe data, or ``None`` (with a
        :class:`PersistenceWarning`) if its class is not registered, its
        ``to_dict`` raises, or its data does not survive a JSON round-trip.
    """
    name = persisted_name(type(obj))
    if name is None:
        warnings.warn(
            f"skipping non-persistable {type(obj).__qualname__}",
            PersistenceWarning,
            stacklevel=2,
        )
        return None
    try:
        to_dict = getattr(obj, "to_dict", None)
        data = to_dict() if callable(to_dict) else dataclasses.asdict(obj)  # type: ignore[call-overload]
        json.dumps(data)  # prove it is JSON-safe now, not at write time
    except Exception as exc:  # a buggy to_dict must not sink the whole snapshot
        warnings.warn(
            f"skipping {name!r}: its state could not be serialized ({exc!r})",
            PersistenceWarning,
            stacklevel=2,
        )
        return None
    if not isinstance(data, dict):
        warnings.warn(
            f"skipping {name!r}: to_dict must return a dict",
            PersistenceWarning,
            stacklevel=2,
        )
        return None
    return data


def restore(name: str, data: Any) -> object | None:
    """Rebuild a persistable instance from the data :func:`dump` produced.

    Args:
        name: The stable name the data was saved under.
        data: The saved data.

    Returns:
        The rebuilt instance, or ``None`` (with a :class:`PersistenceWarning`)
        if ``name`` is not registered, the data no longer fits the class, or
        its ``from_dict`` raises.
    """
    cls = persistable_class(name)
    if cls is None:
        warnings.warn(
            f"cannot restore {name!r}: no class is registered under that name",
            PersistenceWarning,
            stacklevel=2,
        )
        return None
    try:
        if not isinstance(data, dict):
            raise TypeError(f"expected a dict, got {type(data).__name__}")
        from_dict = getattr(cls, "from_dict", None)
        if callable(from_dict) and hasattr(cls, "to_dict"):
            return from_dict(data)
        known = {f.name for f in dataclasses.fields(cls) if f.init}
        return cls(**{k: v for k, v in data.items() if k in known})
    except Exception as exc:  # e.g. KeyError from from_dict on an older snapshot
        warnings.warn(
            f"cannot restore {name!r}: {exc!r}", PersistenceWarning, stacklevel=2
        )
        return None


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"
