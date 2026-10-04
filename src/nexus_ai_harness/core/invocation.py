from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Deferred: protocols import core, so core can't import protocols at runtime.
    from nexus_ai_harness.protocols.plugin import Plugin


@dataclass
class Invocation:
    """One call to a plugin method, as seen by the interceptors wrapping it.

    Attributes:
        plugin: The plugin instance the call is being made on.
        method: The name of the method being invoked.
        args: Positional arguments passed to the method.
        kwargs: Keyword arguments passed to the method.
    """

    plugin: Plugin
    method: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvocationOutcome:
    """How an invocation ended, passed to interceptors' ``after``.

    Attributes:
        result: The value the method returned. ``None`` when it raised instead.
        error: The exception the method raised, including one raised by an
            earlier interceptor's ``before``. ``None`` when it returned normally.
    """

    result: Any = None
    error: BaseException | None = None
