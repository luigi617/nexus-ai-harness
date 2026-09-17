from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import ClassVar

from core.events import Event
from protocols.context import Context
from protocols.hook import Hook

EventHandler = Callable[..., None]


def _as_binary(handler: EventHandler) -> Callable[[Event, Context], None]:
    """Adapt a one- or two-argument handler to a ``(event, ctx)`` callable."""
    params = [
        p
        for p in inspect.signature(handler).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
    ]
    wants_ctx = any(p.kind == p.VAR_POSITIONAL for p in params) or len(params) >= 2
    if wants_ctx:
        return lambda event, ctx: handler(event, ctx)
    return lambda event, ctx: handler(event)


class CallbackHook(Hook):
    """A Hook that fans emitted events out to type-keyed callbacks.

    Backs :meth:`~harness.harness.NexusAIHarness.on`. A handler registered for
    an event type fires for that type and any subclass, so a handler on
    :class:`~core.events.Event` observes every event while one on
    ``ToolCallStarted`` observes only that. Handlers are synchronous and may
    accept either ``(event)`` or ``(event, ctx)``.
    """

    kind: ClassVar[str] = "hook"

    def __init__(self) -> None:
        self._handlers: list[tuple[type[Event], Callable[[Event, Context], None]]] = []

    def register(self, event_type: type[Event], handler: EventHandler) -> None:
        """Add ``handler``, to be called on each emitted ``event_type`` event."""
        self._handlers.append((event_type, _as_binary(handler)))

    def on(self, event: Event, ctx: Context) -> None:
        for event_type, handler in self._handlers:
            if isinstance(event, event_type):
                handler(event, ctx)
