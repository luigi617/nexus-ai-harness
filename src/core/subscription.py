from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.events import Event


class Subscription:
    """A handle to an owned event subscription.

    Returned by ``Context.on``. It ties an event handler to the plugin that
    registered it, so the harness can drop the handler automatically when that
    plugin is removed. Call :meth:`remove` to cancel it earlier by hand.

    Attributes:
        event_type: The event class the subscription fires on; an event matches
            when it is an instance of this type.
        handler: The callable invoked with ``(event, ctx)`` on a match.
        owner: The plugin the subscription belongs to, or ``None`` when the
            handler is not a bound method and so has no inferable owner.
    """

    def __init__(
        self,
        event_type: type[Event],
        handler: Callable[[Event, Any], None],
        owner: object | None,
        on_remove: Callable[[Subscription], None],
    ) -> None:
        self.event_type = event_type
        self.handler = handler
        self.owner = owner
        self._on_remove = on_remove
        self._active = True

    def matches(self, event: Event) -> bool:
        """Whether ``event`` should be dispatched to this subscription."""
        return isinstance(event, self.event_type)

    def remove(self) -> None:
        """Cancel the subscription; a no-op if it was already removed."""
        if self._active:
            self._active = False
            self._on_remove(self)
