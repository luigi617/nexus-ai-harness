from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from core.events import Event
from core.message import Message
from core.subscription import Subscription
from protocols.plugin import Plugin

T = TypeVar("T")
P = TypeVar("P", bound=Plugin)


class Context(ABC):
    """The run-scoped view a plugin is handed to reach the session and registry."""

    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @property
    @abstractmethod
    def history(self) -> list[Message]: ...

    @property
    @abstractmethod
    def interrupted(self) -> bool: ...

    @abstractmethod
    def state(self, cls: type[T]) -> T: ...

    @abstractmethod
    def apply_interventions(self) -> Awaitable[None]: ...

    @abstractmethod
    def add_message(self, message: Message) -> None: ...

    @abstractmethod
    def emit(self, event: Event) -> None: ...

    @abstractmethod
    def on(
        self, event_type: type[Event], handler: Callable[[Event, Context], None]
    ) -> Subscription:
        """Subscribe ``handler`` to every future ``event_type`` event.

        A plugin typically calls this from its lifecycle ``start`` to react to
        events without being a standalone ``Hook``. The subscription is owned by
        the plugin whose bound method ``handler`` is, so it is removed
        automatically when that plugin is removed via ``NexusAIHarness.unuse``.

        Args:
            event_type: The event class to listen for; the handler fires on any
                instance of it, subclasses included.
            handler: Called with ``(event, ctx)`` each time a matching event is
                emitted. May be sync only, since emission is synchronous.

        Returns:
            A :class:`~core.subscription.Subscription` handle whose ``remove``
            cancels the subscription ahead of the owner being removed.
        """

    @abstractmethod
    def get(self, cls: type[P]) -> P | None: ...

    @abstractmethod
    def all(self, cls: type[P]) -> list[P]: ...

    @abstractmethod
    def invoke(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Awaitable[Any]:
        """Call plugin method ``fn`` with any interceptors wrapping it around it.

        ``fn`` may be sync or ``async def`` and is passed ``*args`` and
        ``**kwargs``.
        """

    @abstractmethod
    def fork(self, plugins: list[object] | None = None) -> Context:
        """Create a forked context."""
