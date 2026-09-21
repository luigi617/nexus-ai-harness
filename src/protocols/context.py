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
        the plugin whose ``start`` is running (or, outside ``start``, the
        instance a bound-method ``handler`` belongs to), so it is removed
        automatically when that plugin is removed via ``NexusAIHarness.unuse``.
        A subscription made outside ``start`` with a non-bound handler has no
        owner and is treated as cross-cutting: it survives ``unuse`` and must be
        cancelled via the returned handle's ``remove``.

        Args:
            event_type: The event class to listen for; the handler fires on any
                instance of it, subclasses included.
            handler: Called with ``(event, ctx)`` each time a matching event is
                emitted. Must be synchronous; an async handler raises.

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
    def fork(
        self,
        plugins: list[Plugin] | None = None,
        overrides: dict[type, Plugin] | None = None,
    ) -> Context:
        """Create an isolated child context.

        The child runs on its own fresh session, so its history and state never
        touch this one's; it inherits this context's plugins unless ``plugins``
        restricts the set.

        Args:
            plugins: The exact plugins the child sees; ``None`` inherits all of
                this context's. Pass a subset to hand a subagent only part of
                the parent's capabilities.
            overrides: A ``target -> replacement`` mapping. Each target is a
                plugin type matched by ``isinstance``: a protocol base swaps out
                every inherited implementer, while a concrete plugin class swaps
                only instances of that class. Matching plugins are dropped and
                the replacement registered in their place, so a subagent can
                share infrastructure while swapping one capability (e.g. its
                memory) even for protocols the parent holds several of.

        Returns:
            The child context, on a new session.
        """
