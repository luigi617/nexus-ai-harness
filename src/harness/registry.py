from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from core.events import Event
from core.phase import Phase
from core.subscription import Subscription
from protocols.interceptor import Interceptor
from protocols.plugin import Plugin

P = TypeVar("P", bound=Plugin)


class Registry:
    def __init__(self) -> None:
        self._plugins: list[Plugin] = []
        self._subscriptions: list[Subscription] = []

    def add(self, plugin: object) -> None:
        if not isinstance(plugin, Plugin):
            raise TypeError(
                f"{type(plugin).__name__} is not a plugin (does not subclass Plugin)"
            )
        if isinstance(plugin, Interceptor) and not hasattr(plugin, "target"):
            raise TypeError(
                f"{type(plugin).__name__} is an interceptor but declares no "
                "'target' plugin type"
            )
        self._plugins.append(plugin)

    def interceptors(self, plugin: object, phase: Phase) -> list[Callable[[Any], Any]]:
        """The ``before``/``after`` callbacks that wrap invoking ``plugin``.

        Returns the phase method of every registered interceptor whose ``target``
        type ``plugin`` is an instance of and that overrides that method, in
        registration order.
        """
        method = "before" if phase is Phase.BEFORE else "after"
        return [
            getattr(p, method)
            for p in self._plugins
            if isinstance(p, Interceptor)
            and isinstance(plugin, p.target)
            and getattr(type(p), method) is not getattr(Interceptor, method)
        ]

    def subscribe(
        self, event_type: type[Event], handler: Callable[[Event, Any], None]
    ) -> Subscription:
        """Register ``handler`` to fire on every future ``event_type`` event.

        Ownership is inferred from the handler: a bound method is attributed to
        its instance, so :meth:`remove` drops the subscription with that plugin.
        """
        owner = getattr(handler, "__self__", None)
        subscription = Subscription(event_type, handler, owner, self._unsubscribe)
        self._subscriptions.append(subscription)
        return subscription

    def _unsubscribe(self, subscription: Subscription) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    def subscribers(self, event: Event) -> list[Subscription]:
        """The subscriptions matching ``event``, as a fresh list safe to mutate."""
        return [s for s in self._subscriptions if s.matches(event)]

    def subscriptions(self) -> list[Subscription]:
        return list(self._subscriptions)

    def remove(self, plugin: object) -> None:
        """Remove ``plugin`` and every registration it owns.

        Drops the plugin itself and its event subscriptions. Because an
        interceptor is itself a plugin, removing one also stops it wrapping its
        target. A plugin or registration that is not present is ignored, so
        removal is idempotent.
        """
        self._plugins = [p for p in self._plugins if p is not plugin]
        self._subscriptions = [s for s in self._subscriptions if s.owner is not plugin]

    def get(self, cls: type[P]) -> P | None:
        for plugin in reversed(self._plugins):
            if isinstance(plugin, cls):
                return plugin
        return None

    def all(self, cls: type[P]) -> list[P]:
        return [p for p in self._plugins if isinstance(p, cls)]

    def plugins(self) -> list[Plugin]:
        return list(self._plugins)
