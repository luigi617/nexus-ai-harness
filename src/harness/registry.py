from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from core.events import Event
from core.phase import Phase
from core.subscription import Subscription
from harness.interception import InterceptorBinding
from protocols.interceptor import Interceptor
from protocols.plugin import Plugin

P = TypeVar("P", bound=Plugin)


class Registry:
    def __init__(self) -> None:
        self._plugins: list[Plugin] = []
        self._interceptors: list[InterceptorBinding] = []
        self._subscriptions: list[Subscription] = []

    def add(self, plugin: object) -> None:
        if not isinstance(plugin, Plugin):
            raise TypeError(
                f"{type(plugin).__name__} is not a plugin (does not subclass Plugin)"
            )
        self._plugins.append(plugin)

    def add_interceptor(
        self, target: type[Plugin], phase: Phase, interceptor: Interceptor
    ) -> None:
        self._interceptors.append(InterceptorBinding(target, phase, interceptor))

    def interceptors(self, plugin: object, phase: Phase) -> list[Interceptor]:
        return [
            b.interceptor
            for b in self._interceptors
            if b.phase is phase and isinstance(plugin, b.target)
        ]

    def interceptor_bindings(self) -> list[InterceptorBinding]:
        return list(self._interceptors)

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

        Drops the plugin itself, its event subscriptions, and any interceptor
        bindings it provides. A plugin or registration that is not present is
        ignored, so removal is idempotent.
        """
        self._plugins = [p for p in self._plugins if p is not plugin]
        self._subscriptions = [s for s in self._subscriptions if s.owner is not plugin]
        self._interceptors = [
            b for b in self._interceptors if b.interceptor is not plugin
        ]

    def get(self, cls: type[P]) -> P | None:
        for plugin in reversed(self._plugins):
            if isinstance(plugin, cls):
                return plugin
        return None

    def all(self, cls: type[P]) -> list[P]:
        return [p for p in self._plugins if isinstance(p, cls)]

    def plugins(self) -> list[Plugin]:
        return list(self._plugins)
