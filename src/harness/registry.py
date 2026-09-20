from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TypeVar

from core.events import Event
from core.phase import Phase
from core.subscription import Subscription
from protocols.interceptor import Interceptor
from protocols.plugin import Plugin

P = TypeVar("P", bound=Plugin)
# Unbounded: lifecycle queries pass ``Lifecycle``, a mixin that is not a Plugin.
L = TypeVar("L")


class PluginStatus(Enum):
    """Where a registered plugin sits in its lifecycle."""

    REGISTERED = auto()  # added; start() has not run (or was stopped)
    STARTED = auto()  # start() ran and stop() has not
    FAILED = auto()  # start() raised; skipped by later starts until removed


@dataclass
class Registration:
    """A registered plugin together with its per-plugin registry state.

    Attributes:
        plugin: The registered plugin instance.
        status: The plugin's lifecycle status — the single source of truth for
            starting a late-added plugin once, stopping only what started, and
            not retrying a plugin whose ``start`` already failed.
    """

    plugin: Plugin
    status: PluginStatus = field(default=PluginStatus.REGISTERED)


class Registry:
    def __init__(self) -> None:
        self._entries: list[Registration] = []
        self._subscriptions: list[Subscription] = []

    def add(self, plugin: object) -> None:
        if not isinstance(plugin, Plugin):
            raise TypeError(
                f"{type(plugin).__name__} is not a plugin (does not subclass Plugin)"
            )
        if isinstance(plugin, Interceptor) and not isinstance(
            getattr(plugin, "target", None), type
        ):
            raise TypeError(
                f"{type(plugin).__name__} is an interceptor but declares no "
                "'target' plugin type"
            )
        self._entries.append(Registration(plugin))

    def _find(self, plugin: object) -> Registration | None:
        for entry in self._entries:
            if entry.plugin is plugin:
                return entry
        return None

    def set_status(self, plugin: object, status: PluginStatus) -> None:
        entry = self._find(plugin)
        if entry is not None:
            entry.status = status

    def status_of(self, plugin: object) -> PluginStatus | None:
        entry = self._find(plugin)
        return entry.status if entry is not None else None

    def is_started(self, plugin: object) -> bool:
        return self.status_of(plugin) is PluginStatus.STARTED

    def unstarted(self, cls: type[L]) -> list[L]:
        """Registered plugins of type ``cls`` eligible to start, in registration order.

        Excludes both started plugins and any whose ``start`` already failed.
        """
        return [
            e.plugin
            for e in self._entries
            if isinstance(e.plugin, cls) and e.status is PluginStatus.REGISTERED
        ]

    def started(self, cls: type[L]) -> list[L]:
        """Registered plugins of type ``cls`` already started, in registration order."""
        return [
            e.plugin
            for e in self._entries
            if isinstance(e.plugin, cls) and e.status is PluginStatus.STARTED
        ]

    def interceptors(self, plugin: object, phase: Phase) -> list[Callable[[Any], Any]]:
        """The ``before``/``after`` callbacks that wrap invoking ``plugin``.

        Returns the phase method of every registered interceptor whose ``target``
        type ``plugin`` is an instance of and that overrides that method, in
        registration order.
        """
        method = "before" if phase is Phase.BEFORE else "after"
        return [
            getattr(e.plugin, method)
            for e in self._entries
            if isinstance(e.plugin, Interceptor)
            and isinstance(plugin, e.plugin.target)
            and getattr(type(e.plugin), method) is not getattr(Interceptor, method)
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

    def remove_subscriptions(self, plugin: object) -> None:
        """Drop every event subscription owned by ``plugin``, keeping it registered."""
        self._subscriptions = [s for s in self._subscriptions if s.owner is not plugin]

    def remove(self, plugin: object) -> None:
        """Remove ``plugin`` and every registration it owns."""
        self._entries = [e for e in self._entries if e.plugin is not plugin]
        self._subscriptions = [s for s in self._subscriptions if s.owner is not plugin]

    def get(self, cls: type[P]) -> P | None:
        for entry in reversed(self._entries):
            if isinstance(entry.plugin, cls):
                return entry.plugin
        return None

    def all(self, cls: type[P]) -> list[P]:
        return [e.plugin for e in self._entries if isinstance(e.plugin, cls)]

    def plugins(self) -> list[Plugin]:
        return [e.plugin for e in self._entries]
