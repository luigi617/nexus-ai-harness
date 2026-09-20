from __future__ import annotations

from typing import Any, TypeVar

from core.phase import Phase
from harness.interception import InterceptorBinding
from protocols.interceptor import Interceptor
from protocols.plugin import Plugin

P = TypeVar("P", bound=Plugin)


class Registry:
    def __init__(self) -> None:
        self._plugins: list[Any] = []
        self._interceptors: list[InterceptorBinding] = []

    def add(self, plugin: object) -> None:
        if not hasattr(plugin, "kind"):
            raise TypeError(
                f"{type(plugin).__name__} is not a plugin (no 'kind' attribute)"
            )
        self._plugins.append(plugin)

    def add_interceptor(
        self, target: type[Plugin], phase: Phase, interceptor: Interceptor
    ) -> None:
        self._interceptors.append(InterceptorBinding(target, phase, interceptor))

    def interceptors(self, plugin: object, phase: Phase) -> list[Interceptor]:
        # kind gates the protocol family
        plugin_kind = getattr(plugin, "kind", None)
        return [
            b.interceptor
            for b in self._interceptors
            if b.phase is phase
            and plugin_kind == b.target.kind
            and isinstance(plugin, b.target)
        ]

    def interceptor_bindings(self) -> list[InterceptorBinding]:
        return list(self._interceptors)

    def get(self, cls: type[P]) -> P | None:
        for plugin in reversed(self._plugins):
            if getattr(plugin, "kind", None) == cls.kind:
                return plugin
        return None

    def all(self, cls: type[P]) -> list[P]:
        return [p for p in self._plugins if getattr(p, "kind", None) == cls.kind]

    def plugins(self) -> list[Any]:
        return list(self._plugins)

    def kinds(self) -> set[str]:
        return {p.kind for p in self._plugins}
