from __future__ import annotations

from typing import TypeVar

from core.phase import Phase
from harness.interception import InterceptorBinding
from protocols.interceptor import Interceptor
from protocols.plugin import Plugin

P = TypeVar("P", bound=Plugin)


class Registry:
    def __init__(self) -> None:
        self._plugins: list[Plugin] = []
        self._interceptors: list[InterceptorBinding] = []

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

    def get(self, cls: type[P]) -> P | None:
        for plugin in reversed(self._plugins):
            if isinstance(plugin, cls):
                return plugin
        return None

    def all(self, cls: type[P]) -> list[P]:
        return [p for p in self._plugins if isinstance(p, cls)]

    def plugins(self) -> list[Plugin]:
        return list(self._plugins)
