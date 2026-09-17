from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

from core.events import Event
from protocols.context import Context
from protocols.plugin import Plugin


@runtime_checkable
class Hook(Plugin, Protocol):
    kind: ClassVar[str] = "hook"

    @abstractmethod
    def on(self, event: Event, ctx: Context) -> None: ...
