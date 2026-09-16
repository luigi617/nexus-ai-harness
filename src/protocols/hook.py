from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

from core.events import Event
from protocols.context import Context


@runtime_checkable
class Hook(Protocol):
    kind: ClassVar[str] = "hook"

    @abstractmethod
    def on(self, event: Event, ctx: Context) -> None: ...
