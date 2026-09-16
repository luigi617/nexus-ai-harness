from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import ClassVar, Protocol, runtime_checkable

from core.message import Message
from protocols.context import Context
from protocols.model import Model


@runtime_checkable
class Router(Protocol):
    """Chooses which model handles a request when several are registered"""

    kind: ClassVar[str] = "router"

    @abstractmethod
    def route(self, history: list[Message], ctx: Context) -> Model | Awaitable[Model]:
        """Pick a model, typically from ``ctx.all(Model)``. May be sync or
        ``async def`` — the harness adapts."""
