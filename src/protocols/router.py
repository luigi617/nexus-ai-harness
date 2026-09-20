from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from core.message import Message
from protocols.context import Context
from protocols.model import Model
from protocols.plugin import Plugin


class Router(Plugin):
    """Chooses which model handles a request when several are registered."""

    @abstractmethod
    def route(self, history: list[Message], ctx: Context) -> Model | Awaitable[Model]:
        """Pick a model, typically from ``ctx.all(Model)``.

        May be sync or ``async def`` — the harness adapts.
        """
