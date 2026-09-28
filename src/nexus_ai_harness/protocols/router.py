from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from nexus_ai_harness.core.message import Message
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.plugin import Plugin


class Router(Plugin):
    """Chooses which model handles a request when several are registered."""

    @abstractmethod
    def route(self, history: list[Message], ctx: Context) -> Model | Awaitable[Model]:
        """Pick a model, typically from ``ctx.all(Model)``.

        May be sync or ``async def`` — the harness adapts.
        """
