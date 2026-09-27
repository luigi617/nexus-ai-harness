from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from core.message import Message
from core.response import Response
from protocols.context import Context
from protocols.plugin import Plugin
from protocols.tool import Tool


class Model(Plugin):
    """A model backend."""

    provider: str = ""
    name: str = ""

    @abstractmethod
    def complete(
        self, history: list[Message], tools: list[Tool], ctx: Context
    ) -> Response | Awaitable[Response]:
        """Return a completion.

        The loop assembles ``tools`` (static plus any contributed by a
        ``ToolProvider``) and hands the model exactly the set it may call, so the
        advertised tools always match what the loop can dispatch. May be
        implemented as sync or ``async def`` — the harness adapts.

        Args:
            history: The conversation so far, oldest message first.
            tools: The tools the model may call this turn; empty for none.
            ctx: The active run context.
        """
