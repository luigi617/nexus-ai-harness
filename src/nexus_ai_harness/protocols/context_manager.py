from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from nexus_ai_harness.core.message import Message
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.plugin import Plugin


class ContextManager(Plugin):
    """Owns everything about managing the context sent to the model."""

    @abstractmethod
    def process(
        self, history: list[Message], ctx: Context
    ) -> list[Message] | Awaitable[list[Message]]:
        """Transform the history before the model call.

        May be sync or ``async def`` — the harness adapts.
        """
