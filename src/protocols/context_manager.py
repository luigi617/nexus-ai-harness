from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import ClassVar, Protocol, runtime_checkable

from core.message import Message
from protocols.context import Context


@runtime_checkable
class ContextManager(Protocol):
    """
    Owns everything about managing the context sent to the model
    """

    kind: ClassVar[str] = "context"

    @abstractmethod
    def process(
        self, history: list[Message], ctx: Context
    ) -> list[Message] | Awaitable[list[Message]]:
        """
        Transform the history before the model call.
        May be sync or ``async def`` — the harness adapts.
        """
