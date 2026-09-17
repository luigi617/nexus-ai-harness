from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import ClassVar, Protocol, runtime_checkable

from core.message import Message
from core.response import Response
from protocols.context import Context
from protocols.plugin import Plugin


@runtime_checkable
class Model(Plugin, Protocol):
    """
    A model backend.
    """

    kind: ClassVar[str] = "model"

    provider: str = ""
    name: str = ""

    @abstractmethod
    def complete(
        self, history: list[Message], ctx: Context
    ) -> Response | Awaitable[Response]:
        """Return a completion.
        May be implemented as sync or ``async def`` — the harness adapts.
        """
