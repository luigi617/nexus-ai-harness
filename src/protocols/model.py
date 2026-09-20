from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from core.message import Message
from core.response import Response
from protocols.context import Context
from protocols.plugin import Plugin


class Model(Plugin):
    """A model backend."""

    provider: str = ""
    name: str = ""

    @abstractmethod
    def complete(
        self, history: list[Message], ctx: Context
    ) -> Response | Awaitable[Response]:
        """Return a completion.

        May be implemented as sync or ``async def`` — the harness adapts.
        """
