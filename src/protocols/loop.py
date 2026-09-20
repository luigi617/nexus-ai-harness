from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from protocols.context import Context
from protocols.plugin import Plugin


class Loop(Plugin):
    """A loop plugin. Drives a Session to completion and returns final text."""

    @abstractmethod
    def run(self, ctx: Context) -> str | Awaitable[str]:
        """Drive the session to completion.

        May be sync or ``async def`` — the harness adapts.
        """
