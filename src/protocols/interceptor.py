from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from protocols.context import Context
from protocols.plugin import Plugin


class Interceptor(Plugin):
    """Runs a side effect around another plugin's invocation."""

    @abstractmethod
    def run(self, ctx: Context) -> Awaitable[None] | None:
        """React to the target's invocation.

        May be sync or ``async def`` — the harness adapts.
        """
