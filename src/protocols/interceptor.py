from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import ClassVar, Protocol, runtime_checkable

from protocols.context import Context
from protocols.plugin import Plugin


@runtime_checkable
class Interceptor(Plugin, Protocol):
    """Runs a side effect around another plugin's invocation."""

    kind: ClassVar[str] = "interceptor"

    @abstractmethod
    def run(self, ctx: Context) -> Awaitable[None] | None:
        """React to the target's invocation.

        May be sync or ``async def`` — the harness adapts.
        """
