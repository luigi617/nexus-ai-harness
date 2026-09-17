from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import ClassVar, Protocol, runtime_checkable

from protocols.context import Context
from protocols.plugin import Plugin


@runtime_checkable
class Loop(Plugin, Protocol):
    """
    A loop plugin. Drives a Session to completion and returns final text.
    """

    kind: ClassVar[str] = "loop"

    @abstractmethod
    def run(self, ctx: Context) -> str | Awaitable[str]:
        """
        Drive the session to completion.
        May be sync or ``async def`` — the harness adapts.
        """
