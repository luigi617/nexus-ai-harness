from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import ClassVar, Protocol, runtime_checkable

from protocols.context import Context


@runtime_checkable
class Tool(Protocol):
    kind: ClassVar[str] = "tool"

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict]

    @abstractmethod
    def run(self, arguments: dict, ctx: Context) -> str | Awaitable[str]:
        """
        Run the tool.
        May be sync or ``async def`` — the harness adapts.
        """
