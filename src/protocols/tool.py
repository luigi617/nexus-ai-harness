from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import ClassVar

from protocols.context import Context
from protocols.plugin import Plugin


class Tool(Plugin):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict]

    @abstractmethod
    def run(self, arguments: dict, ctx: Context) -> str | Awaitable[str]:
        """Run the tool.

        May be sync or ``async def`` — the harness adapts.
        """
