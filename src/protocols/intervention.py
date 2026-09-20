from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable

from protocols.context import Context


class Intervention(ABC):
    """An external command applied to a running agent at a safe checkpoint."""

    @abstractmethod
    def apply(self, ctx: Context) -> Awaitable[None] | None:
        """Mutate the run (e.g. inject a message).

        May be sync or ``async def`` — the harness adapts.
        """
