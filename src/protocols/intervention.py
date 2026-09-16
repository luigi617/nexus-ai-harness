from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from protocols.context import Context


@runtime_checkable
class Intervention(Protocol):
    """An external command applied to a running agent at a safe checkpoint."""

    @abstractmethod
    def apply(self, ctx: Context) -> Awaitable[None] | None:
        """Mutate the run (e.g. inject a message). May be sync or ``async def``
        — the harness adapts."""
