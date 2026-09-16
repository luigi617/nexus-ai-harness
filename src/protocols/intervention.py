from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from protocols.mediator import Context


@runtime_checkable
class Intervention(Protocol):
    """An external command applied to a running agent at a safe checkpoint —
    before the loop's next iteration. Posted via ``session.submit(...)`` and
    drained by the loop. Not a registered plugin: the caller constructs and
    submits instances directly.
    """

    @abstractmethod
    def apply(self, ctx: Context) -> Awaitable[None] | None:
        """Mutate the run (e.g. inject a message). May be sync or ``async def``
        — the harness adapts."""
