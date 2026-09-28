from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.plugin import Plugin


class Spawner(Plugin):
    """Runs a subagent's child context to completion and returns the distilled result.

    May be sync or ``async def`` — the harness adapts.
    """

    @abstractmethod
    def run(self, child_ctx: Context, task: str) -> str | Awaitable[str]:
        """Run ``task`` on the given child context and return its final text."""
