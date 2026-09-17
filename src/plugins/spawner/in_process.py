from __future__ import annotations

import asyncio

from core.spawn import SpawnState
from protocols.context import Context
from protocols.spawner import Spawner
from services.runner import run_session


class InProcessSpawner(Spawner):
    """Run each subagent on the current event loop.

    Bounded by a global semaphore so fan-out can't overwhelm the model with
    concurrent calls.
    """

    def __init__(self, max_concurrent: int = 5, max_depth: int = 2) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max_depth = max_depth

    async def run(self, child_ctx: Context, task: str) -> str:
        if child_ctx.state(SpawnState).depth > self._max_depth:
            return f"error: max subagent depth ({self._max_depth}) exceeded"
        async with self._sem:
            return await run_session(child_ctx, task)
