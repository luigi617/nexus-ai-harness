from __future__ import annotations

import asyncio

from harness.mediator import RunContext
from harness.registry import Registry
from harness.result import RunResult
from harness.session import Session
from services.runner import run_session


class GraphAIHarness:
    def __init__(self) -> None:
        self._registry = Registry()

    def use(self, plugin: object) -> GraphAIHarness:
        self._registry.add(plugin)
        return self

    async def run(
        self, user_input: str, *, session: Session | None = None
    ) -> RunResult:
        cur_session = session or Session()
        ctx = RunContext(cur_session, self._registry)
        output = await run_session(ctx, user_input)
        return RunResult(output=output, session=cur_session)

    def run_sync(self, user_input: str, *, session: Session | None = None) -> RunResult:
        """Synchronous entry point over run()."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(user_input, session=session))
        raise RuntimeError(
            "GraphAIHarness.run_sync() cannot be called from a running event "
            "loop; await run() instead."
        )
