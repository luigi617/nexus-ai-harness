from __future__ import annotations

import asyncio

from harness.context import RunContext
from harness.registry import Registry
from harness.result import RunResult
from harness.session import Session
from harness.validation import describe_registry, validate_registry
from services.runner import run_session


class NexusAIHarness:
    def __init__(self) -> None:
        self._registry = Registry()

    def use(self, plugin: object) -> NexusAIHarness:
        self._registry.add(plugin)
        return self

    def validate(self) -> NexusAIHarness:
        """Check that every registered plugin's declared ``requires`` are
        satisfied by another registered plugin.

        Raises :class:`~harness.validation.MissingDependencyError` (with a
        rendered dependency tree) when a dependency is missing.

        returns ``self`` so it can be chained after ``use(...)``.
        """
        validate_registry(self._registry)
        return self

    def describe_dependencies(self) -> str:
        """Render the dependency tree of every plugin that declares
        ``requires``. Descriptive only — never raises."""
        return describe_registry(self._registry)

    async def run(
        self, user_input: str, *, session: Session | None = None
    ) -> RunResult:
        cur_session = session or Session()
        cur_session.clear_interrupt()  # a new turn isn't pre-interrupted
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
            "NexusAIHarness.run_sync() cannot be called from a running event "
            "loop; await run() instead."
        )
