from __future__ import annotations

import asyncio
import contextlib
from types import TracebackType

from core.invoke import call
from core.phase import Phase
from harness.context import RunContext
from harness.registry import Registry
from harness.result import RunResult
from harness.session import Session
from harness.validation import describe_registry, validate_registry
from protocols.interceptor import Interceptor
from protocols.lifecycle import Lifecycle
from protocols.plugin import Plugin
from services.runner import run_session


class NexusAIHarness:
    def __init__(self) -> None:
        self._registry = Registry()
        self._started = False

    def use(self, plugin: object) -> NexusAIHarness:
        self._registry.add(plugin)
        return self

    def use_before(
        self, target: type[Plugin], interceptor: Interceptor
    ) -> NexusAIHarness:
        """Register ``interceptor`` to run before each invocation of ``target``.

        A lifecycle-based alternative to observing an ``Event``: whenever the
        harness invokes a ``target`` plugin (e.g. ``Model``, ``Tool``,
        ``Loop``), the interceptor's ``run(ctx)`` fires first. Neither plugin
        needs to know about the other. Chainable.

        Args:
            target: The plugin type to wrap (a protocol such as ``Model``).
            interceptor: The :class:`~protocols.interceptor.Interceptor` to run.
        """
        self._registry.add_interceptor(target, Phase.BEFORE, interceptor)
        return self

    def use_after(
        self, target: type[Plugin], interceptor: Interceptor
    ) -> NexusAIHarness:
        """Register ``interceptor`` to run after each invocation of ``target``.

        The mirror of :meth:`use_before`; ``run(ctx)`` fires once the ``target``
        invocation returns. Chainable.

        Args:
            target: The plugin type to wrap (a protocol such as ``Model``).
            interceptor: The :class:`~protocols.interceptor.Interceptor` to run.
        """
        self._registry.add_interceptor(target, Phase.AFTER, interceptor)
        return self

    def validate(self) -> NexusAIHarness:
        """Validate that every plugin's declared ``requires`` are satisfied.

        Raises :class:`~harness.validation.MissingDependencyError` (with a
        rendered dependency tree) when a dependency is missing. Returns ``self``
        so it can be chained after ``use(...)``.
        """
        validate_registry(self._registry)
        return self

    def describe_dependencies(self) -> str:
        """Render the dependency tree of every plugin that declares ``requires``.

        Descriptive only — never raises.
        """
        return describe_registry(self._registry)

    async def start(self) -> NexusAIHarness:
        """Initialize every registered lifecycle plugin, in registration order.

        Calls ``start(ctx)`` on each registered ``Lifecycle`` plugin (see
        :class:`~protocols.lifecycle.Lifecycle`), passing a context over the
        registry so setup can resolve other plugins. Register dependencies
        before dependents so each is initialized after what it needs. If any
        ``start`` raises, the plugins already started are rolled back (their
        ``stop`` runs) before the error propagates. Idempotent: a second call
        before :meth:`stop` does nothing. :meth:`run` calls this automatically
        but never auto-stops, so callers release resources via :meth:`stop` or
        by using the harness as an ``async with`` block.

        Returns:
            The harness itself, so the call can be awaited and chained.
        """
        if self._started:
            return self
        # Set before the first await so a concurrent run()/start() can't double-init.
        self._started = True
        ctx = RunContext(Session(), self._registry)
        started: list[Lifecycle] = []
        try:
            for plugin in self._registry.plugins():
                if isinstance(plugin, Lifecycle):
                    await call(plugin.start, ctx)
                    started.append(plugin)
        except BaseException:
            self._started = False
            for started_plugin in reversed(started):  # roll back what started
                # Best-effort rollback; surface the original start failure.
                with contextlib.suppress(Exception):
                    await call(started_plugin.stop)
            raise
        return self

    async def stop(self) -> None:
        """Release every lifecycle plugin, in reverse registration order.

        Calls ``stop()`` on each registered ``Lifecycle`` plugin so a plugin is
        torn down before its dependencies. Every ``stop`` runs even if an earlier
        one raises; the first exception is re-raised once all have been
        attempted, so one plugin's failure can't leak another's resources.
        A ``BaseException`` such as ``CancelledError`` propagates immediately to
        preserve cooperative cancellation. Idempotent: a no-op if the harness
        was never started.
        """
        if not self._started:
            return
        self._started = False
        first_error: Exception | None = None
        for plugin in reversed(self._registry.plugins()):
            if isinstance(plugin, Lifecycle):
                try:
                    await call(plugin.stop)
                except Exception as exc:  # keep tearing the rest down
                    first_error = first_error or exc
        if first_error is not None:
            raise first_error

    async def __aenter__(self) -> NexusAIHarness:
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def run(
        self, user_input: str, *, session: Session | None = None
    ) -> RunResult:
        await self.start()  # guarantee lifecycle plugins are initialized
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
