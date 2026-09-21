from __future__ import annotations

import asyncio
import contextlib
from types import TracebackType

from core.invoke import call
from harness.context import RunContext
from harness.graph import HarnessGraph, build_graph
from harness.registry import PluginStatus, Registry
from harness.result import RunResult
from harness.session import Session
from harness.validation import describe_registry, validate_registry
from protocols.lifecycle import Lifecycle
from protocols.plugin import Plugin
from services.runner import run_session


class NexusAIHarness:
    def __init__(self) -> None:
        self._registry = Registry()
        self._started = False
        # Serializes start()/stop() so concurrent run()s await one in-flight init.
        self._lifecycle_lock = asyncio.Lock()

    def use(self, plugin: Plugin) -> NexusAIHarness:
        """Register ``plugin``."""
        self._registry.add(plugin)
        return self

    async def unuse(self, plugin: Plugin) -> NexusAIHarness:
        """Remove ``plugin`` and automatically drop every registration it owns."""
        # TODO: drain in-flight runs before teardown so unuse()/stop() are safe
        # to call concurrently with run().
        async with self._lifecycle_lock:
            if isinstance(plugin, Lifecycle) and self._registry.is_started(plugin):
                await call(plugin.stop)
            self._registry.remove(plugin)
        return self

    def clone(self) -> NexusAIHarness:
        """Return an unstarted copy sharing this harness's plugin instances."""
        twin = type(self)()
        twin._registry = self._registry.clone()
        return twin

    def replace(self, capability: type, plugin: Plugin) -> NexusAIHarness:
        """Swap the registered provider(s) of ``capability`` for ``plugin``."""
        if self._started:
            raise RuntimeError("cannot replace plugins on a started harness")
        self._registry.replace(capability, plugin)
        return self

    def validate(self) -> NexusAIHarness:
        """Validate that every plugin's declared ``requires`` are satisfied.

        Raises :class:`~harness.validation.MissingDependencyError` (with a
        rendered dependency tree) when a dependency is missing. Returns ``self``
        so it can be chained after ``use(...)``.
        """
        validate_registry(self._registry)
        return self

    def graph(self) -> HarnessGraph:
        """Return the harness's composition as a first-class dependency graph."""
        return build_graph(self._registry, name=type(self).__name__)

    def describe_dependencies(self) -> str:
        """Render the dependency tree of every plugin that declares ``requires``.

        Descriptive only — never raises.
        """
        return describe_registry(self._registry)

    async def start(self) -> NexusAIHarness:
        """Initialize every not-yet-started lifecycle plugin, in registration order.

        Returns:
            The harness itself, so the call can be awaited and chained.
        """
        async with self._lifecycle_lock:
            session = Session()
            newly: list[Lifecycle] = []
            current: Lifecycle | None = None
            try:
                for plugin in self._registry.unstarted(Lifecycle):
                    current = plugin
                    # A context owned by the plugin, so subscriptions its start()
                    # makes belong to it whatever the handler's shape.
                    await call(
                        plugin.start,
                        RunContext(session, self._registry, owner=plugin),
                    )
                    self._registry.set_status(plugin, PluginStatus.STARTED)
                    newly.append(plugin)
                    current = None
            except BaseException as exc:
                if current is not None:
                    # Drop what the interrupted start() had already subscribed,
                    self._registry.remove_subscriptions(current)
                    # A genuine start error disables the plugin so later runs
                    # skip it; a cancellation is not a defect, so leave it
                    # retryable.
                    if isinstance(exc, Exception):
                        self._registry.set_status(current, PluginStatus.FAILED)
                for started_plugin in reversed(newly):  # roll back this pass only
                    try:
                        # Best-effort rollback
                        with contextlib.suppress(Exception):
                            await call(started_plugin.stop)
                    finally:
                        # Un-track even if stop() is interrupted, so a retried
                        # start() re-initializes this plugin instead of skipping
                        # a half-torn-down one; drop the effects its start() set up.
                        self._registry.set_status(
                            started_plugin, PluginStatus.REGISTERED
                        )
                        self._registry.remove_subscriptions(started_plugin)
                raise
            # Set only once every plugin is up, so waiters see a ready harness.
            self._started = True
        return self

    async def stop(self) -> None:
        """Release every started lifecycle plugin, in reverse registration order.

        Like :meth:`unuse`, not synchronized with an in-flight ``run`` (see the
        draining TODO there); call it once runs are quiesced.
        """
        if not self._started:
            return
        async with self._lifecycle_lock:
            if not self._started:
                return
            first_error: Exception | None = None
            for plugin in reversed(self._registry.started(Lifecycle)):
                try:
                    await call(plugin.stop)
                except Exception as exc:  # keep tearing the rest down
                    first_error = first_error or exc
                # A BaseException propagates before these, leaving the plugin
                # marked started so a retried stop() resumes where it left off.
                self._registry.set_status(plugin, PluginStatus.REGISTERED)
                # Drop the subscriptions its start() set up, so a later start()
                # re-subscribes cleanly instead of stacking duplicate handlers.
                self._registry.remove_subscriptions(plugin)
            # Flip only after the sweep: a BaseException mid-teardown leaves the
            # harness started, so cleanup can be retried instead of leaking.
            self._started = False
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
        if exc is not None:
            with contextlib.suppress(Exception):
                await self.stop()
            return
        await self.stop()

    async def run(
        self, user_input: str, *, session: Session | None = None
    ) -> RunResult:
        await self.start()  # await full lifecycle init, incl. any in-flight start
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
