from __future__ import annotations

import asyncio

from core.events import Event
from harness.callbacks import CallbackHook, EventHandler
from harness.context import RunContext
from harness.registry import Registry
from harness.result import RunResult
from harness.session import Session
from harness.validation import describe_registry, validate_registry
from services.runner import run_session


class NexusAIHarness:
    def __init__(self) -> None:
        self._registry = Registry()
        self._callbacks = CallbackHook()
        self._registry.add(self._callbacks)

    def use(self, plugin: object) -> NexusAIHarness:
        self._registry.add(plugin)
        return self

    def on(self, event_type: type[Event], handler: EventHandler) -> NexusAIHarness:
        """Register ``handler`` to run whenever an ``event_type`` event is emitted.

        A lightweight alternative to writing a :class:`~protocols.hook.Hook`: the
        handler observes the typed event directly instead of switching on event
        type itself. It fires for ``event_type`` and any subclass, so listening
        on :class:`~core.events.Event` observes every event.

        Args:
            event_type: The event class to listen for.
            handler: A sync callable taking ``(event)`` or ``(event, ctx)``.

        Returns:
            ``self``, so registrations can be chained after ``use(...)``.
        """
        self._callbacks.register(event_type, handler)
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
