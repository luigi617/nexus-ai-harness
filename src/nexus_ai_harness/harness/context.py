from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from nexus_ai_harness.core.errors import mark_model_failure
from nexus_ai_harness.core.events import Event, MessageAdded
from nexus_ai_harness.core.invocation import Invocation, InvocationOutcome
from nexus_ai_harness.core.invoke import call
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.spawn import SpawnState
from nexus_ai_harness.core.subscription import Subscription
from nexus_ai_harness.harness.registry import Registry
from nexus_ai_harness.harness.session import Session
from nexus_ai_harness.protocols.approver import Approver
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.hook import Hook
from nexus_ai_harness.protocols.interceptor import Interceptor
from nexus_ai_harness.protocols.intervention import Intervention
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.plugin import Plugin

T = TypeVar("T")
P = TypeVar("P", bound=Plugin)


def _dispatch(handler: Callable[[Event, Any], Any], event: Event, ctx: Context) -> None:
    """Invoke an event handler, failing loudly if it is mistakenly async."""
    result = handler(event, ctx)
    if inspect.isawaitable(result) or inspect.isasyncgen(result):
        if inspect.iscoroutine(result):
            result.close()  # avoid a "coroutine was never awaited" warning
        raise TypeError(
            f"event handler {getattr(handler, '__qualname__', handler)!r} returned "
            f"{type(result).__name__}; handlers must be synchronous because emit() "
            "is synchronous"
        )


def _overrides(interceptor: Interceptor, method: str) -> bool:
    """Whether ``interceptor`` overrides the given phase method of ``Interceptor``."""
    return getattr(type(interceptor), method) is not getattr(Interceptor, method)


def _wants_invocation(override: Callable[..., Any]) -> bool:
    """Whether a ``before``/``after`` override expects invocation metadata.

    Distinguishes the current signature from the single-argument
    ``before(ctx)`` / ``after(ctx)`` one kept working for compatibility.
    """
    return len(inspect.signature(override).parameters) > 1


class RunContext(Context):
    def __init__(
        self, session: Session, registry: Registry, owner: object | None = None
    ) -> None:
        self._session = session
        self._registry = registry
        self._owner = owner  # the plugin this context acts for

    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def history(self) -> list[Message]:
        return self._session.history

    @property
    def interrupted(self) -> bool:
        return self._session.interrupted

    def state(self, cls: type[T]) -> T:
        return self._session.state(cls)

    @property
    def parent_session_id(self) -> str | None:
        return self._session.parent_id

    def persisted_state(self) -> dict[str, dict]:
        return self._session.persisted_state()

    def pending_interventions(self) -> list[Intervention]:
        return self._session.pending_interventions()

    async def apply_interventions(self) -> None:
        for intervention in self._session.take_interventions():
            await call(intervention.apply, self)

    def add_message(self, message: Message) -> None:
        self._session.history.append(message)
        self.emit(MessageAdded(message))

    def emit(self, event: Event) -> None:
        for hook in self._registry.all(Hook):
            _dispatch(hook.on, event, self)
        for subscription in self._registry.subscribers(event):
            _dispatch(subscription.handler, event, self)

    def on(
        self, event_type: type[Event], handler: Callable[[Event, Context], None]
    ) -> Subscription:
        owner = (
            self._owner
            if self._owner is not None
            else getattr(handler, "__self__", None)
        )
        return self._registry.subscribe(event_type, handler, owner)

    def get(self, cls: type[P]) -> P | None:
        return self._registry.get(cls)

    def all(self, cls: type[P]) -> list[P]:
        return self._registry.all(cls)

    async def invoke(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        plugin = getattr(fn, "__self__", None)
        if plugin is None:
            interceptors: list[Interceptor] = []
            invocation = None
        else:
            interceptors = self._registry.interceptors_for(plugin)
            invocation = Invocation(plugin, fn.__name__, args, kwargs)
        result: Any = None
        error: BaseException | None = None
        # Only entered interceptors get an after, avoiding unpaired teardown.
        entered: list[Interceptor] = []
        try:
            for interceptor in interceptors:
                if _overrides(interceptor, "before"):
                    assert invocation is not None  # interceptors implies a plugin
                    if _wants_invocation(interceptor.before):
                        await call(interceptor.before, invocation, self)
                    else:
                        await call(interceptor.before, self)
                entered.append(interceptor)
            try:
                result = await call(fn, *args, **kwargs)
            except Exception as exc:
                if isinstance(plugin, Model):  # blame the model, not interceptors
                    mark_model_failure(exc)
                raise
        except BaseException as exc:  # captured, re-raised once teardown is done
            error = exc
        outcome = InvocationOutcome(result, error)
        after_error: BaseException | None = None
        for interceptor in entered:
            if not _overrides(interceptor, "after"):
                continue
            try:
                assert invocation is not None  # entered implies a plugin
                if _wants_invocation(interceptor.after):
                    await call(interceptor.after, invocation, outcome, self)
                else:
                    await call(interceptor.after, self)
            except BaseException as exc:  # keep running the remaining ones
                after_error = after_error or exc
        if error is not None:
            # Chain the teardown error onto the primary's context so it isn't lost.
            if after_error is not None:
                tail: BaseException = error
                seen = {id(error)}
                while tail.__context__ is not None and id(tail.__context__) not in seen:
                    tail = tail.__context__
                    seen.add(id(tail))
                if tail is not after_error and tail.__context__ is None:
                    tail.__context__ = after_error
            raise error
        if after_error is not None:
            raise after_error
        return result

    def fork(
        self,
        plugins: list[Plugin] | None = None,
        overrides: dict[type, Plugin] | None = None,
    ) -> RunContext:
        # None → inherit all parent plugins; a list → the child sees only these.
        members = plugins if plugins is not None else list(self._registry.plugins())
        if overrides:
            targets = tuple(overrides)
            members = [p for p in members if not isinstance(p, targets)]
            members = [*members, *overrides.values()]
        child_registry = Registry()
        for plugin in members:
            child_registry.add(plugin)
        # Subscriptions follow their owner into the child; unowned ones cross-cut.
        for subscription in self._registry.subscriptions():
            owner = subscription.owner
            if owner is None or any(owner is member for member in members):
                child_registry.subscribe(
                    subscription.event_type, subscription.handler, owner=owner
                )
        # Headless subagent: inherit the parent's approver unless given one.
        if child_registry.get(Approver) is None:
            approver = self._registry.get(Approver)
            if approver is not None:
                child_registry.add(approver)

        child_session = Session()
        # Share the interrupt signal so interrupting the root stops subagents.
        child_session._interrupt = self._session._interrupt
        child = RunContext(child_session, child_registry)
        child.state(SpawnState).depth = self.state(SpawnState).depth + 1
        return child
