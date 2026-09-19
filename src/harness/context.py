from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from core.events import Event, MessageAdded
from core.invoke import call
from core.message import Message
from core.phase import Phase
from core.spawn import SpawnState
from harness.registry import Registry
from harness.session import Session
from protocols.approver import Approver
from protocols.context import Context
from protocols.hook import Hook
from protocols.plugin import Plugin

T = TypeVar("T")
P = TypeVar("P", bound=Plugin)


class RunContext(Context):
    def __init__(self, session: Session, registry: Registry) -> None:
        self._session = session
        self._registry = registry

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

    async def apply_interventions(self) -> None:
        for intervention in self._session.take_interventions():
            await call(intervention.apply, self)

    def add_message(self, message: Message) -> None:
        self._session.history.append(message)
        self.emit(MessageAdded(message))

    def emit(self, event: Event) -> None:
        for hook in self._registry.all(Hook):
            hook.on(event, self)

    def get(self, cls: type[P]) -> P | None:
        return self._registry.get(cls)

    def all(self, cls: type[P]) -> list[P]:
        return self._registry.all(cls)

    async def invoke(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        plugin = getattr(fn, "__self__", None)
        result: Any = None
        error: BaseException | None = None
        try:
            if plugin is not None:
                for interceptor in self._registry.interceptors(plugin, Phase.BEFORE):
                    await call(interceptor.run, self)
            result = await call(fn, *args, **kwargs)
        except BaseException as exc:  # captured, re-raised once teardown is done
            error = exc
        after_error: BaseException | None = None
        if plugin is not None:
            for interceptor in self._registry.interceptors(plugin, Phase.AFTER):
                try:
                    await call(interceptor.run, self)
                except BaseException as exc:  # keep running the remaining ones
                    after_error = after_error or exc
        if error is not None:
            raise error
        if after_error is not None:
            raise after_error
        return result

    def fork(self, plugins: list[object] | None = None) -> RunContext:
        # None → inherit all parent plugins; a list → the child sees only these.
        if plugins is None:
            plugins = self._registry.plugins()
        child_registry = Registry()
        for plugin in plugins:
            child_registry.add(plugin)
        # Interceptors are cross-cutting, so a child always inherits them.
        for binding in self._registry.interceptor_bindings():
            child_registry.add_interceptor(
                binding.target, binding.phase, binding.interceptor
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
