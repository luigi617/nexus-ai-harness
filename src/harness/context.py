from __future__ import annotations

from typing import TypeVar

from core.events import Event, MessageAdded
from core.message import Message
from core.spawn import SpawnState
from harness.registry import Registry
from harness.session import Session
from protocols.approver import Approver
from protocols.context import Context
from protocols.hook import Hook
from protocols.plugin import Plugin
from services.invoke import _invoke

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
            await _invoke(intervention.apply, self)

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

    def fork(self, plugins: list[object] | None = None) -> RunContext:
        # None → inherit all parent plugins; a list → the child sees only these.
        if plugins is None:
            plugins = self._registry.plugins()
        child_registry = Registry()
        for plugin in plugins:
            child_registry.add(plugin)
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
