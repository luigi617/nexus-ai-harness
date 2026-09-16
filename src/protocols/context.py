from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import Protocol, TypeVar, runtime_checkable

from core.events import Event
from core.message import Message
from protocols.plugin import Plugin

T = TypeVar("T")
P = TypeVar("P", bound=Plugin)


@runtime_checkable
class Context(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def history(self) -> list[Message]: ...

    @property
    def interrupted(self) -> bool: ...

    @abstractmethod
    def state(self, cls: type[T]) -> T: ...

    @abstractmethod
    def apply_interventions(self) -> Awaitable[None]: ...

    @abstractmethod
    def add_message(self, message: Message) -> None: ...

    @abstractmethod
    def emit(self, event: Event) -> None: ...

    @abstractmethod
    def get(self, cls: type[P]) -> P | None: ...

    @abstractmethod
    def all(self, cls: type[P]) -> list[P]: ...

    @abstractmethod
    def fork(self, plugins: list[object] | None = None) -> Context:
        """Create a forked context"""
