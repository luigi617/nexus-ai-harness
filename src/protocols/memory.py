from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Protocol, runtime_checkable

from protocols.plugin import Plugin


@runtime_checkable
class MemoryItem(Protocol):
    """The shape consumers (the memory tools) rely on.

    Concrete stores return their own item type — which may carry extra,
    store-specific fields — as long as it satisfies this interface.
    """

    text: str
    id: str


@runtime_checkable
class MemoryStore(Plugin, Protocol):
    """Durable, cross-session fact storage."""

    kind: ClassVar[str] = "memory"

    @abstractmethod
    def save(self, text: str, id: str | None = None) -> MemoryItem:
        """Upsert a memory.

        No ``id`` creates a new one; an existing ``id`` overwrites that memory
        (create and update are one operation).
        """

    @abstractmethod
    def get(self, id: str) -> MemoryItem | None: ...

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> Sequence[MemoryItem]: ...

    @abstractmethod
    def all(self) -> Sequence[MemoryItem]: ...

    @abstractmethod
    def delete(self, id: str) -> bool: ...
