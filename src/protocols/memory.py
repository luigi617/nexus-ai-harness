from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from protocols.plugin import Plugin


@dataclass
class MemoryItem:
    """The fields every stored memory carries.

    A store returns its own item type (subclassing this) so it can add
    store-specific fields — e.g. where the memory lives — on top of these.

    Attributes:
        text: The memory's text.
        id: Its stable, store-unique identifier.
    """

    text: str
    id: str


class MemoryStore(Plugin):
    """Durable, cross-session fact storage."""

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
