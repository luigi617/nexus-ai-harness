from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class Plugin(Protocol):
    """Common base of every plugin protocol."""

    kind: ClassVar[str]
    requires: ClassVar[tuple[type[Plugin], ...]] = ()
