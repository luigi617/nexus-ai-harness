from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

from core.guard import GuardDecision
from protocols.context import Context


@runtime_checkable
class Guard(Protocol):
    kind: ClassVar[str] = "guard"

    @abstractmethod
    def check(self, ctx: Context) -> GuardDecision: ...
