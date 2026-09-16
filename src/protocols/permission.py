from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

from core.permission import PermissionDecision
from protocols.context import Context


@runtime_checkable
class Permission(Protocol):
    kind: ClassVar[str] = "permission"

    @abstractmethod
    def check(self, call: dict, ctx: Context) -> PermissionDecision: ...
