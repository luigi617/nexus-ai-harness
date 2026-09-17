from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

from core.permission import PermissionDecision
from protocols.context import Context
from protocols.plugin import Plugin


@runtime_checkable
class Permission(Plugin, Protocol):
    kind: ClassVar[str] = "permission"

    @abstractmethod
    def check(self, call: dict, ctx: Context) -> PermissionDecision: ...
