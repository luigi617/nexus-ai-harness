from __future__ import annotations

from abc import abstractmethod

from core.permission import PermissionDecision
from protocols.context import Context
from protocols.plugin import Plugin


class Permission(Plugin):
    @abstractmethod
    def check(self, call: dict, ctx: Context) -> PermissionDecision: ...
