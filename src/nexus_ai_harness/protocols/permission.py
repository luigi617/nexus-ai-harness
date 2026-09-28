from __future__ import annotations

from abc import abstractmethod

from nexus_ai_harness.core.permission import PermissionDecision
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.plugin import Plugin


class Permission(Plugin):
    @abstractmethod
    def check(self, call: dict, ctx: Context) -> PermissionDecision: ...
