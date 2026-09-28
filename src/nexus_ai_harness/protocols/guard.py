from __future__ import annotations

from abc import abstractmethod

from nexus_ai_harness.core.guard import GuardDecision
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.plugin import Plugin


class Guard(Plugin):
    @abstractmethod
    def check(self, ctx: Context) -> GuardDecision: ...
