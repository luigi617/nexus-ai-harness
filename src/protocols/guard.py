from __future__ import annotations

from abc import abstractmethod

from core.guard import GuardDecision
from protocols.context import Context
from protocols.plugin import Plugin


class Guard(Plugin):
    @abstractmethod
    def check(self, ctx: Context) -> GuardDecision: ...
