from __future__ import annotations

from nexus_ai_harness.core.guard import GuardDecision
from nexus_ai_harness.plugins.hooks import ElapsedState, ElapsedTime
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.guard import Guard


class Timeout(Guard):
    """Stop once elapsed time exceeds the budget."""

    requires = (ElapsedTime,)

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def check(self, ctx: Context) -> GuardDecision:
        if ctx.state(ElapsedState).elapsed >= self._seconds:
            return GuardDecision.halt(f"exceeded {self._seconds}s time budget")
        return GuardDecision.proceed()
