from __future__ import annotations

from core.guard import GuardDecision
from plugins.hooks import ElapsedState
from protocols.context import Context
from protocols.guard import Guard


class Timeout(Guard):
    """
    Stop once elapsed time exceeds the budget.
    Pair with an ElapsedTime hook, which populates ElapsedState.
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def check(self, ctx: Context) -> GuardDecision:
        if ctx.state(ElapsedState).elapsed >= self._seconds:
            return GuardDecision.halt(f"exceeded {self._seconds}s time budget")
        return GuardDecision.proceed()
