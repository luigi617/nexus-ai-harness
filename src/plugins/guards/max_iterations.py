from __future__ import annotations

from core.guard import GuardDecision
from plugins.hooks import IterationCounter, IterationState
from protocols.context import Context
from protocols.guard import Guard


class MaxIterations(Guard):
    """
    Stop once the iteration count reaches the limit.
    """

    requires = (IterationCounter,)

    def __init__(self, limit: int) -> None:
        self._limit = limit

    def check(self, ctx: Context) -> GuardDecision:
        if ctx.state(IterationState).index >= self._limit:
            return GuardDecision.halt(f"reached {self._limit} iterations")
        return GuardDecision.proceed()
