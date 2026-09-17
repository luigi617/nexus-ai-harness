from __future__ import annotations

from core.guard import GuardDecision
from plugins.hooks import CostCounter, CostState
from protocols.context import Context
from protocols.guard import Guard


class BudgetGuard(Guard):
    """Stop once accumulated cost (USD) reaches the budget."""

    requires = (CostCounter,)

    def __init__(self, max_cost: float) -> None:
        self._max_cost = max_cost

    def check(self, ctx: Context) -> GuardDecision:
        if ctx.state(CostState).total >= self._max_cost:
            return GuardDecision.halt(f"exceeded ${self._max_cost} budget")
        return GuardDecision.proceed()
