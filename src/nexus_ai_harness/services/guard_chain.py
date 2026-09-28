from __future__ import annotations

from nexus_ai_harness.core.guard import GuardDecision
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.guard import Guard


class GuardChain:
    """Consults every registered guard; the first to halt stops the loop."""

    def check(self, ctx: Context) -> GuardDecision:
        for guard in ctx.all(Guard):
            decision = guard.check(ctx)
            if decision.stop:
                return decision
        return GuardDecision.proceed()
