from __future__ import annotations

from nexus_ai_harness.core.events import Event, ResponseReceived
from nexus_ai_harness.core.run import CostState
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.hook import Hook

__all__ = ["CostCounter", "CostState"]


class CostCounter(Hook):
    """Accumulate each response's cost (USD) and token usage into CostState."""

    def on(self, event: Event, ctx: Context) -> None:
        if isinstance(event, ResponseReceived):
            state = ctx.state(CostState)
            response = event.response
            state.total += response.cost
            state.input_tokens += response.usage.get("input_tokens", 0)
            state.output_tokens += response.usage.get("output_tokens", 0)
