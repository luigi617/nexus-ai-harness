from __future__ import annotations

from dataclasses import dataclass

from core.events import Event, ResponseReceived
from protocols.context import Context
from protocols.hook import Hook


@dataclass
class CostState:
    total: float = 0.0


class CostCounter(Hook):
    """Accumulate the running cost (USD) from each response into CostState."""

    def on(self, event: Event, ctx: Context) -> None:
        if isinstance(event, ResponseReceived):
            ctx.state(CostState).total += event.response.cost
