from __future__ import annotations

from dataclasses import dataclass

from core.events import Event, IterationStarted
from protocols.context import Context
from protocols.hook import Hook


@dataclass
class IterationState:
    index: int = 0


class IterationCounter(Hook):
    """Record the loop's iteration index into IterationState."""

    def on(self, event: Event, ctx: Context) -> None:
        if isinstance(event, IterationStarted):
            ctx.state(IterationState).index = event.index
