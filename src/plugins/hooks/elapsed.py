from __future__ import annotations

import time
from dataclasses import dataclass

from core.events import Event
from protocols.context import Context
from protocols.hook import Hook


@dataclass
class ElapsedState:
    started_at: float | None = None
    elapsed: float = 0.0


class ElapsedTime(Hook):
    """Track seconds elapsed since the first event, in ElapsedState."""

    def on(self, event: Event, ctx: Context) -> None:
        now = time.monotonic()
        state = ctx.state(ElapsedState)
        if state.started_at is None:
            state.started_at = now
        state.elapsed = now - state.started_at
