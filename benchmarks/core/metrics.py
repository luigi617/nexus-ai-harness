from __future__ import annotations

from dataclasses import dataclass

from benchmarks.core.task import RunMetrics
from core.events import Event, ResponseReceived
from core.run import RunState
from harness.session import Session
from protocols.context import Context
from protocols.hook import Hook


@dataclass
class MetricsState:
    """Running usage totals, stored in the session's typed state."""

    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    cost: float = 0.0


class MetricsCollector(Hook):
    """Sum token usage, cost, and call counts from every model response."""

    def on(self, event: Event, ctx: Context) -> None:
        if isinstance(event, ResponseReceived):
            state = ctx.state(MetricsState)
            response = event.response
            state.input_tokens += response.usage.get("input_tokens", 0)
            state.output_tokens += response.usage.get("output_tokens", 0)
            state.cost += response.cost
            state.model_calls += 1
            state.tool_calls += len(response.tool_calls)


def read_metrics(session: Session, *, seconds: float) -> RunMetrics:
    """Snapshot the accumulated metrics for a finished run."""
    state = session.state(MetricsState)
    return RunMetrics(
        seconds=seconds,
        cost=state.cost,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        model_calls=state.model_calls,
        tool_calls=state.tool_calls,
        stop_reason=session.state(RunState).stop_reason,
    )
