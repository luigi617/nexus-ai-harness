from __future__ import annotations

from tests.conftest import make_ctx

from core.events import IterationStarted, MessageAdded, ResponseReceived
from core.response import Response
from plugins.hooks import (
    CostCounter,
    CostState,
    ElapsedState,
    ElapsedTime,
    IterationCounter,
    IterationState,
)


def test_iteration_counter_records_index():
    ctx = make_ctx()
    IterationCounter().on(IterationStarted(4), ctx)
    assert ctx.state(IterationState).index == 4


def test_cost_counter_accumulates():
    ctx = make_ctx()
    h = CostCounter()
    h.on(ResponseReceived(Response(cost=0.1)), ctx)
    h.on(ResponseReceived(Response(cost=0.25)), ctx)
    assert abs(ctx.state(CostState).total - 0.35) < 1e-9


def test_cost_counter_ignores_other_events():
    ctx = make_ctx()
    CostCounter().on(IterationStarted(0), ctx)
    assert ctx.state(CostState).total == 0.0


def test_elapsed_time_sets_started_and_grows():
    ctx = make_ctx()
    h = ElapsedTime()
    h.on(MessageAdded(None), ctx)  # any event
    state = ctx.state(ElapsedState)
    assert state.started_at is not None
    assert state.elapsed >= 0.0
