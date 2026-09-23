from __future__ import annotations

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
from tests.conftest import make_ctx


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


def test_elapsed_time_pins_start_to_first_event_and_accumulates(monkeypatch):
    # Controlled clock: the first read pins started_at, later reads only grow elapsed.
    import plugins.hooks.elapsed as elapsed_mod

    ticks = iter([100.0, 103.5])

    class _FakeTime:
        def monotonic(self) -> float:
            return next(ticks)

    monkeypatch.setattr(elapsed_mod, "time", _FakeTime())
    ctx = make_ctx()
    h = ElapsedTime()

    h.on(MessageAdded(None), ctx)
    assert ctx.state(ElapsedState).started_at == 100.0
    assert ctx.state(ElapsedState).elapsed == 0.0

    h.on(MessageAdded(None), ctx)
    assert ctx.state(ElapsedState).started_at == 100.0  # pinned to the FIRST event
    assert ctx.state(ElapsedState).elapsed == 3.5  # grew across events


def test_iteration_counter_tracks_latest_index_across_events():
    # Fire two IterationStarted events with changing indices; latest wins.
    ctx = make_ctx()
    h = IterationCounter()
    h.on(IterationStarted(0), ctx)
    assert ctx.state(IterationState).index == 0
    h.on(IterationStarted(3), ctx)
    assert ctx.state(IterationState).index == 3
