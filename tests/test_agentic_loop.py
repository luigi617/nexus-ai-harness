from __future__ import annotations

import asyncio
import time

from core.events import Event, LoopStopped
from core.message import Message
from core.response import Response
from core.run import RunState
from plugins.guards import BudgetGuard, MaxIterations, Timeout
from plugins.hooks import CostCounter, CostState, ElapsedTime, IterationCounter
from plugins.loops import AgenticLoop
from plugins.permissions import AllowList, AutoApprove
from protocols.context_manager import ContextManager
from protocols.hook import Hook
from protocols.model import Model
from protocols.router import Router
from protocols.tool import Tool
from tests.conftest import RecordingTool, ScriptedModel, make_ctx


class LoopStopRecorder(Hook):
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def on(self, event: Event, ctx) -> None:
        if isinstance(event, LoopStopped):
            self.reasons.append(event.reason)


# --- ContextManager middleware chain (finding 1) -------------------------


class AppendMarker(ContextManager):
    """Appends a system marker message to the history it is handed."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def process(self, history, ctx):
        return [*history, Message(role="system", content=self.marker)]


class UppercaseLast(ContextManager):
    """Uppercases the content of the last message (returns a new list)."""

    def process(self, history, ctx):
        if not history:
            return list(history)
        out = list(history)
        last = out[-1]
        out[-1] = Message(role=last.role, content=last.content.upper())
        return out


class ReplaceHistory(ContextManager):
    """Discards the input and returns a brand-new history."""

    def __init__(self, replacement) -> None:
        self._replacement = replacement

    def process(self, history, ctx):
        return list(self._replacement)


def test_context_manager_chain_threads_history_in_order():
    # Two CMs applied in registration order, each fed the previous one's output:
    # AppendMarker adds "cm1", then UppercaseLast uppercases that appended marker,
    # proving the accumulator (not raw ctx.history) is threaded through.
    model = ScriptedModel(Response(text="done"))
    ctx = make_ctx(model, AppendMarker("cm1"), UppercaseLast())
    ctx.add_message(Message(role="user", content="hi"))
    asyncio.run(AgenticLoop().run(ctx))
    assert [m.content for m in model.calls[0]] == ["hi", "CM1"]
    # the transient marker was never persisted to the real history
    assert "cm1" not in [m.content for m in ctx.history]


def test_context_manager_can_replace_history_wholesale():
    model = ScriptedModel(Response(text="ok"))
    replacement = [Message(role="user", content="brand new")]
    ctx = make_ctx(model, ReplaceHistory(replacement))
    ctx.add_message(Message(role="user", content="original"))
    asyncio.run(AgenticLoop().run(ctx))
    # the model saw the replacement, not the original history
    assert [m.content for m in model.calls[0]] == ["brand new"]
    assert ctx.history[0].content == "original"


# --- guard stops a real multi-iteration loop via hook state (finding 2) ---


def test_guard_stops_multi_iteration_loop_via_hook_state():
    # A forever-tool-calling model; IterationCounter feeds MaxIterations(2).
    tool_call = {"name": "echo", "id": "1", "arguments": {}}
    model = ScriptedModel(Response(text="", tool_calls=[tool_call]))
    rec = LoopStopRecorder()
    ctx = make_ctx(
        model,
        IterationCounter(),
        MaxIterations(2),
        RecordingTool("echo"),
        AllowList(["echo"]),
        AutoApprove(),
        rec,
    )
    result = asyncio.run(AgenticLoop().run(ctx))
    assert len(model.calls) == 2  # stopped at the start of iteration index 2
    assert result == "stopped: reached 2 iterations"
    assert ctx.state(RunState).stop_reason == "guard: reached 2 iterations"
    assert rec.reasons == ["guard: reached 2 iterations"]


def test_budget_guard_stops_live_loop_one_iteration_late():
    # Per-turn cost 0.6 crosses the 1.0 budget after turn 1 (total 1.2), but the
    # guard runs at the START of an iteration and CostCounter accumulates after
    # the response, so the stop lands one iteration late — at the start of turn 2.
    tool_call = {"name": "echo", "id": "1", "arguments": {}}
    model = ScriptedModel(Response(text="", tool_calls=[tool_call], cost=0.6))
    ctx = make_ctx(
        model,
        CostCounter(),
        BudgetGuard(1.0),
        RecordingTool("echo"),
        AllowList(["echo"]),
        AutoApprove(),
    )
    result = asyncio.run(AgenticLoop().run(ctx))
    assert len(model.calls) == 2
    assert result == "stopped: exceeded $1.0 budget"
    assert ctx.state(RunState).stop_reason == "guard: exceeded $1.0 budget"


def test_budget_guard_checks_the_final_responses_cost():
    # A single response with no tool_calls ends the loop; guards are re-checked
    # after its cost is accounted, so an over-budget answer stops instead of
    # being returned as a clean completion.
    model = ScriptedModel(Response(text="answer", cost=100.0))
    rec = LoopStopRecorder()
    ctx = make_ctx(model, CostCounter(), BudgetGuard(1.0), rec)
    result = asyncio.run(AgenticLoop().run(ctx))
    assert result == "stopped: exceeded $1.0 budget"
    assert ctx.state(RunState).stop_reason == "guard: exceeded $1.0 budget"
    assert rec.reasons == ["guard: exceeded $1.0 budget"]
    assert abs(ctx.state(CostState).total - 100.0) < 1e-9


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0


class _FakeTime:
    def __init__(self, clock: _Clock) -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.t


class ClockTool(Tool):
    """Advances a fake clock by a fixed step each time it runs."""

    def __init__(self, clock: _Clock, step: float) -> None:
        self.name = "tick"
        self.description = ""
        self.parameters = {}
        self._clock = clock
        self._step = step

    def run(self, arguments, ctx):
        self._clock.t += self._step
        return "tick"


def test_timeout_stops_live_loop_after_expected_turns(monkeypatch):
    import plugins.hooks.elapsed as elapsed_mod

    clock = _Clock()
    monkeypatch.setattr(elapsed_mod, "time", _FakeTime(clock))
    tool_call = {"name": "tick", "id": "1", "arguments": {}}
    model = ScriptedModel(Response(text="", tool_calls=[tool_call]))
    ctx = make_ctx(
        model,
        ElapsedTime(),
        Timeout(10.0),
        ClockTool(clock, 6.0),
        AllowList(["tick"]),
        AutoApprove(),
    )
    result = asyncio.run(AgenticLoop().run(ctx))
    # each turn advances the clock 6s; elapsed reaches 12s at the start of turn 2.
    assert len(model.calls) == 2
    assert result == "stopped: exceeded 10.0s time budget"
    assert ctx.state(RunState).stop_reason == "guard: exceeded 10.0s time budget"


def test_max_iterations_fails_open_without_its_counter_hook():
    # MaxIterations registered WITHOUT IterationCounter: IterationState.index
    # stays 0, so the cap never trips and the loop runs to natural completion.
    # Documents a fail-open footgun (see bugsFound) by pinning current behavior.
    tc = {"name": "echo", "id": "1", "arguments": {}}
    model = ScriptedModel(
        Response(text="", tool_calls=[tc]),
        Response(text="", tool_calls=[tc]),
        Response(text="final"),
    )
    ctx = make_ctx(
        model,
        MaxIterations(2),  # no IterationCounter registered
        RecordingTool("echo"),
        AllowList(["echo"]),
        AutoApprove(),
    )
    result = asyncio.run(AgenticLoop().run(ctx))
    assert result == "final"  # ran all 3 turns; the 2-iteration cap never fired
    assert len(model.calls) == 3
    assert ctx.state(RunState).stop_reason == "completed"


# --- Router branch (finding 5) -------------------------------------------


class PickModel(Router):
    def __init__(self, target: Model) -> None:
        self._target = target

    def route(self, history, ctx):
        return self._target


class NoneRouter(Router):
    def route(self, history, ctx):
        return None


class RecordingRouter(Router):
    def __init__(self, target: Model) -> None:
        self._target = target
        self.histories: list[list] = []

    def route(self, history, ctx):
        self.histories.append(list(history))
        return self._target


def test_router_selects_the_named_model():
    a = ScriptedModel(Response(text="from-a"))
    b = ScriptedModel(Response(text="from-b"))
    ctx = make_ctx(a, b, PickModel(b))
    result = asyncio.run(AgenticLoop().run(ctx))
    assert result == "from-b"
    assert len(b.calls) == 1 and a.calls == []


def test_router_returning_none_raises_lookup_error():
    ctx = make_ctx(ScriptedModel(Response(text="x")), NoneRouter())
    raised = False
    msg = ""
    try:
        asyncio.run(AgenticLoop().run(ctx))
    except LookupError as exc:
        raised = True
        msg = str(exc)
    assert raised
    assert msg == "no model registered"


def test_router_receives_the_context_managed_history():
    a = ScriptedModel(Response(text="done"))
    router = RecordingRouter(a)
    ctx = make_ctx(a, AppendMarker("cm-marker"), router)
    ctx.add_message(Message(role="user", content="hi"))
    asyncio.run(AgenticLoop().run(ctx))
    assert [m.content for m in router.histories[0]] == ["hi", "cm-marker"]


# --- interrupt path (finding 8) ------------------------------------------


def test_interrupt_before_run_emits_loopstopped_and_sets_state():
    model = ScriptedModel(Response(text="never"))
    rec = LoopStopRecorder()
    ctx = make_ctx(model, rec)
    ctx._session.interrupt()
    result = asyncio.run(AgenticLoop().run(ctx))
    assert result == "stopped: interrupted"
    assert rec.reasons == ["interrupted"]
    assert ctx.state(RunState).stop_reason == "interrupted"
    assert model.calls == []  # stopped before any model call


class InterruptingTool(Tool):
    """A tool that interrupts the session as a side effect."""

    def __init__(self) -> None:
        self.name = "boom"
        self.description = ""
        self.parameters = {}

    def run(self, arguments, ctx):
        ctx._session.interrupt()
        return "interrupting"


def test_interrupt_raised_mid_loop_stops_before_next_model_call():
    tc = {"name": "boom", "id": "1", "arguments": {}}
    model = ScriptedModel(
        Response(text="", tool_calls=[tc]),
        Response(text="second"),
    )
    rec = LoopStopRecorder()
    ctx = make_ctx(model, InterruptingTool(), AllowList(["boom"]), AutoApprove(), rec)
    result = asyncio.run(AgenticLoop().run(ctx))
    assert len(model.calls) == 1  # stopped before the second turn's model call
    assert result == "stopped: interrupted"
    assert rec.reasons == ["interrupted"]
    assert ctx.state(RunState).stop_reason == "interrupted"


def test_tool_call_then_finish():
    tool = RecordingTool("echo", "42")
    model = ScriptedModel(
        Response(text="", tool_calls=[{"name": "echo", "id": "1", "arguments": {}}]),
        Response(text="final answer"),
    )
    ctx = make_ctx(model, tool, AllowList(["echo"]), AutoApprove())
    result = asyncio.run(AgenticLoop().run(ctx))
    assert result == "final answer"
    assert tool.calls == [{}]


def test_persisted_assistant_message_preserves_tool_calls():
    # The recorded assistant turn must retain the model's tool_calls across runs.
    tool = RecordingTool("echo", "42")
    call = {"name": "echo", "id": "1", "arguments": {"x": 1}}
    model = ScriptedModel(
        Response(text="calling", tool_calls=[call]),
        Response(text="final answer"),
    )
    ctx = make_ctx(model, tool, AllowList(["echo"]), AutoApprove())
    asyncio.run(AgenticLoop().run(ctx))
    assistant = ctx.history[0]  # no prior turns, so the first turn is the assistant
    assert assistant.role == "assistant"
    assert assistant.content == "calling"
    assert assistant.tool_calls == [call]


def test_no_provider_raises():
    ctx = make_ctx()
    try:
        asyncio.run(AgenticLoop().run(ctx))
        raised = False
    except LookupError:
        raised = True
    assert raised


def test_guard_stops_loop():
    model = ScriptedModel(Response(text="never reached"))
    rec = LoopStopRecorder()
    ctx = make_ctx(model, MaxIterations(0), IterationCounter(), rec)
    result = asyncio.run(AgenticLoop().run(ctx))
    assert result.startswith("stopped")
    assert any("guard" in r for r in rec.reasons)


def test_natural_exit_emits_completed():
    model = ScriptedModel(Response(text="done"))
    rec = LoopStopRecorder()
    ctx = make_ctx(model, rec)
    asyncio.run(AgenticLoop().run(ctx))
    assert rec.reasons == ["completed"]


def test_parallel_tool_calls_run_concurrently():
    class SlowTool(Tool):
        def __init__(self, name):
            self.name = name
            self.description = ""
            self.parameters = {}

        def run(self, arguments, ctx):
            time.sleep(0.2)
            return "slow"

    model = ScriptedModel(
        Response(
            text="",
            tool_calls=[
                {"name": "a", "id": "1", "arguments": {}},
                {"name": "b", "id": "2", "arguments": {}},
                {"name": "c", "id": "3", "arguments": {}},
            ],
        ),
        Response(text="done"),
    )
    ctx = make_ctx(
        model,
        SlowTool("a"),
        SlowTool("b"),
        SlowTool("c"),
        AllowList(["a", "b", "c"]),
        AutoApprove(),
    )
    start = time.perf_counter()
    asyncio.run(AgenticLoop().run(ctx))
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5  # 3x0.2s ran in parallel, not 0.6s sequential
