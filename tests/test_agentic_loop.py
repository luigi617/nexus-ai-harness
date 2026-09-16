from __future__ import annotations

import asyncio
import time

from core.events import Event, LoopStopped
from core.response import Response
from plugins.guards import MaxIterations
from plugins.hooks import IterationCounter
from plugins.loops import AgenticLoop
from plugins.permissions import AllowList, AutoApprove
from protocols.hook import Hook
from protocols.tool import Tool
from tests.conftest import RecordingTool, ScriptedModel, make_ctx


class LoopStopRecorder(Hook):
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def on(self, event: Event, ctx) -> None:
        if isinstance(event, LoopStopped):
            self.reasons.append(event.reason)


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
