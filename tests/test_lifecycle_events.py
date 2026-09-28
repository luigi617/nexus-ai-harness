from __future__ import annotations

from nexus_ai_harness.core.events import (
    Event,
    IterationCompleted,
    ModelCallStarted,
    ResponseReceived,
)
from nexus_ai_harness.core.response import Response
from nexus_ai_harness.harness.harness import NexusAIHarness
from nexus_ai_harness.plugins.loops import AgenticLoop
from nexus_ai_harness.plugins.permissions import AllowList, AutoApprove
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.hook import Hook
from tests.conftest import RecordingTool, ScriptedModel


class RecordingHook(Hook):
    """Append every emitted event to a shared list."""

    def __init__(self, sink: list[Event]) -> None:
        self._sink = sink

    def on(self, event: Event, ctx: Context) -> None:
        self._sink.append(event)


def _harness(model, *plugins) -> NexusAIHarness:
    harness = NexusAIHarness().use(AgenticLoop()).use(model)
    for plugin in plugins:
        harness.use(plugin)
    return harness


def test_model_call_started_emitted_before_response():
    seen: list[Event] = []
    harness = _harness(ScriptedModel(Response(text="done")), RecordingHook(seen))
    harness.run_sync("go")
    lifecycle = (ModelCallStarted, ResponseReceived)
    order = [type(e).__name__ for e in seen if isinstance(e, lifecycle)]
    assert order == ["ModelCallStarted", "ResponseReceived"]


def test_iteration_completed_emitted_per_turn():
    seen: list[Event] = []
    model = ScriptedModel(
        Response(text="", tool_calls=[{"name": "echo", "id": "1", "arguments": {}}]),
        Response(text="final"),
    )
    harness = _harness(
        model,
        RecordingTool("echo", "ok"),
        AllowList(["echo"]),
        AutoApprove(),
        RecordingHook(seen),
    )
    harness.run_sync("go")
    indices = [e.index for e in seen if isinstance(e, IterationCompleted)]
    assert indices == [0, 1]
