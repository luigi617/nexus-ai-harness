from __future__ import annotations

from core.events import Event, IterationCompleted, ModelCallStarted, ResponseReceived
from core.response import Response
from harness.harness import NexusAIHarness
from plugins.loops import AgenticLoop
from plugins.permissions import AllowList, AutoApprove
from protocols.context import Context
from protocols.hook import Hook
from tests.conftest import RecordingTool, ScriptedModel


class RecordingHook(Hook):
    """Append every emitted event to a shared list."""

    kind = "hook"

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
