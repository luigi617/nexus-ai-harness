from __future__ import annotations

from core.events import (
    Event,
    IterationCompleted,
    ModelCallStarted,
    ResponseReceived,
    ToolCallCompleted,
)
from core.response import Response
from harness.callbacks import CallbackHook
from harness.harness import NexusAIHarness
from plugins.loops import AgenticLoop
from plugins.permissions import AllowList, AutoApprove
from tests.conftest import RecordingTool, ScriptedModel, make_ctx


def _harness(model, *plugins) -> NexusAIHarness:
    harness = NexusAIHarness().use(AgenticLoop()).use(model)
    for plugin in plugins:
        harness.use(plugin)
    return harness


def test_on_returns_self_for_chaining():
    harness = NexusAIHarness()
    assert harness.on(Event, lambda e: None) is harness


def test_on_invokes_handler_with_event():
    seen: list[ResponseReceived] = []
    harness = _harness(ScriptedModel(Response(text="hi")))
    harness.on(ResponseReceived, seen.append)
    harness.run_sync("go")
    assert len(seen) == 1
    assert seen[0].response.text == "hi"


def test_on_passes_ctx_to_two_arg_handler():
    captured: list[str] = []
    harness = _harness(ScriptedModel(Response(text="hi")))
    harness.on(ResponseReceived, lambda event, ctx: captured.append(ctx.session_id))
    result = harness.run_sync("go")
    assert captured == [result.session.id]


def test_on_event_base_observes_everything():
    count = 0

    def bump(_event: Event) -> None:
        nonlocal count
        count += 1

    harness = _harness(ScriptedModel(Response(text="hi")))
    harness.on(Event, bump)
    harness.run_sync("go")
    assert count > 1  # at least SessionStarted, ModelCallStarted, ... SessionEnded


def test_on_filters_by_type():
    tool_events: list[ToolCallCompleted] = []
    harness = _harness(ScriptedModel(Response(text="no tools here")))
    harness.on(ToolCallCompleted, tool_events.append)
    harness.run_sync("go")
    assert tool_events == []


def test_model_call_started_emitted_before_response():
    order: list[str] = []
    harness = _harness(ScriptedModel(Response(text="done")))
    harness.on(ModelCallStarted, lambda e: order.append("before"))
    harness.on(ResponseReceived, lambda e: order.append("after"))
    harness.run_sync("go")
    assert order == ["before", "after"]


def test_iteration_completed_emitted_per_turn():
    indices: list[int] = []
    model = ScriptedModel(
        Response(text="", tool_calls=[{"name": "echo", "id": "1", "arguments": {}}]),
        Response(text="final"),
    )
    harness = _harness(
        model, RecordingTool("echo", "ok"), AllowList(["echo"]), AutoApprove()
    )
    harness.on(IterationCompleted, lambda e: indices.append(e.index))
    harness.run_sync("go")
    assert indices == [0, 1]


def test_callback_hook_dispatches_by_subclass():
    ctx = make_ctx()
    hook = CallbackHook()
    seen: list[Event] = []
    hook.register(Event, seen.append)
    hook.register(ResponseReceived, seen.append)
    hook.on(ModelCallStarted([]), ctx)  # matches Event only
    hook.on(ResponseReceived(Response()), ctx)  # matches both
    assert len(seen) == 3
