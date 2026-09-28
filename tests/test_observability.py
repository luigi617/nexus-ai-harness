from __future__ import annotations

import asyncio
import logging

import pytest

from nexus_ai_harness.core.events import (
    Event,
    ModelCallCompleted,
    ModelCallStarted,
    ResponseReceived,
    SessionEnded,
    ToolCallCompleted,
    ToolCallStarted,
)
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.response import Response
from nexus_ai_harness.harness.harness import NexusAIHarness
from nexus_ai_harness.plugins import LoggingHook as ExportedLoggingHook
from nexus_ai_harness.plugins.hooks import LoggingHook
from nexus_ai_harness.plugins.loops import AgenticLoop, ChatLoop
from nexus_ai_harness.plugins.permissions import AllowList, AutoApprove
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.hook import Hook
from tests.conftest import RecordingTool, ScriptedModel, make_ctx


class _Sink(Hook):
    def __init__(self) -> None:
        self.events: list[Event] = []

    def on(self, event: Event, ctx: Context) -> None:
        self.events.append(event)


def _tool_turn() -> ScriptedModel:
    return ScriptedModel(
        Response(
            text="",
            tool_calls=[{"name": "echo", "id": "t1", "arguments": {"secret": "x"}}],
            usage={"input_tokens": 7, "output_tokens": 2},
        ),
        Response(text="final", usage={"input_tokens": 9, "output_tokens": 1}),
    )


def _agentic(model: ScriptedModel, *plugins: object) -> NexusAIHarness:
    harness = (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(model)
        .use(RecordingTool("echo", "ok"))
        .use(AllowList(["echo"]))
        .use(AutoApprove())
    )
    for plugin in plugins:
        harness.use(plugin)
    return harness


# --- model call events ---------------------------------------------------


def test_agentic_loop_pairs_each_model_call_with_a_completion():
    sink = _Sink()
    _agentic(_tool_turn(), sink).run_sync("go")
    started = [e for e in sink.events if isinstance(e, ModelCallStarted)]
    completed = [e for e in sink.events if isinstance(e, ModelCallCompleted)]
    assert len(started) == len(completed) == 2
    assert [s.call_id for s in started] == [c.call_id for c in completed]
    assert len({s.call_id for s in started}) == 2  # distinct per call
    assert all(c.duration >= 0 for c in completed)
    assert [c.response.usage["input_tokens"] for c in completed] == [7, 9]


def test_model_call_completed_precedes_response_received():
    sink = _Sink()
    _agentic(ScriptedModel(Response(text="done")), sink).run_sync("go")
    lifecycle = (ModelCallStarted, ModelCallCompleted, ResponseReceived)
    order = [type(e).__name__ for e in sink.events if isinstance(e, lifecycle)]
    assert order == ["ModelCallStarted", "ModelCallCompleted", "ResponseReceived"]


def test_agentic_loop_model_call_duration_uses_the_clock(monkeypatch):
    import nexus_ai_harness.plugins.loops.agentic as agentic_mod

    ticks = iter([1.0, 1.75])

    class _FakeTime:
        def perf_counter(self) -> float:
            return next(ticks)

    monkeypatch.setattr(agentic_mod, "time", _FakeTime())
    sink = _Sink()
    ctx = make_ctx(ScriptedModel(Response(text="hi")), sink)
    asyncio.run(AgenticLoop().run(ctx))
    [completed] = [e for e in sink.events if isinstance(e, ModelCallCompleted)]
    assert completed.duration == 0.75


def test_chat_loop_emits_model_call_completed(monkeypatch):
    import nexus_ai_harness.plugins.loops.chat as chat_mod

    ticks = iter([5.0, 5.5])

    class _FakeTime:
        def perf_counter(self) -> float:
            return next(ticks)

    monkeypatch.setattr(chat_mod, "time", _FakeTime())
    sink = _Sink()
    response = Response(text="hi")
    ctx = make_ctx(ScriptedModel(response), sink)
    asyncio.run(ChatLoop().run(ctx))
    [started] = [e for e in sink.events if isinstance(e, ModelCallStarted)]
    [completed] = [e for e in sink.events if isinstance(e, ModelCallCompleted)]
    assert completed.call_id == started.call_id
    assert completed.response is response
    assert completed.duration == 0.5


def test_model_call_started_keeps_its_positional_shape():
    event = ModelCallStarted([])
    assert event.history == []
    assert event.call_id.startswith("mcall_")


def test_tool_call_events_correlate_inside_a_run():
    sink = _Sink()
    _agentic(_tool_turn(), sink).run_sync("go")
    [started] = [e for e in sink.events if isinstance(e, ToolCallStarted)]
    [completed] = [e for e in sink.events if isinstance(e, ToolCallCompleted)]
    assert completed.call_id == started.call_id
    assert completed.duration >= 0


# --- library logging -----------------------------------------------------


def test_package_root_has_a_null_handler_and_no_config():
    root = logging.getLogger("nexus_ai_harness")
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)
    assert root.level == logging.NOTSET  # the application decides the level


def test_model_calls_are_logged_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="nexus_ai_harness"):
        _agentic(ScriptedModel(Response(text="done"))).run_sync("go")
    names = {r.name for r in caplog.records}
    assert "nexus_ai_harness.plugins.loops.agentic" in names
    assert "nexus_ai_harness.services.runner" in names
    assert all(r.levelno == logging.DEBUG for r in caplog.records)


# --- LoggingHook ---------------------------------------------------------


def test_logging_hook_is_exported():
    assert ExportedLoggingHook is LoggingHook


def test_logging_hook_logs_every_event_with_structured_extra(caplog):
    sink = _Sink()
    with caplog.at_level(logging.INFO, logger="nexus_ai_harness.events"):
        result = _agentic(_tool_turn(), sink, LoggingHook()).run_sync("go")
    records = [r for r in caplog.records if r.name == "nexus_ai_harness.events"]
    assert [r.event for r in records] == [type(e).__name__ for e in sink.events]
    assert {r.session_id for r in records} == {result.session.id}

    by_event = {r.event: r for r in records}
    completed = by_event["ModelCallCompleted"].event_fields
    assert completed["call_id"].startswith("mcall_")
    assert completed["duration"] >= 0
    assert completed["response"]["usage"] == {"input_tokens": 9, "output_tokens": 1}
    assert by_event["ModelCallStarted"].event_fields["history"] >= 1
    tool = by_event["ToolCallCompleted"]
    assert tool.event_fields["call"] == {"id": "t1", "name": "echo"}
    assert tool.getMessage().startswith("ToolCallCompleted call_id=tcall_")


def test_logging_hook_omits_content_by_default(caplog):
    with caplog.at_level(logging.INFO, logger="nexus_ai_harness.events"):
        _agentic(_tool_turn(), LoggingHook()).run_sync("top secret prompt")
    for record in caplog.records:
        assert "top secret prompt" not in repr(record.event_fields)
        assert "secret" not in repr(record.event_fields.get("call", {}))
    [ended] = [r for r in caplog.records if r.event == "SessionEnded"]
    assert ended.event_fields["result"] == {"chars": len("final")}


def test_logging_hook_include_content_logs_text_and_arguments(caplog):
    with caplog.at_level(logging.INFO, logger="nexus_ai_harness.events"):
        _agentic(_tool_turn(), LoggingHook(include_content=True)).run_sync("hello")
    by_event = {r.event: r for r in caplog.records}
    call = by_event["ToolCallStarted"].event_fields["call"]
    assert call["arguments"] == {"secret": "x"}
    assert by_event["SessionEnded"].event_fields["result"] == "final"
    first_message = next(r for r in caplog.records if r.event == "MessageAdded")
    assert first_message.event_fields["message"]["content"] == "hello"


def test_logging_hook_escalates_tool_errors_to_warning(caplog):
    logger = logging.getLogger("tests.observability.custom")
    hook = LoggingHook(logger=logger, level=logging.DEBUG)
    call = {"name": "boom", "id": "1"}
    completed = ToolCallCompleted(
        call,
        Message(role="tool", content="error: kaboom"),
        call_id="tcall_1",
        duration=0.1,
        error=ValueError("kaboom"),
    )
    with caplog.at_level(logging.DEBUG, logger="tests.observability.custom"):
        hook.on(completed, make_ctx())
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.event_fields["error"] == "ValueError: kaboom"


def test_logging_hook_skips_work_when_level_is_disabled(caplog):
    logger = logging.getLogger("tests.observability.quiet")
    logger.setLevel(logging.ERROR)
    try:
        LoggingHook(logger=logger).on(SessionEnded("s", "done"), make_ctx())
    finally:
        logger.setLevel(logging.NOTSET)
    assert caplog.records == []


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO])
def test_logging_hook_uses_the_configured_level(caplog, level):
    with caplog.at_level(logging.DEBUG, logger="nexus_ai_harness.events"):
        LoggingHook(level=level).on(SessionEnded("s", "done"), make_ctx())
    [record] = caplog.records
    assert record.levelno == level
