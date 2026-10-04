from __future__ import annotations

import asyncio
import logging

from nexus_ai_harness.core.events import Event, ToolCallCompleted, ToolCallStarted
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.plugins.permissions import AllowList, DenyList
from nexus_ai_harness.protocols.hook import Hook
from nexus_ai_harness.protocols.tool import Tool
from nexus_ai_harness.services.tool_runner import ToolRunner
from tests.conftest import RecordingTool, make_ctx


class EventRecorder(Hook):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on(self, event: Event, ctx) -> None:
        self.events.append(type(event).__name__)


def run(call, *plugins, tools=()):
    ctx = make_ctx(*plugins)
    runner = ToolRunner(tools)
    return asyncio.run(runner.run(call, ctx)), ctx


def test_allowed_tool_runs_and_emits_started_completed():
    tool = RecordingTool("echo", "result")
    rec = EventRecorder()
    msg, _ = run(
        {"name": "echo", "id": "1", "arguments": {"a": 1}},
        AllowList(["echo"]),
        rec,
        tools=[tool],
    )
    assert msg.content == "result"
    assert tool.calls == [{"a": 1}]
    assert rec.events == ["ToolCallStarted", "ToolCallCompleted"]


def test_denied_tool_blocked_and_emits_denied():
    tool = RecordingTool("echo")
    rec = EventRecorder()
    msg, _ = run({"name": "echo", "id": "1"}, AllowList([]), rec, tools=[tool])
    assert "denied" in msg.content
    assert tool.calls == []  # never ran
    assert rec.events == ["ToolCallDenied"]


def test_unknown_tool_reports_error():
    rec = EventRecorder()
    msg, _ = run({"name": "ghost", "id": "1"}, AllowList(["ghost"]), rec)
    assert "unknown tool" in msg.content
    assert rec.events == ["ToolCallDenied"]


def test_result_message_links_tool_use_id():
    tool = RecordingTool("echo")
    msg, _ = run({"name": "echo", "id": "abc"}, AllowList(["echo"]), tools=[tool])
    assert msg.role == "tool" and msg.tool_use_id == "abc" and msg.name == "echo"


class _ExplodingTool(Tool):
    def __init__(self) -> None:
        self.name = "boom"
        self.description = ""
        self.parameters = {}

    def run(self, arguments, ctx):
        raise ValueError("kaboom")


def test_raising_tool_becomes_error_message_not_a_crash():
    # A raising tool must degrade to an observable tool result, not tear down the run.
    rec = EventRecorder()
    msg, _ = run(
        {"name": "boom", "id": "1"}, AllowList(["boom"]), rec, tools=[_ExplodingTool()]
    )
    assert msg.role == "tool"
    assert "error" in msg.content and "kaboom" in msg.content
    assert rec.events == ["ToolCallStarted", "ToolCallCompleted"]


def test_late_registered_tool_runs():
    # A tool added after construction (not via the constructor) must still run normally.
    tool = RecordingTool("echo", "ok")
    rec = EventRecorder()
    ctx = make_ctx(AllowList(["echo"]), rec)
    runner = ToolRunner()  # no tools at construction
    runner.add(tool)
    call = {"name": "echo", "id": "1", "arguments": {"a": 2}}
    msg = asyncio.run(runner.run(call, ctx))
    assert msg.content == "ok"
    assert tool.calls == [{"a": 2}]
    assert rec.events == ["ToolCallStarted", "ToolCallCompleted"]


def test_call_missing_name_denied_by_allowlist():
    # A call with no "name" defaults it to "", which AllowList([]) denies.
    tool = RecordingTool("echo")
    rec = EventRecorder()
    msg, _ = run({"id": "1", "arguments": {}}, AllowList([]), rec, tools=[tool])
    assert "denied" in msg.content
    assert tool.calls == []
    assert rec.events == ["ToolCallDenied"]


def test_call_missing_name_permissive_policy_hits_unknown_tool():
    # A permissive DenyList lets the empty name pass, but no tool exists under "".
    rec = EventRecorder()
    msg, _ = run({"id": "1", "arguments": {}}, DenyList([]), rec)
    assert "unknown tool" in msg.content
    assert rec.events == ["ToolCallDenied"]


class _EventSink(Hook):
    def __init__(self) -> None:
        self.events: list[Event] = []

    def on(self, event: Event, ctx) -> None:
        self.events.append(event)


def test_started_and_completed_share_a_call_id_and_report_duration(monkeypatch):
    import nexus_ai_harness.services.tool_runner as runner_mod

    ticks = iter([10.0, 12.5])

    class _FakeTime:
        def perf_counter(self) -> float:
            return next(ticks)

    monkeypatch.setattr(runner_mod, "time", _FakeTime())
    sink = _EventSink()
    run(
        {"name": "echo", "id": "1"},
        AllowList(["echo"]),
        sink,
        tools=[RecordingTool("echo")],
    )
    started, completed = sink.events
    assert isinstance(started, ToolCallStarted)
    assert isinstance(completed, ToolCallCompleted)
    assert started.call_id.startswith("tcall_")
    assert completed.call_id == started.call_id
    assert completed.duration == 2.5
    assert completed.error is None


def test_each_tool_call_gets_a_distinct_call_id():
    sink = _EventSink()
    ctx = make_ctx(AllowList(["echo"]), sink)
    runner = ToolRunner([RecordingTool("echo")])
    asyncio.run(runner.run({"name": "echo", "id": "1"}, ctx))
    asyncio.run(runner.run({"name": "echo", "id": "2"}, ctx))
    ids = {e.call_id for e in sink.events if isinstance(e, ToolCallStarted)}
    assert len(ids) == 2


def test_raising_tool_logs_traceback_and_keeps_model_message(caplog):
    sink = _EventSink()
    with caplog.at_level(logging.WARNING, logger="nexus_ai_harness"):
        msg, _ = run(
            {"name": "boom", "id": "1"},
            AllowList(["boom"]),
            sink,
            tools=[_ExplodingTool()],
        )
    assert msg.content == "error: kaboom"  # unchanged string for the model
    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert record.name == "nexus_ai_harness.services.tool_runner"
    assert record.exc_info is not None and record.exc_info[0] is ValueError
    completed = sink.events[-1]
    assert isinstance(completed, ToolCallCompleted)
    assert isinstance(completed.error, ValueError)
    assert completed.duration >= 0


def test_denied_tool_call_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="nexus_ai_harness"):
        run({"name": "echo", "id": "1"}, AllowList([]), tools=[RecordingTool("echo")])
    assert any("denied" in r.getMessage() for r in caplog.records)


def test_legacy_positional_construction_still_works():
    call = {"name": "echo"}
    started = ToolCallStarted(call)
    completed = ToolCallCompleted(call, Message(role="tool"))
    assert started.call_id
    assert (completed.call_id, completed.duration, completed.error) == ("", 0.0, None)
