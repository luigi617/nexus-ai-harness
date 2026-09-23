from __future__ import annotations

import asyncio

from core.events import Event
from plugins.permissions import AllowList, DenyList
from protocols.hook import Hook
from protocols.tool import Tool
from services.tool_runner import ToolRunner
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
