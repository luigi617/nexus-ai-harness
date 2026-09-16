from __future__ import annotations

import asyncio

from core.events import Event
from plugins.permissions import AllowList
from protocols.hook import Hook
from services.tool_runner import ToolRunner
from tests.conftest import RecordingTool, make_ctx


class EventRecorder(Hook):
    kind = "hook"

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
