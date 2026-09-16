from __future__ import annotations

import asyncio

from core.events import Event, SessionEnded, SessionStarted
from core.response import Response
from harness import GraphAIHarness
from plugins.loops import AgenticLoop
from protocols.hook import Hook
from tests.conftest import ScriptedProvider


class SessionRecorder(Hook):
    kind = "hook"

    def __init__(self) -> None:
        self.events: list[str] = []

    def on(self, event: Event, ctx) -> None:
        if isinstance(event, SessionStarted | SessionEnded):
            self.events.append(type(event).__name__)


def build(*extra):
    h = GraphAIHarness().use(AgenticLoop()).use(ScriptedProvider(Response(text="hi")))
    for plugin in extra:
        h.use(plugin)
    return h


def test_run_sync_returns_result():
    assert build().run_sync("q") == "hi"


def test_run_async_returns_result():
    assert asyncio.run(build().run("q")) == "hi"


def test_run_sync_raises_inside_running_loop():
    async def go():
        try:
            build().run_sync("q")
            return False
        except RuntimeError:
            return True

    assert asyncio.run(go()) is True


def test_missing_loop_raises():
    h = GraphAIHarness().use(ScriptedProvider(Response(text="x")))
    try:
        h.run_sync("q")
        raised = False
    except LookupError:
        raised = True
    assert raised


def test_session_lifecycle_events():
    rec = SessionRecorder()
    build(rec).run_sync("q")
    assert rec.events == ["SessionStarted", "SessionEnded"]
