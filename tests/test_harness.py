from __future__ import annotations

import asyncio

from core.events import Event, SessionEnded, SessionStarted
from core.response import Response
from harness import GraphAIHarness
from plugins.loops import AgenticLoop
from protocols.hook import Hook
from tests.conftest import ScriptedModel


class SessionRecorder(Hook):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on(self, event: Event, ctx) -> None:
        if isinstance(event, SessionStarted | SessionEnded):
            self.events.append(type(event).__name__)


def build(*extra):
    h = GraphAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))
    for plugin in extra:
        h.use(plugin)
    return h


def test_run_sync_returns_result():
    assert build().run_sync("q").output == "hi"


def test_run_async_returns_result():
    assert asyncio.run(build().run("q")).output == "hi"


def test_run_sync_raises_inside_running_loop():
    async def go():
        try:
            build().run_sync("q")
            return False
        except RuntimeError:
            return True

    assert asyncio.run(go()) is True


def test_missing_loop_raises():
    h = GraphAIHarness().use(ScriptedModel(Response(text="x")))
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


def test_default_harness_runs(tmp_path):
    from plugins import default_harness

    h = default_harness(ScriptedModel(Response(text="hi")), memory_dir=str(tmp_path))
    assert h.run_sync("q").output == "hi"


def test_default_harness_bounds_a_runaway_loop(tmp_path):
    from plugins import default_harness

    # A model that never stops calling a tool would loop forever; the default
    # MaxIterations guard must halt it. Use `recall` (trusted, so no approval
    # prompt) so the test stays non-interactive.
    model = ScriptedModel(
        Response(
            text="",
            tool_calls=[{"name": "recall", "id": "1", "arguments": {"query": "x"}}],
        )
    )
    h = default_harness(model, max_iterations=3, memory_dir=str(tmp_path))
    result = h.run_sync("go")
    assert result.output.startswith("stopped")
    assert result.stop_reason.startswith("guard:")


def test_conversation_continues_across_runs():
    # A model that echoes how many user messages it has seen; a continued
    # session must accumulate history across run() calls.
    class Counter(ScriptedModel):
        async def complete(self, history, ctx):
            users = sum(1 for m in history if m.role == "user")
            return Response(text=f"seen {users}")

    h = GraphAIHarness().use(AgenticLoop()).use(Counter())
    r1 = h.run_sync("first")
    assert r1.output == "seen 1"
    r2 = h.run_sync("second", session=r1.session)  # continue the conversation
    assert r2.output == "seen 2"
    assert r2.session is r1.session


def test_result_exposes_run_state(tmp_path):
    from plugins import default_harness
    from plugins.hooks import CostState

    h = default_harness(ScriptedModel(Response(text="done")), memory_dir=str(tmp_path))
    result = h.run_sync("q")
    assert result.stop_reason == "completed"
    # cost is reachable via the returned session's typed state
    assert result.session.state(CostState).total == 0.0
