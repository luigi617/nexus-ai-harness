from __future__ import annotations

import asyncio

from core.events import Event, SessionEnded, SessionStarted
from core.response import Response
from harness import NexusAIHarness
from plugins.loops import AgenticLoop
from protocols.hook import Hook
from protocols.lifecycle import Lifecycle
from protocols.plugin import Plugin
from tests.conftest import ScriptedModel


class SessionRecorder(Hook):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on(self, event: Event, ctx) -> None:
        if isinstance(event, SessionStarted | SessionEnded):
            self.events.append(type(event).__name__)


def build(*extra):
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))
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
    h = NexusAIHarness().use(ScriptedModel(Response(text="x")))
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

    # MaxIterations must halt a runaway tool loop; `recall` is trusted so no prompt.
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
    # A continued session must accumulate history across run() calls.
    class Counter(ScriptedModel):
        async def complete(self, history, ctx):
            users = sum(1 for m in history if m.role == "user")
            return Response(text=f"seen {users}")

    h = NexusAIHarness().use(AgenticLoop()).use(Counter())
    r1 = h.run_sync("first")
    assert r1.output == "seen 1"
    r2 = h.run_sync("second", session=r1.session)  # continue the conversation
    assert r2.output == "seen 2"
    assert r2.session is r1.session


class SlowStopLifecycle(Plugin, Lifecycle):
    """A lifecycle plugin whose stop() yields, to force stop() calls to interleave."""

    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    async def start(self, ctx) -> None:
        self.starts += 1

    async def stop(self) -> None:
        # Yield so a second stop() interleaves and hits the double-checked return.
        await asyncio.sleep(0)
        self.stops += 1


def test_concurrent_double_stop_tears_down_exactly_once():
    plugin = SlowStopLifecycle()

    async def go() -> None:
        h = NexusAIHarness().use(plugin)
        await h.start()
        assert plugin.starts == 1
        # Race two stop()s: the second must early-return after re-checking _started.
        await asyncio.gather(h.stop(), h.stop())
        assert plugin.stops == 1  # teardown ran exactly once despite the race
        assert h._started is False

    asyncio.run(go())


def test_result_exposes_run_state(tmp_path):
    from plugins import default_harness
    from plugins.hooks import CostState

    h = default_harness(ScriptedModel(Response(text="done")), memory_dir=str(tmp_path))
    result = h.run_sync("q")
    assert result.stop_reason == "completed"
    # cost is reachable via the returned session's typed state
    assert result.session.state(CostState).total == 0.0
