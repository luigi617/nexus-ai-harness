from __future__ import annotations

import asyncio

from nexus_ai_harness.core.events import Event, SessionEnded, SessionStarted
from nexus_ai_harness.core.response import Response
from nexus_ai_harness.harness import NexusAIHarness
from nexus_ai_harness.plugins.loops import AgenticLoop
from nexus_ai_harness.protocols.hook import Hook
from nexus_ai_harness.protocols.lifecycle import Lifecycle
from nexus_ai_harness.protocols.plugin import Plugin
from tests.conftest import FailingModel, ScriptedModel


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
    from nexus_ai_harness.plugins import default_harness

    h = default_harness(ScriptedModel(Response(text="hi")), memory_dir=str(tmp_path))
    assert h.run_sync("q").output == "hi"


def test_default_harness_workspace_registers_sandbox_and_fs_tools(tmp_path):
    from nexus_ai_harness.plugins import default_harness
    from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
    from nexus_ai_harness.protocols.sandbox import Sandbox
    from nexus_ai_harness.protocols.tool import Tool

    ws = tmp_path / "ws"
    h = default_harness(
        ScriptedModel(Response(text="hi")),
        memory_dir=str(tmp_path / "mem"),
        workspace=str(ws),
    )
    assert isinstance(h._registry.get(Sandbox), WorkspaceSandbox)
    tool_names = {t.name for t in h._registry.all(Tool)}
    assert {"read_file", "write_file", "list_dir", "shell"} <= tool_names


def test_default_harness_write_file_and_shell_are_not_auto_trusted(tmp_path):
    # Only the read-only tools bypass the approver; write_file and shell mutate
    # state and must resolve to ASK (regression for the trusted-list contract).
    from nexus_ai_harness.core.permission import PermissionVerdict
    from nexus_ai_harness.plugins import default_harness
    from nexus_ai_harness.protocols.permission import Permission

    h = default_harness(
        ScriptedModel(Response(text="hi")),
        memory_dir=str(tmp_path / "mem"),
        workspace=str(tmp_path / "ws"),
    )
    from tests.conftest import make_ctx

    ctx = make_ctx()  # AskUnless.check keys off the call name; ctx is unused
    ask = next(
        p for p in h._registry.all(Permission) if type(p).__name__ == "AskUnless"
    )
    verdict = lambda name: ask.check({"name": name}, ctx).verdict  # noqa: E731
    assert verdict("read_file") == PermissionVerdict.ALLOW
    assert verdict("list_dir") == PermissionVerdict.ALLOW
    assert verdict("write_file") == PermissionVerdict.ASK
    assert verdict("shell") == PermissionVerdict.ASK


def test_default_harness_session_store_registers_autosave(tmp_path):
    from nexus_ai_harness.plugins import FileSessionStore, default_harness
    from nexus_ai_harness.plugins.persistence.autosave import AutoSave
    from nexus_ai_harness.protocols.hook import Hook
    from nexus_ai_harness.protocols.session_store import SessionStore

    store = FileSessionStore(tmp_path / "sessions")
    h = default_harness(
        ScriptedModel(Response(text="hi")),
        memory_dir=str(tmp_path / "mem"),
        session_store=store,
    )
    assert h._registry.get(SessionStore) is store
    assert any(isinstance(hook, AutoSave) for hook in h._registry.all(Hook))


def test_default_harness_mcp_servers_registers_provider(tmp_path):
    from nexus_ai_harness.plugins import MCPServer, default_harness
    from nexus_ai_harness.protocols.tool_provider import ToolProvider

    h = default_harness(
        ScriptedModel(Response(text="hi")),
        memory_dir=str(tmp_path / "mem"),
        mcp_servers=[MCPServer(name="demo", command=["x"])],
    )
    assert h._registry.get(ToolProvider) is not None


def test_default_harness_bounds_a_runaway_loop(tmp_path):
    from nexus_ai_harness.plugins import default_harness

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
        async def complete(self, history, tools, ctx):
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
    from nexus_ai_harness.plugins import default_harness
    from nexus_ai_harness.plugins.hooks import CostState

    h = default_harness(ScriptedModel(Response(text="done")), memory_dir=str(tmp_path))
    result = h.run_sync("q")
    assert result.stop_reason == "completed"
    # cost is reachable via the returned session's typed state
    assert result.session.state(CostState).total == 0.0


def test_model_error_returns_a_result_and_still_ends_the_session():
    rec = SessionRecorder()
    model = FailingModel(RuntimeError("down"))
    h = NexusAIHarness().use(AgenticLoop()).use(model).use(rec)
    result = h.run_sync("q")
    assert result.stop_reason == "model_error"
    assert result.output == "stopped: model error: down"
    assert rec.events == ["SessionStarted", "SessionEnded"]
