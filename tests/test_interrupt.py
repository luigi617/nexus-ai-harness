from __future__ import annotations

from core.events import Event, LoopStopped
from core.response import Response
from harness import NexusAIHarness, Session
from plugins.loops import AgenticLoop
from protocols.hook import Hook
from protocols.model import Model
from protocols.tool import Tool
from tests.conftest import ScriptedModel


class RecordingHook(Hook):
    def __init__(self) -> None:
        self.seen: list[Event] = []

    def on(self, event: Event, ctx: object) -> None:
        self.seen.append(event)


def test_session_interrupt_sets_flag():
    s = Session()
    assert s.interrupted is False
    s.interrupt()
    assert s.interrupted is True


def test_interrupt_stops_between_iterations():
    session = Session()

    class Interrupting(Model):
        """Requests a stop on its first call, then would keep looping."""

        def __init__(self, s: Session) -> None:
            self._s = s

        async def complete(self, history, ctx) -> Response:
            self._s.interrupt()
            return Response(
                text="", tool_calls=[{"name": "noop", "id": "1", "arguments": {}}]
            )

    h = NexusAIHarness().use(AgenticLoop()).use(Interrupting(session))
    result = h.run_sync("go", session=session)
    assert result.stop_reason == "interrupted"
    assert result.output == "stopped: interrupted"


def test_run_resets_stale_interrupt():
    session = Session()
    session.interrupt()  # stale request from before this turn
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))
    result = h.run_sync("q", session=session)
    assert result.output == "hi"  # reset at run start, so it ran normally


def test_interrupt_raised_mid_iteration_stops_on_the_next_turn_after_the_tool_ran():
    # The loop checks interrupts only at each iteration's top, so a mid-tool one defers.
    session = Session()

    class InterruptingTool(Tool):
        name = "stop"
        description = ""

        def __init__(self, s: Session) -> None:
            self.parameters: dict = {}
            self._s = s
            self.calls: list[dict] = []

        def run(self, arguments: dict, ctx) -> str:
            self.calls.append(arguments)
            self._s.interrupt()  # requested mid-iteration, during tool execution
            return "did work"

    class CallToolThenLoop(Model):
        def __init__(self) -> None:
            self.n = 0

        async def complete(self, history, ctx) -> Response:
            self.n += 1
            if self.n == 1:
                return Response(
                    text="", tool_calls=[{"name": "stop", "id": "1", "arguments": {}}]
                )
            return Response(text="should never be reached")

    tool = InterruptingTool(session)
    model = CallToolThenLoop()
    hook = RecordingHook()
    h = NexusAIHarness().use(AgenticLoop()).use(model).use(tool).use(hook)
    result = h.run_sync("go", session=session)

    assert result.stop_reason == "interrupted"
    assert result.output == "stopped: interrupted"
    assert tool.calls == [{}]  # the tool's turn ran to completion
    assert model.n == 1  # the loop stopped before a second model call
    # The tool's own turn still completed: its result message was recorded.
    assert any(m.role == "tool" and m.content == "did work" for m in session.history)
    stops = [e for e in hook.seen if isinstance(e, LoopStopped)]
    assert len(stops) == 1
    assert stops[0].reason == "interrupted"


def test_interrupt_during_a_completed_turn_does_not_preempt_the_finish():
    # An interrupt during a response with no tool_calls is ignored; the turn finishes.
    session = Session()

    class InterruptWhileFinishing(Model):
        def __init__(self, s: Session) -> None:
            self._s = s

        async def complete(self, history, ctx) -> Response:
            self._s.interrupt()  # set mid-turn, but this turn has no tool calls
            return Response(text="final answer")

    hook = RecordingHook()
    h = (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(InterruptWhileFinishing(session))
        .use(hook)
    )
    result = h.run_sync("go", session=session)

    assert result.output == "final answer"
    assert result.stop_reason == "completed"
    stops = [e for e in hook.seen if isinstance(e, LoopStopped)]
    assert [s.reason for s in stops] == ["completed"]
