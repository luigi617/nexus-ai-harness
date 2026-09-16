from __future__ import annotations

from core.response import Response
from harness import GraphAIHarness, Session
from plugins.loops import AgenticLoop
from protocols.model import Model
from tests.conftest import ScriptedModel


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

    h = GraphAIHarness().use(AgenticLoop()).use(Interrupting(session))
    result = h.run_sync("go", session=session)
    assert result.stop_reason == "interrupted"
    assert result.output == "stopped: interrupted"


def test_run_resets_stale_interrupt():
    session = Session()
    session.interrupt()  # stale request from before this turn
    h = GraphAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))
    result = h.run_sync("q", session=session)
    assert result.output == "hi"  # reset at run start, so it ran normally
