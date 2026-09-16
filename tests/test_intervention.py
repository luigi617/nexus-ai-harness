from __future__ import annotations

import asyncio

from core.response import Response
from core.spawn import SpawnState
from harness import NexusAIHarness, Session
from harness.context import RunContext
from harness.registry import Registry
from plugins.interventions import InjectMessage
from plugins.loops import AgenticLoop
from protocols.intervention import Intervention
from protocols.model import Model
from tests.conftest import ScriptedModel, make_ctx


def test_inject_message_is_an_intervention():
    assert isinstance(InjectMessage("hi"), Intervention)


def test_inject_message_adds_a_message():
    ctx = make_ctx()
    InjectMessage("focus on X").apply(ctx)
    assert ctx.history[-1].role == "user"
    assert ctx.history[-1].content == "focus on X"


def test_injected_message_steers_the_next_turn():
    # The model echoes the last user message; an injected message must reach it.
    class Echo(Model):
        async def complete(self, history, ctx) -> Response:
            last = next(m.content for m in reversed(history) if m.role == "user")
            return Response(text=f"answering: {last}")

    session = Session()
    session.submit(InjectMessage("use the metric system"))
    h = NexusAIHarness().use(AgenticLoop()).use(Echo())
    result = h.run_sync("convert 5 miles", session=session)
    # the injected message was appended after the user input, so it's the last
    assert result.output == "answering: use the metric system"


def test_interventions_are_drained_once():
    session = Session()
    session.submit(InjectMessage("a"))
    session.submit(InjectMessage("b"))
    assert len(session.take_interventions()) == 2
    assert session.take_interventions() == []  # drained


# --- interrupt propagation to subagents ----------------------------------


def test_fork_shares_interrupt_signal_with_child():
    parent = RunContext(Session(), Registry())
    child = parent.fork(None)
    assert child.interrupted is False
    parent._session.interrupt()  # interrupt the root
    assert child.interrupted is True  # propagates down to the subagent


def test_child_interrupt_stops_its_loop():
    parent = make_ctx(AgenticLoop(), ScriptedModel(Response(text="x")))
    child = parent.fork(None)
    parent._session.interrupt()  # root interrupt reaches the child
    result = asyncio.run(AgenticLoop().run(child))
    assert result.startswith("stopped: interrupted")


def test_child_has_its_own_empty_inbox():
    parent = RunContext(Session(), Registry())
    parent._session.submit(InjectMessage("root only"))
    child = parent.fork(None)
    assert child._session.take_interventions() == []  # interventions don't propagate
    assert child.state(SpawnState).depth == 1
