from __future__ import annotations

import asyncio

from core.events import (
    Event,
    IterationStarted,
    LoopStopped,
)
from core.message import Message
from core.response import Response
from core.run import RunState
from plugins.loops import ChatLoop
from protocols.context_manager import ContextManager
from protocols.hook import Hook
from protocols.model import Model
from protocols.router import Router
from tests.conftest import ScriptedModel, make_ctx


def test_chat_loop_returns_text_and_appends_assistant_message():
    ctx = make_ctx(ScriptedModel(Response(text="hello")))
    result = asyncio.run(ChatLoop().run(ctx))
    assert result == "hello"
    assert ctx.history[-1].role == "assistant"
    assert ctx.history[-1].content == "hello"


def test_chat_loop_raises_without_provider():
    try:
        asyncio.run(ChatLoop().run(make_ctx()))
        raised = False
    except LookupError:
        raised = True
    assert raised


# --- ContextManager middleware chain (finding 1) -------------------------


class AppendMarker(ContextManager):
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def process(self, history, ctx):
        return [*history, Message(role="system", content=self.marker)]


class UppercaseLast(ContextManager):
    def process(self, history, ctx):
        if not history:
            return list(history)
        out = list(history)
        last = out[-1]
        out[-1] = Message(role=last.role, content=last.content.upper())
        return out


def test_chat_loop_applies_context_manager_chain_in_order():
    model = ScriptedModel(Response(text="done"))
    ctx = make_ctx(model, AppendMarker("cm1"), UppercaseLast())
    ctx.add_message(Message(role="user", content="hi"))
    asyncio.run(ChatLoop().run(ctx))
    # AppendMarker first, then UppercaseLast uppercased the appended marker
    assert [m.content for m in model.calls[0]] == ["hi", "CM1"]
    assert "cm1" not in [m.content for m in ctx.history]


# --- Router branch (finding 5) -------------------------------------------


class PickModel(Router):
    def __init__(self, target: Model) -> None:
        self._target = target

    def route(self, history, ctx):
        return self._target


class NoneRouter(Router):
    def route(self, history, ctx):
        return None


def test_chat_loop_routes_to_the_selected_model():
    a = ScriptedModel(Response(text="from-a"))
    b = ScriptedModel(Response(text="from-b"))
    ctx = make_ctx(a, b, PickModel(b))
    assert asyncio.run(ChatLoop().run(ctx)) == "from-b"
    assert len(b.calls) == 1 and a.calls == []


def test_chat_loop_router_returning_none_raises_lookup_error():
    ctx = make_ctx(ScriptedModel(Response(text="x")), NoneRouter())
    raised = False
    msg = ""
    try:
        asyncio.run(ChatLoop().run(ctx))
    except LookupError as exc:
        raised = True
        msg = str(exc)
    assert raised
    assert msg == "no model registered"


# --- completed-state and event contract (finding 9) ----------------------


class _EventRecorder(Hook):
    def __init__(self) -> None:
        self.events: list[Event] = []

    def on(self, event: Event, ctx) -> None:
        self.events.append(event)


def test_chat_loop_sets_completed_stop_reason():
    ctx = make_ctx(ScriptedModel(Response(text="hi")))
    asyncio.run(ChatLoop().run(ctx))
    assert ctx.state(RunState).stop_reason == "completed"


def test_chat_loop_emits_only_call_response_and_message_events():
    rec = _EventRecorder()
    ctx = make_ctx(ScriptedModel(Response(text="hi")), rec)
    asyncio.run(ChatLoop().run(ctx))
    kinds = {type(e).__name__ for e in rec.events}
    assert kinds == {"ModelCallStarted", "ResponseReceived", "MessageAdded"}
    # unlike AgenticLoop, ChatLoop emits no LoopStopped / IterationStarted
    assert not any(isinstance(e, (LoopStopped, IterationStarted)) for e in rec.events)


def test_chat_loop_ignores_a_pre_set_interrupt():
    ctx = make_ctx(ScriptedModel(Response(text="hi")))
    ctx._session.interrupt()
    result = asyncio.run(ChatLoop().run(ctx))
    assert result == "hi"  # single-shot; the interrupt is not honored
    assert ctx.state(RunState).stop_reason == "completed"
