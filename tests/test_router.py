from __future__ import annotations

import asyncio

from core.response import Response
from plugins.loops import AgenticLoop
from plugins.routers import LLMRouter, StickyRouter
from protocols.model import Model
from protocols.router import Router
from tests.conftest import make_ctx


class FakeModel(Model):
    provider = "test"

    def __init__(self, name: str, description: str = "", reply: str = "") -> None:
        self.name = name
        self.description = description
        self.reply = reply or name
        self.calls = 0

    async def complete(self, history, ctx) -> Response:
        self.calls += 1
        return Response(text=self.reply, tool_calls=[])


# --- LLMRouter (candidates come from ctx.all(Model)) -----------------------


def test_llm_router_picks_the_model_the_decider_names():
    fast = FakeModel("fast", "cheap, simple tasks")
    smart = FakeModel("smart", "strong, hard tasks")
    decider = FakeModel("decider", reply="smart")  # names smart's model id tail
    ctx = make_ctx(fast, smart)
    assert asyncio.run(LLMRouter(decider).route([], ctx)) is smart


def test_llm_router_falls_back_to_first_on_no_match():
    fast = FakeModel("fast")
    smart = FakeModel("smart")
    decider = FakeModel("decider", reply="???")
    ctx = make_ctx(fast, smart)
    assert asyncio.run(LLMRouter(decider).route([], ctx)) is fast


def test_llm_router_defaults_decider_to_first_llm():
    fast = FakeModel("fast", reply="smart")  # first Model decides; names "smart"
    smart = FakeModel("smart")
    ctx = make_ctx(fast, smart)
    assert asyncio.run(LLMRouter().route([], ctx)) is smart


def test_llm_router_raises_without_llms():
    try:
        asyncio.run(LLMRouter().route([], make_ctx()))
        raised = False
    except LookupError:
        raised = True
    assert raised


def test_llm_router_is_a_router():
    assert isinstance(LLMRouter(), Router)


def test_llm_router_respects_provider_on_name_collision():
    # Two models share a name under different providers; the reply names one
    # fully-qualified id, so the provider must decide — not registration order.
    bedrock = FakeModel("claude")
    bedrock.provider = "bedrock"
    vertex = FakeModel("claude")
    vertex.provider = "vertex"
    assert LLMRouter._match("vertex$claude", [bedrock, vertex]) is vertex


def test_llm_router_prefers_most_specific_nested_name():
    # A nested name ("gpt-4") must not shadow the more specific "gpt-4o".
    gpt4 = FakeModel("gpt-4")
    gpt4o = FakeModel("gpt-4o")
    assert LLMRouter._match("gpt-4o", [gpt4, gpt4o]) is gpt4o


# --- StickyRouter --------------------------------------------------------


class FlipRouter(Router):
    """Returns a different Model each call — to prove Sticky pins the first."""

    def __init__(self, models):
        self._models = models
        self.calls = 0

    def route(self, history, ctx):
        model = self._models[self.calls % len(self._models)]
        self.calls += 1
        return model


def test_sticky_router_reuses_the_first_choice():
    a, b = FakeModel("a"), FakeModel("b")
    inner = FlipRouter([a, b])
    sticky = StickyRouter(inner)
    ctx = make_ctx()
    first = asyncio.run(sticky.route([], ctx))
    second = asyncio.run(sticky.route([], ctx))
    assert first is a and second is a
    assert inner.calls == 1


# --- loop integration ----------------------------------------------------


def test_loop_routes_via_llm_router():
    fast = FakeModel("fast")
    smart = FakeModel("smart")
    decider = FakeModel("decider", reply="smart")
    ctx = make_ctx(AgenticLoop(), fast, smart, LLMRouter(decider))
    assert asyncio.run(AgenticLoop().run(ctx)) == "smart"


def test_sticky_over_llm_decides_once_then_reuses():
    fast = FakeModel("fast")
    smart = FakeModel("smart")
    replies = iter(["smart", "fast"])

    class FlipDecider(Model):
        async def complete(self, history, ctx):
            return Response(text=next(replies))

    ctx = make_ctx(fast, smart)
    router = StickyRouter(LLMRouter(FlipDecider()))
    a = asyncio.run(router.route([], ctx))
    b = asyncio.run(router.route([], ctx))
    assert a is b is smart
