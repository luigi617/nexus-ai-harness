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
    # Same name, two providers; the fully-qualified id decides, not registration order.
    bedrock = FakeModel("claude")
    bedrock.provider = "bedrock"
    vertex = FakeModel("claude")
    vertex.provider = "vertex"
    assert LLMRouter._match("vertex$claude", [bedrock, vertex]) is vertex


def test_llm_router_exact_bare_name_beats_nested_prefix():
    # Tier 3: exact bare-name reply wins even when another name ("gpt-4") is a prefix.
    gpt4 = FakeModel("gpt-4")
    gpt4o = FakeModel("gpt-4o")
    assert LLMRouter._match("gpt-4o", [gpt4, gpt4o]) is gpt4o


def test_match_tier2_id_substring_in_prose_reply():
    # Tier 2: a fully-qualified id embedded in a chatty reply (not an exact match).
    fast = FakeModel("fast")
    smart = FakeModel("smart")  # _id == "test$smart"
    assert LLMRouter._match("I choose test$smart for this", [fast, smart]) is smart


def test_match_tier2_longest_id_wins_when_both_substrings():
    # Both ids appear as substrings of the reply; the longest (most specific) wins.
    gpt4 = FakeModel("gpt-4")  # _id == "test$gpt-4"
    gpt4o = FakeModel("gpt-4o")  # _id == "test$gpt-4o" (superstring)
    assert LLMRouter._match("go with test$gpt-4o please", [gpt4, gpt4o]) is gpt4o


def test_match_tier4_bare_name_substring_in_prose_reply():
    # Tier 4: no id present, bare name appears as a substring of the reply.
    fast = FakeModel("fast")
    smart = FakeModel("smart")
    assert LLMRouter._match("use the fast model please", [fast, smart]) is fast


def test_match_tier4_longest_bare_name_wins():
    gpt4 = FakeModel("gpt-4")
    gpt4o = FakeModel("gpt-4o")
    assert LLMRouter._match("go with gpt-4o", [gpt4, gpt4o]) is gpt4o


def test_route_resolves_via_id_substring_path():
    # End-to-end: a prose decider reply drives the substring (tier 2) path.
    fast = FakeModel("fast")
    smart = FakeModel("smart")
    decider = FakeModel("decider", reply="I think test$smart is best here")
    ctx = make_ctx(fast, smart)
    assert asyncio.run(LLMRouter(decider).route([], ctx)) is smart


class RecordingDecider(Model):
    """A decider that records the request it was asked to complete."""

    provider = "test"
    name = "decider"

    def __init__(self, reply: str = "fast") -> None:
        self.reply = reply
        self.seen: list | None = None

    async def complete(self, history, ctx) -> Response:
        self.seen = list(history)
        return Response(text=self.reply)


def test_route_extracts_last_user_task_and_builds_decider_request():
    from core.message import Message

    fast = FakeModel("fast", "cheap tasks")
    smart = FakeModel("smart", "hard tasks")
    decider = RecordingDecider(reply="fast")
    ctx = make_ctx(fast, smart)
    history = [
        Message(role="system", content="be helpful"),
        Message(role="user", content="the FIRST task"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="the LATEST task"),
    ]
    chosen = asyncio.run(LLMRouter(decider).route(history, ctx))
    assert chosen is fast

    assert decider.seen is not None
    system_msg, user_msg = decider.seen
    # The router prompt lists every candidate by id (+ description).
    assert system_msg.role == "system"
    assert "test$fast" in system_msg.content
    assert "test$smart" in system_msg.content
    assert "cheap tasks" in system_msg.content
    # The task handed to the decider is the MOST RECENT user turn, not the first.
    assert user_msg.role == "user"
    assert user_msg.content == "the LATEST task"


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
