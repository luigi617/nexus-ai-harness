from __future__ import annotations

import asyncio
import itertools
import random

import pytest

from nexus_ai_harness.core.events import (
    ContextCompacted,
    ModelCallStarted,
    ResponseReceived,
)
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.response import Response
from nexus_ai_harness.harness import NexusAIHarness
from nexus_ai_harness.harness.context import RunContext
from nexus_ai_harness.harness.registry import Registry
from nexus_ai_harness.plugins.context_manager import (
    CharTokenEstimator,
    SummarizingContextManager,
    SummaryState,
)
from nexus_ai_harness.plugins.loops import ChatLoop
from nexus_ai_harness.protocols.context_manager import ContextManager
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.token_estimator import TokenEstimator
from tests.conftest import ScriptedModel, make_ctx


class SummarizerModel(Model):
    def __init__(self) -> None:
        self.calls = 0
        self.bodies: list[str] = []

    async def complete(self, history, tools, ctx) -> Response:
        self.calls += 1
        self.bodies.append(history[-1].content)
        return Response(text=f"RECAP-{self.calls}")


def history(n: int) -> list[Message]:
    msgs = [Message(role="system", content="sys"), Message(role="user", content="task")]
    for i in range(n):
        msgs.append(Message(role="user" if i % 2 else "assistant", content=f"m{i}"))
    return msgs


def process(cm, hist, ctx):
    return asyncio.run(cm.process(hist, ctx))


def capture(ctx) -> list[ContextCompacted]:
    events: list[ContextCompacted] = []
    ctx.on(ContextCompacted, lambda event, _ctx: events.append(event))
    return events


def assert_alternates(messages: list[Message]) -> None:
    # Strict-alternation backends reject two adjacent user turns.
    roles = [m.role for m in messages if m.role != "system"]
    for a, b in itertools.pairwise(roles):
        assert not (a == "user" and b == "user"), roles


def test_below_threshold_is_noop():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    out = process(cm, history(3), ctx)
    assert len(out) == 5  # unchanged
    assert model.calls == 0  # never summarized


def test_compaction_keeps_prefix_summary_and_tail():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    out = process(cm, history(10), ctx)
    assert model.calls == 1
    assert out[0].role == "system"
    # Summary and the tail's leading user turn merge into the head user message.
    assert out[1].role == "user"
    assert out[1].content.startswith("task\n\n[Conversation summary so far]")
    assert "RECAP" in out[1].content
    assert out[1].content.endswith("m7")
    assert [m.content for m in out[2:]] == ["m8", "m9"]
    assert_alternates(out)


def test_cached_summary_reused_without_new_call():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    h = history(10)
    out1 = process(cm, list(h), ctx)
    h2 = [
        *h,
        Message(role="assistant", content="m10"),
        Message(role="user", content="m11"),
    ]
    out2 = process(cm, list(h2), ctx)
    assert model.calls == 1  # no re-summarize under threshold
    # cache prefix (head + summary) is byte-stable across turns
    assert [(m.role, m.content) for m in out1[:2]] == [
        (m.role, m.content) for m in out2[:2]
    ]


def test_second_fold_accumulates_prior_summary():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    process(cm, history(10), ctx)  # fold 1 -> RECAP-1
    out = process(cm, history(20), ctx)  # tail past budget again -> fold 2
    assert model.calls == 2
    assert "RECAP-2" in out[1].content  # summary advanced
    assert "RECAP-1" in model.bodies[1]  # prior recap folded into the new one


def test_drop_fallback_when_no_provider():
    ctx = make_ctx()
    events = capture(ctx)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    out = process(cm, history(10), ctx)
    # Over budget with nothing to summarize with: drop the middle, keep the head.
    assert out[0].content == "sys"
    assert out[1].content.startswith("task\n\n[7 earlier messages omitted")
    assert [m.content for m in out[2:]] == ["m8", "m9"]
    assert_alternates(out)
    assert [e.strategy for e in events] == ["drop"]
    assert events[0].fallback
    assert "summarization unavailable" in events[0].reason
    state = cm.state(ctx)
    assert state.fallbacks == 1 and state.last_fallback.startswith("drop")


def test_boundary_skips_orphaned_tool_results():
    # Kept tail must not begin with a tool result whose parent tool_use was folded.
    hist = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="u1"),
        Message(role="assistant", content="", tool_calls=[{"id": "c", "name": "t"}]),
        Message(role="tool", content="r1", tool_use_id="c", name="t"),
        Message(role="user", content="u2"),
    ]
    ctx = make_ctx(SummarizerModel())
    cm = SummarizingContextManager(max_messages=3, keep_recent=2)
    out = process(cm, hist, ctx)
    tail = out[out.index(next(m for m in out if "RECAP" in m.content)) + 1 :]
    assert not tail or tail[0].role != "tool"


def test_summary_transcript_includes_tool_call_intent():
    # Tool-only assistant turns (content="" with tool_calls) must not be dropped.
    rendered = SummarizingContextManager._render(
        Message(
            role="assistant",
            content="",
            tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
        )
    )
    assert rendered is not None
    assert "search" in rendered and "tool_call" in rendered


class RaisingModel(Model):
    """A model whose complete() always raises — to prove _summarize is resilient."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, history, tools, ctx) -> Response:
        self.calls += 1
        raise RuntimeError("backend exploded")


def test_summarize_failure_drops_turns_without_caching_a_summary():
    model = RaisingModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    hist = history(10)
    out = process(cm, hist, ctx)

    assert model.calls == 1  # it did try
    # No summary injected; the deterministic drop fallback bounds the history.
    assert not any("summary" in (m.content or "").lower() for m in out)
    assert len(out) < len(hist)
    assert "omitted to fit the context window" in out[1].content
    # No partial corruption of the cached summary state.
    state = cm.state(ctx)
    assert state.upto == 0
    assert state.text == ""
    assert state.fallbacks == 1


class FlakySummarizer(Model):
    """Raises on its first summarize, then succeeds — to prove recovery."""

    def __init__(self) -> None:
        self.calls = 0
        self.bodies: list[str] = []

    async def complete(self, history, tools, ctx) -> Response:
        self.calls += 1
        self.bodies.append(history[-1].content)
        if self.calls == 1:
            raise RuntimeError("backend exploded")
        return Response(text=f"RECAP-{self.calls}")


def test_successful_compaction_still_works_after_a_failure():
    # A failed summarize must not poison later runs: recovery compacts normally.
    model = FlakySummarizer()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)

    failed = process(cm, history(10), ctx)
    assert model.calls == 1
    assert not any("RECAP" in (m.content or "") for m in failed)  # no compaction

    # The drop is sticky, so summarization retries once the budget is hit again.
    out = process(cm, history(20), ctx)
    assert model.calls == 2  # retried on the recovered provider
    assert "m0" in model.bodies[1]  # and folded the turns dropped earlier
    assert "RECAP" in out[1].content  # and compacted this time
    assert "omitted" not in out[1].content  # the summary covers dropped turns


def test_constructor_rejects_keep_recent_ge_max_messages():
    with pytest.raises(ValueError):
        SummarizingContextManager(max_messages=5, keep_recent=5)
    with pytest.raises(ValueError):
        SummarizingContextManager(max_messages=5, keep_recent=6)


def test_render_drops_empty_message():
    assert (
        SummarizingContextManager._render(
            Message(role="assistant", content="", tool_calls=[])
        )
        is None
    )


def test_empty_messages_omitted_from_summary_transcript():
    # An empty assistant turn in the folded range must not add a transcript line.
    hist = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="assistant", content="line-a"),
        Message(role="assistant", content="", tool_calls=[]),  # dropped
        Message(role="assistant", content="line-c"),
        Message(role="user", content="line-d"),
    ]
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=3, keep_recent=1)
    process(cm, hist, ctx)
    # bodies[0] is the transcript the summarizer received for the folded slice.
    assert model.bodies[0] == "assistant: line-a\nassistant: line-c"


# --- token budgeting -----------------------------------------------------------


def big(n_tokens: int) -> str:
    return "x" * (n_tokens * 4)  # CharTokenEstimator: ~4 chars per token


def huge_history() -> list[Message]:
    return [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="assistant", content=big(3000)),
        Message(role="user", content=big(3000)),
        Message(role="assistant", content=big(3000)),
        Message(role="user", content="latest"),
    ]


def test_token_budget_compacts_few_huge_messages():
    model = SummarizerModel()
    ctx = make_ctx(model)
    events = capture(ctx)
    cm = SummarizingContextManager(max_messages=40, max_tokens=4000)
    out = process(cm, huge_history(), ctx)
    assert model.calls >= 1  # message count alone would never have triggered
    assert [e.strategy for e in events] == ["summarize"]
    assert events[0].tokens_after < events[0].tokens_before
    assert out[0].content == "sys"
    assert "RECAP" in out[1].content and out[1].content.endswith("latest")
    assert len(out) == 2
    assert cm._estimate(out, ctx) <= 4000


def test_token_budget_below_threshold_is_noop():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=None, max_tokens=50_000)
    hist = huge_history()
    assert process(cm, hist, ctx) is hist
    assert model.calls == 0


def test_max_messages_none_uses_tokens_only():
    ctx = make_ctx(SummarizerModel())
    cm = SummarizingContextManager(max_messages=None, max_tokens=10_000)
    hist = history(200)  # many tiny messages fit the token budget
    assert process(cm, hist, ctx) is hist


def test_reported_input_tokens_anchor_the_estimate():
    # The provider saw far more tokens (e.g. tool schemas) than the heuristic.
    model = SummarizerModel()
    ctx = make_ctx(model)
    events = capture(ctx)
    cm = SummarizingContextManager(max_messages=None, max_tokens=1000)
    asyncio.run(cm.start(ctx))
    hist = history(6)
    assert process(cm, hist, ctx) is hist  # heuristic alone: well under budget

    ctx.emit(ModelCallStarted(list(hist)))
    ctx.emit(ResponseReceived(Response(text="ok", usage={"input_tokens": 900})))
    hist.append(Message(role="assistant", content="m6"))
    hist.append(Message(role="user", content="q" * 400))
    out = process(cm, hist, ctx)
    assert model.calls == 1  # the reported usage pushed it over budget
    assert len(out) < len(hist)
    assert events[0].strategy == "summarize"
    assert cm._estimate(out, ctx) <= 1000


def test_overhead_alone_over_budget_reports_overflow_without_summarizing():
    # Tool schemas alone exceed the budget: folding history can't help.
    model = SummarizerModel()
    ctx = make_ctx(model)
    events = capture(ctx)
    cm = SummarizingContextManager(max_messages=None, max_tokens=1000)
    asyncio.run(cm.start(ctx))
    hist = history(6)
    ctx.emit(ModelCallStarted(list(hist)))
    ctx.emit(ResponseReceived(Response(text="ok", usage={"input_tokens": 5000})))
    for i in range(6):
        hist.append(Message(role="assistant" if i % 2 == 0 else "user", content="m"))
        process(cm, list(hist), ctx)
    assert model.calls == 0
    assert [e.strategy for e in events] == ["overflow"]


def test_small_reported_usage_keeps_noop():
    ctx = make_ctx(SummarizerModel())
    cm = SummarizingContextManager(max_messages=None, max_tokens=1000)
    asyncio.run(cm.start(ctx))
    hist = history(6)
    ctx.emit(ModelCallStarted(list(hist)))
    ctx.emit(ResponseReceived(Response(usage={"input_tokens": 120})))
    hist.append(Message(role="assistant", content="m6"))
    assert cm._estimate(hist, ctx) > 120  # grows by the new message's estimate
    assert process(cm, hist, ctx) is hist


def test_cached_input_tokens_count_toward_usage():
    ctx = make_ctx()
    cm = SummarizingContextManager(max_messages=None, max_tokens=1000)
    asyncio.run(cm.start(ctx))
    hist = history(2)
    ctx.emit(ModelCallStarted(list(hist)))
    usage = {"input_tokens": 10, "cache_read_input_tokens": 900}
    ctx.emit(ResponseReceived(Response(usage=usage)))
    assert cm._estimate(hist, ctx) == 910


def test_usage_is_recorded_through_a_started_harness():
    model = ScriptedModel(Response(text="done", usage={"input_tokens": 777}))
    cm = SummarizingContextManager(max_messages=None, max_tokens=100_000)
    harness = NexusAIHarness().use(ChatLoop()).use(cm).use(model)
    result = harness.run_sync("hi")
    ctx = RunContext(result.session, Registry())
    # Estimate is the reported prompt plus the assistant reply added since.
    reply = CharTokenEstimator().estimate([result.session.history[-1]])
    assert cm._estimate(result.session.history, ctx) == 777 + reply


class FixedEstimator(TokenEstimator):
    def __init__(self, per_message: int) -> None:
        self.per_message = per_message

    def estimate(self, messages: list[Message]) -> int:
        return self.per_message * len(messages)


def test_estimator_argument_is_used():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(
        max_messages=None, max_tokens=500, estimator=FixedEstimator(100)
    )
    process(cm, history(6), ctx)  # 8 messages * 100 > 500
    assert model.calls == 1


def test_registered_estimator_is_resolved():
    model = SummarizerModel()
    ctx = make_ctx(model, FixedEstimator(100))
    cm = SummarizingContextManager(max_messages=None, max_tokens=500)
    process(cm, history(6), ctx)
    assert model.calls == 1


def test_char_estimator_counts_tool_call_arguments():
    est = CharTokenEstimator(per_message=0)
    plain = Message(role="assistant", content="")
    call = Message(
        role="assistant",
        content="",
        tool_calls=[{"name": "write", "arguments": {"body": "y" * 400}}],
    )
    assert est.estimate([plain]) == 0
    assert est.estimate([call]) > 100


def test_char_estimator_rejects_non_positive_ratio():
    with pytest.raises(ValueError):
        CharTokenEstimator(chars_per_token=0)


# --- fallback --------------------------------------------------------------------


def test_drop_fallback_is_sticky_until_budget_is_hit_again():
    ctx = make_ctx()
    events = capture(ctx)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    hist = history(10)
    first = process(cm, hist, ctx)
    hist.append(Message(role="assistant", content="m10"))
    second = process(cm, hist, ctx)
    assert len(events) == 1  # no re-drop while within budget
    assert [(m.role, m.content) for m in second[:2]] == [
        (m.role, m.content) for m in first[:2]
    ]
    assert second[-1].content == "m10"


def test_truncate_fallback_caps_oversized_tool_result():
    hist = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="assistant", content="", tool_calls=[{"id": "c", "name": "t"}]),
        Message(role="tool", content=big(20_000), tool_use_id="c", name="t"),
    ]
    ctx = make_ctx()  # no summarizer: only the deterministic fallback can help
    events = capture(ctx)
    cm = SummarizingContextManager(max_messages=None, max_tokens=2000)
    out = process(cm, hist, ctx)
    assert [e.strategy for e in events] == ["truncate"]
    assert cm._estimate(out, ctx) <= 2000
    assert "characters truncated to fit the context window" in out[-1].content
    # The pair survives intact, and the head is untouched.
    assert out[-2].tool_calls[0]["id"] == out[-1].tool_use_id == "c"
    assert [m.content for m in out[:2]] == ["sys", "task"]
    assert hist[-1].content == big(20_000)  # the session history is not mutated


def test_truncate_shrinks_largest_non_tool_message_last():
    hist = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="assistant", content="ok"),
        Message(role="user", content=big(10_000)),
    ]
    ctx = make_ctx(RaisingModel())
    cm = SummarizingContextManager(max_messages=None, max_tokens=1000)
    out = process(cm, hist, ctx)
    assert cm._estimate(out, ctx) <= 1000
    assert out[1].content.startswith("task")
    assert "truncated" in out[-1].content
    assert cm.state(ctx).last_fallback.startswith("truncate")


def test_small_head_is_never_dropped_or_truncated():
    hist = huge_history()
    ctx = make_ctx()
    cm = SummarizingContextManager(max_messages=None, max_tokens=500)
    out = process(cm, hist, ctx)
    assert out[0].content == "sys"
    assert out[1].content.startswith("task\n\n")
    assert_alternates(out)


# --- role alternation ------------------------------------------------------------


def test_no_head_user_places_summary_before_tail():
    hist = [Message(role="system", content="sys")]
    for i in range(10):
        hist.append(Message(role="assistant" if i % 2 else "user", content=f"m{i}"))
    ctx = make_ctx(SummarizerModel())
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    out = process(cm, hist, ctx)
    assert out[0].role == "system"
    assert out[1].role == "user" and "RECAP" in out[1].content
    assert_alternates(out)


def test_summary_is_not_merged_into_assistant_tail_start():
    ctx = make_ctx(SummarizerModel())
    cm = SummarizingContextManager(max_messages=6, keep_recent=4)
    out = process(cm, history(10), ctx)  # tail starts on an assistant turn
    assert out[1].content.endswith("RECAP-1")
    assert out[2].role == "assistant"
    assert_alternates(out)


# --- tool pairing ----------------------------------------------------------------


def assert_tool_pairs_intact(messages: list[Message]) -> None:
    seen: set[str] = set()
    for m in messages:
        if m.role == "tool":
            assert m.tool_use_id in seen, f"orphaned tool result {m.tool_use_id}"
        for call in m.tool_calls:
            seen.add(call["id"])


def test_cut_never_orphans_a_non_adjacent_tool_result():
    # A user turn between a call and its result must not become the cut point.
    hist = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="u1"),
        Message(role="assistant", content="", tool_calls=[{"id": "c", "name": "t"}]),
        Message(role="user", content="interjection"),
        Message(role="tool", content="r1", tool_use_id="c", name="t"),
        Message(role="assistant", content="a2"),
    ]
    ctx = make_ctx(SummarizerModel())
    # keep_recent=3 alone would start the tail on "interjection", orphaning r1.
    cm = SummarizingContextManager(max_messages=4, keep_recent=3)
    out = process(cm, hist, ctx)
    assert_tool_pairs_intact(out)
    assert out[-1].content == "a2"


def test_parallel_tool_results_are_kept_together():
    hist = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="assistant", content="a0"),
        Message(role="user", content="u0"),
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "a", "name": "t"}, {"id": "b", "name": "t"}],
        ),
        Message(role="tool", content="ra", tool_use_id="a", name="t"),
        Message(role="tool", content="rb", tool_use_id="b", name="t"),
        Message(role="assistant", content="done"),
    ]
    for ctx in (make_ctx(SummarizerModel()), make_ctx()):
        cm = SummarizingContextManager(max_messages=3, keep_recent=2)
        out = process(cm, hist, ctx)
        assert_tool_pairs_intact(out)


def test_randomized_histories_keep_tool_pairs_and_head():
    rng = random.Random(7)
    for trial in range(60):
        head = "t" * rng.choice([1, 1, rng.randint(1, 12_000)])
        hist = [
            Message(role="system", content="sys"),
            Message(role="user", content=head),
        ]
        call = 0
        length = rng.randint(8, 40)
        while len(hist) < length:
            if rng.random() < 0.5:
                ids = [f"c{call + k}" for k in range(rng.randint(1, 3))]
                call += len(ids)
                body = "w" * rng.choice([0, 0, 6000])
                hist.append(
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=[
                            {"id": i, "name": "t", "arguments": {"body": body}}
                            for i in ids
                        ],
                    )
                )
                for i in ids:
                    size = rng.choice([10, 10, 4000])
                    hist.append(Message(role="tool", content="r" * size, tool_use_id=i))
            else:
                hist.append(
                    Message(role="assistant", content="a" * rng.randint(1, 900))
                )
                hist.append(Message(role="user", content="u" * rng.randint(1, 900)))
        model = SummarizerModel()
        ctx = make_ctx(model) if trial % 2 else make_ctx()
        events = capture(ctx)
        cm = SummarizingContextManager(
            max_messages=rng.choice([None, 8]), keep_recent=4, max_tokens=1500
        )
        asyncio.run(cm.start(ctx))
        # Fixed provider-side overhead (e.g. tool schemas) the history can't show.
        overhead = rng.choice([0, 150, 400])
        grown: list[Message] = []
        for m in hist:  # replay turn by turn, like the loop does
            grown.append(m)
            out = process(cm, list(grown), ctx)
            assert_tool_pairs_intact(out)
            assert_alternates(out)
            assert out[0].content == "sys"
            if len(grown) > 1:
                assert out[1].content.startswith("t")
            assert cm._estimate(out, ctx) <= 1500
            newest = grown[-1]  # kept as is, or merged into the seam user turn
            assert out[-1].id == newest.id or newest.content[:100] in out[-1].content
            ctx.emit(ModelCallStarted(list(out)))
            sent = CharTokenEstimator().estimate(out) + overhead
            ctx.emit(ResponseReceived(Response(usage={"input_tokens": sent})))
        assert "overflow" not in [e.strategy for e in events]
        # Each fold leaves room to grow, so summarizing is not a per-turn cost.
        assert cm.state(ctx).summaries <= len(hist) // 2
        folds = [e for e in events if e.strategy == "summarize"]
        assert all(e.tokens_after < e.tokens_before for e in folds)


# --- summary length and drift ------------------------------------------------------


class VerboseSummarizer(Model):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, history, tools, ctx) -> Response:
        self.prompts.append(history[0].content)
        return Response(text="\n".join(f"fact {i} " + "z" * 60 for i in range(400)))


def test_summary_length_is_instructed_and_enforced():
    model = VerboseSummarizer()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(
        max_messages=6, keep_recent=3, max_summary_tokens=200
    )
    process(cm, history(10), ctx)
    process(cm, history(30), ctx)  # a re-fold must not grow the recap either
    assert len(model.prompts) == 2
    assert all("under 200 tokens" in p for p in model.prompts)
    text = cm.state(ctx).text
    assert CharTokenEstimator().estimate([Message(role="user", content=text)]) <= 220
    assert text.endswith("[recap truncated to its length limit]")


def test_large_fold_is_summarized_in_chunks():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=None, max_tokens=4000, keep_recent=2)
    hist = [Message(role="system", content="sys"), Message(role="user", content="t")]
    for i in range(12):
        hist.append(Message(role="assistant", content=f"a{i} " + "q" * 3000))
        hist.append(Message(role="user", content=f"u{i}"))
    process(cm, hist, ctx)
    assert model.calls > 1  # the slice exceeded one chunk
    assert "Existing recap:\nRECAP-1" in model.bodies[1]  # rolling recap
    assert all(len(b) < 4000 * 4 for b in model.bodies)


def test_render_truncates_huge_messages():
    rendered = SummarizingContextManager._render(
        Message(role="tool", content="r" * 50_000)
    )
    assert rendered is not None
    assert len(rendered) < 5000
    assert "characters truncated" in rendered


# --- state and configuration -------------------------------------------------------


def test_rewound_history_resets_cached_cuts():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    process(cm, history(20), ctx)
    short = history(2)
    out = process(cm, short, ctx)
    assert out is short  # stale summary state is not applied to a new history
    assert cm.state(ctx).text == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_messages": None, "max_tokens": None},
        {"keep_recent": -1},
        {"max_tokens": 0},
        {"max_summary_tokens": 0},
    ],
)
def test_constructor_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        SummarizingContextManager(**kwargs)


def test_default_harness_budgets_tokens_without_changing_small_runs(tmp_path):
    from nexus_ai_harness.plugins import default_harness

    harness = default_harness(ScriptedModel(), memory_dir=str(tmp_path))
    cm = harness._registry.get(ContextManager)
    assert isinstance(cm, SummarizingContextManager)
    assert cm._max_tokens == 100_000
    assert cm._max_messages == 40
    hist = history(10)
    assert process(cm, hist, make_ctx(ScriptedModel())) is hist

    off = default_harness(
        ScriptedModel(), memory_dir=str(tmp_path), max_context_tokens=None
    )
    off_cm = off._registry.get(ContextManager)
    assert isinstance(off_cm, SummarizingContextManager)
    assert off_cm._max_tokens is None


# --- review regressions -------------------------------------------------------------


def turn(i: int, n_tokens: int) -> Message:
    return Message(role="assistant" if i % 2 == 0 else "user", content=big(n_tokens))


def test_large_head_does_not_resummarize_every_turn():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=None, max_tokens=10_000)
    hist = [
        Message(role="system", content="sys"),
        Message(role="user", content=big(6000)),
    ]
    summarized_turns = 0
    for i in range(30):
        hist.append(turn(i, 500))
        before = model.calls
        out = process(cm, list(hist), ctx)
        summarized_turns += model.calls > before
        assert cm._estimate(out, ctx) <= 10_000
        assert_alternates(out)
    # Each fold leaves room for several turns of growth before the next one.
    assert summarized_turns <= 30 // 3


def test_head_over_budget_is_truncated_and_recorded():
    model = SummarizerModel()
    ctx = make_ctx(model)
    events = capture(ctx)
    cm = SummarizingContextManager(max_messages=None, max_tokens=1000)
    doc = "START" + "x" * 20_000 + "END"
    hist = [Message(role="system", content="sys"), Message(role="user", content=doc)]
    out = process(cm, list(hist), ctx)
    # The head alone is over budget: cut it rather than send it silently.
    assert [e.strategy for e in events] == ["truncate"]
    assert "head" in events[0].reason
    assert cm._estimate(out, ctx) <= 1000
    assert out[0].content == "sys"
    assert out[1].content.startswith("START") and out[1].content.endswith("END")
    assert cm.state(ctx).fallbacks == 1
    for i in range(16):
        hist.append(turn(i, 20))
        out = process(cm, list(hist), ctx)
        assert cm._estimate(out, ctx) <= 1000
        assert out[1].content.startswith("START")
    assert model.calls <= 4  # not one summarizer call per turn
    assert hist[1].content == doc  # the session history is not mutated


def test_unfixable_overflow_is_recorded_once():
    # Only a huge system prompt remains: nothing may be cut, so say so once.
    ctx = make_ctx(SummarizerModel())
    events = capture(ctx)
    cm = SummarizingContextManager(max_messages=None, max_tokens=1000)
    hist = [
        Message(role="system", content=big(5000)),
        Message(role="user", content="task"),
    ]
    for i in range(5):
        process(cm, list(hist), ctx)
        hist.append(turn(i, 10))
    assert [e.strategy for e in events] == ["overflow"]
    assert events[0].fallback
    assert cm.state(ctx).last_fallback.startswith("overflow")


def test_chained_managers_keep_separate_state():
    model = SummarizerModel()
    ctx = make_ctx(model)
    a = SummarizingContextManager(max_messages=10, keep_recent=4)
    b = SummarizingContextManager(max_messages=None, max_tokens=100_000)
    hist = history(2)
    for i in range(30):
        hist.append(Message(role="assistant" if i % 2 == 0 else "user", content="t"))
        out = list(hist)
        for cm in (a, b):
            out = process(cm, out, ctx)
    # Only ``a`` compacts, once per ~6 new messages, not once per turn.
    assert model.calls <= 5
    assert isinstance(a.state(ctx), SummaryState)
    assert a.state(ctx) is not b.state(ctx)
    assert a.state(ctx).summaries == model.calls
    assert b.state(ctx).summaries == 0


def test_truncate_shrinks_huge_tool_call_arguments():
    body = "B" * 40_000
    call = {
        "id": "c",
        "name": "write_file",
        "arguments": {"path": "a.py", "body": body},
    }
    hist = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content="ok", tool_use_id="c", name="write_file"),
    ]
    ctx = make_ctx()
    cm = SummarizingContextManager(max_messages=None, max_tokens=2000)
    out = process(cm, hist, ctx)
    assert cm._estimate(out, ctx) <= 2000
    shrunk = out[-2].tool_calls[0]
    assert shrunk["id"] == "c" and shrunk["arguments"]["path"] == "a.py"
    assert "characters truncated" in shrunk["arguments"]["body"]
    assert hist[2].tool_calls[0]["arguments"]["body"] == body  # not mutated
