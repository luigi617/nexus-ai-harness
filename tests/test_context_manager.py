from __future__ import annotations

import asyncio

from core.message import Message
from core.response import Response
from plugins.context_manager import SummarizingContextManager
from protocols.model import Model
from tests.conftest import make_ctx


class SummarizerModel(Model):
    def __init__(self) -> None:
        self.calls = 0
        self.bodies: list[str] = []

    async def complete(self, history, ctx) -> Response:
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
    assert out[1].content == "task"
    assert "RECAP" in out[2].content  # summary injected
    assert len(out) == 3 + 3  # prefix(2) + summary(1) + keep_recent(3)


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
    assert [(m.role, m.content) for m in out1[:3]] == [
        (m.role, m.content) for m in out2[:3]
    ]


def test_second_fold_accumulates_prior_summary():
    model = SummarizerModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    process(cm, history(10), ctx)  # fold 1 -> RECAP-1
    out = process(cm, history(20), ctx)  # tail past budget again -> fold 2
    assert model.calls == 2
    assert "RECAP-2" in out[2].content  # summary advanced
    assert "RECAP-1" in model.bodies[1]  # prior recap folded into the new one


def test_noop_when_no_provider():
    ctx = make_ctx()
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    out = process(cm, history(10), ctx)
    assert len(out) == 12  # unchanged; summarization unavailable


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

    async def complete(self, history, ctx) -> Response:
        self.calls += 1
        raise RuntimeError("backend exploded")


def test_summarize_failure_leaves_history_and_state_untouched():
    model = RaisingModel()
    ctx = make_ctx(model)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    hist = history(10)
    out = process(cm, hist, ctx)

    assert model.calls == 1  # it did try
    # No summary injected; history returned unchanged.
    assert [(m.role, m.content) for m in out] == [(m.role, m.content) for m in hist]
    assert not any("summary" in (m.content or "").lower() for m in out)
    # No partial corruption of the cached summary state.
    from plugins.context_manager.summarizing import SummaryState

    state = ctx.state(SummaryState)
    assert state.upto == 0
    assert state.text == ""


class FlakySummarizer(Model):
    """Raises on its first summarize, then succeeds — to prove recovery."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, history, ctx) -> Response:
        self.calls += 1
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

    out = process(cm, history(10), ctx)
    assert model.calls == 2  # retried on the recovered provider
    assert "RECAP" in out[2].content  # and compacted this time


def test_constructor_rejects_keep_recent_ge_max_messages():
    import pytest

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
