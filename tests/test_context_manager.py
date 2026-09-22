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
    # new_cutoff = len - keep_recent = 5 lands on the tool result at index 5,
    # whose parent tool_use (index 4) is folded into the summary; the kept tail
    # must not begin with that orphaned tool result.
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
