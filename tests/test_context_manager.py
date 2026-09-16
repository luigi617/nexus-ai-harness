from __future__ import annotations

import asyncio

from core.message import Message
from core.response import Response
from plugins.context_manager import SummarizingContextManager
from tests.conftest import make_ctx


class SummarizerProvider:
    kind = "provider"

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
    prov = SummarizerProvider()
    ctx = make_ctx(prov)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    out = process(cm, history(3), ctx)
    assert len(out) == 5  # unchanged
    assert prov.calls == 0  # never summarized


def test_compaction_keeps_prefix_summary_and_tail():
    prov = SummarizerProvider()
    ctx = make_ctx(prov)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    out = process(cm, history(10), ctx)
    assert prov.calls == 1
    assert out[0].role == "system"
    assert out[1].content == "task"
    assert "RECAP" in out[2].content  # summary injected
    assert len(out) == 3 + 3  # prefix(2) + summary(1) + keep_recent(3)


def test_cached_summary_reused_without_new_call():
    prov = SummarizerProvider()
    ctx = make_ctx(prov)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    h = history(10)
    out1 = process(cm, list(h), ctx)
    h2 = [
        *h,
        Message(role="assistant", content="m10"),
        Message(role="user", content="m11"),
    ]
    out2 = process(cm, list(h2), ctx)
    assert prov.calls == 1  # no re-summarize under threshold
    # cache prefix (head + summary) is byte-stable across turns
    assert [(m.role, m.content) for m in out1[:3]] == [
        (m.role, m.content) for m in out2[:3]
    ]


def test_second_fold_accumulates_prior_summary():
    prov = SummarizerProvider()
    ctx = make_ctx(prov)
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    process(cm, history(10), ctx)  # fold 1 -> RECAP-1
    out = process(cm, history(20), ctx)  # tail past budget again -> fold 2
    assert prov.calls == 2
    assert "RECAP-2" in out[2].content  # summary advanced
    assert "RECAP-1" in prov.bodies[1]  # prior recap folded into the new one


def test_noop_when_no_provider():
    ctx = make_ctx()
    cm = SummarizingContextManager(max_messages=6, keep_recent=3)
    out = process(cm, history(10), ctx)
    assert len(out) == 12  # unchanged; summarization unavailable
