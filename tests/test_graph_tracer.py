from __future__ import annotations

from core.events import IterationStarted, MessageAdded
from core.message import Message
from graph import Graph
from plugins.tracers import GraphTracer
from tests.conftest import make_ctx


def feed(tracer, ctx, *messages):
    for m in messages:
        tracer.on(MessageAdded(m), ctx)


def test_ignores_non_message_events():
    g = Graph()
    GraphTracer(g).on(IterationStarted(0), make_ctx())
    assert len(g) == 0


def test_records_messages_and_chains_them_with_next_edges():
    g = Graph()
    ctx = make_ctx()
    a = Message(role="user", content="hi")
    b = Message(role="assistant", content="hello")
    feed(GraphTracer(g), ctx, a, b)
    assert len(g) == 2
    assert g.successors(a.id) == [b.id]  # sequential "next" edge
    edge = g.in_edges(b.id)[0]
    assert edge.metadata["role"] == "next"


def test_links_tool_result_to_its_originating_call():
    g = Graph()
    ctx = make_ctx()
    call = Message(
        role="assistant",
        content="",
        tool_calls=[{"id": "c1", "name": "calc", "arguments": {}}],
    )
    result = Message(role="tool", content="4", name="calc", tool_use_id="c1")
    feed(GraphTracer(g), ctx, call, result)

    roles = {e.metadata.get("role") for e in g.in_edges(result.id)}
    assert "tool_result" in roles  # extra edge from the call node to its result
    tool_edge = next(
        e for e in g.in_edges(result.id) if e.metadata.get("role") == "tool_result"
    )
    assert tool_edge.source == call.id
    assert tool_edge.metadata["call_id"] == "c1"


def test_does_not_bridge_the_chain_across_sessions():
    # A forked/subagent session inherits the same tracer instance; its chain
    # must stay independent instead of linking back to the parent's last node.
    g = Graph()
    tracer = GraphTracer(g)
    parent = make_ctx()
    child = make_ctx()  # a distinct Session -> distinct session_id
    assert parent.session_id != child.session_id

    last_parent = Message(role="assistant", content="parent-final")
    first_child = Message(role="user", content="child-first")
    tracer.on(MessageAdded(last_parent), parent)
    tracer.on(MessageAdded(first_child), child)

    assert g.successors(last_parent.id) == []  # no cross-session "next" edge
