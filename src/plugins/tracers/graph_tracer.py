from __future__ import annotations

from core.events import Event, MessageAdded
from graph import Graph
from protocols.context import Context
from protocols.hook import Hook


class GraphTracer(Hook):
    """A hook that records each message node into a caller-owned graph."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._last: dict[str, str] = {}  # session id -> last node id
        # session id -> {tool call id -> owning node id}
        self._owner: dict[str, dict[str, str]] = {}

    def on(self, event: Event, ctx: Context) -> None:
        if not isinstance(event, MessageAdded):
            return
        node = event.message
        self._graph.add_node(node.id, data=node)

        sid = ctx.session_id
        last = self._last.get(sid)
        if last is not None:
            self._graph.add_edge(last, node.id, role="next")
        self._last[sid] = node.id

        owner = self._owner.setdefault(sid, {})
        for call in node.tool_calls:
            cid = call.get("id")
            if cid:
                owner[cid] = node.id

        if node.tool_use_id and node.tool_use_id in owner:
            self._graph.add_edge(
                owner[node.tool_use_id],
                node.id,
                role="tool_result",
                call_id=node.tool_use_id,
                name=node.name,
            )
