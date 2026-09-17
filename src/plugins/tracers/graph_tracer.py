from __future__ import annotations

from core.events import Event, MessageAdded
from graph import Graph
from protocols.context import Context
from protocols.hook import Hook


class GraphTracer(Hook):
    """A hook that records each message node into a caller-owned graph."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._last: str | None = None
        self._owner: dict[str, str] = {}  # tool call id -> assistant node id

    def on(self, event: Event, ctx: Context) -> None:
        if not isinstance(event, MessageAdded):
            return
        node = event.message
        self._graph.add_node(node.id, data=node)

        if self._last is not None:
            self._graph.add_edge(self._last, node.id, role="next")
        self._last = node.id

        for call in node.tool_calls:
            cid = call.get("id")
            if cid:
                self._owner[cid] = node.id

        if node.tool_use_id and node.tool_use_id in self._owner:
            self._graph.add_edge(
                self._owner[node.tool_use_id],
                node.id,
                role="tool_result",
                call_id=node.tool_use_id,
                name=node.name,
            )
