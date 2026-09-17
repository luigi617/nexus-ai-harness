from __future__ import annotations

from collections.abc import Iterable, Mapping

from graph.edge import Edge
from graph.node import Node


class Graph:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._succ: dict[str, list[str]] = {}
        self._pred: dict[str, list[str]] = {}
        self._edges: list[Edge] = []

    def add_node(self, node: Node | str, data=None) -> Node:
        n = node if isinstance(node, Node) else Node(id=node, data=data)
        existing = self._nodes.get(n.id)
        if existing is None:
            self._nodes[n.id] = n
            self._succ[n.id] = []
            self._pred[n.id] = []
            return n
        if existing.data is None and n.data is not None:
            existing.data = n.data
        return existing

    def add_edge(self, source: str, target: str, **metadata) -> Edge:
        """Connect two nodes that already exist.

        Both endpoints must be present: auto-creating them would produce a node
        with no payload, which is exactly the node a caller cannot run.
        """
        self._require(source)
        self._require(target)
        edge = Edge(source=source, target=target, metadata=dict(metadata))
        self._edges.append(edge)
        if target not in self._succ[source]:
            self._succ[source].append(target)
        if source not in self._pred[target]:
            self._pred[target].append(source)
        return edge

    def attach(
        self,
        node_id: str,
        *,
        data=None,
        needs: str | Iterable[str] | None = None,
        kwargs: Mapping[str, str] | None = None,
    ) -> str:
        """Add a node wired to already-present parents, and return its final id."""
        if node_id in self._nodes:
            node_id = f"{node_id}#{len(self._nodes)}"
        self.add_node(node_id, data=data)
        for dep in (needs,) if isinstance(needs, str) else tuple(needs or ()):
            if dep not in self._nodes:
                raise KeyError(f"node {node_id!r} needs unknown node {dep!r}")
            self.add_edge(dep, node_id)
        for name, dep in (kwargs or {}).items():
            if dep not in self._nodes:
                raise KeyError(
                    f"node {node_id!r} kwarg {name!r} needs unknown node {dep!r}"
                )
            self.add_edge(dep, node_id, role="kwarg", name=name)
        return node_id

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def node(self, node_id: str) -> Node:
        self._require(node_id)
        return self._nodes[node_id]

    def node_ids(self) -> list[str]:
        return list(self._nodes)

    def edges(self) -> list[Edge]:
        return list(self._edges)

    def in_edges(self, node_id: str) -> list[Edge]:
        self._require(node_id)
        return [e for e in self._edges if e.target == node_id]

    def successors(self, node_id: str) -> list[str]:
        self._require(node_id)
        return list(self._succ[node_id])

    def predecessors(self, node_id: str) -> list[str]:
        self._require(node_id)
        return list(self._pred[node_id])

    def leaves(self) -> list[str]:
        return [nid for nid in self._nodes if not self._succ[nid]]

    def levels(self) -> list[list[str]]:
        """Group the nodes into dependency waves, each ready to run in parallel.

        Wave 0 is the nodes with no predecessors; wave N+1 is what becomes ready
        once wave N is done. Ids are sorted within a wave so a run is
        reproducible. Raises ValueError if the graph contains a cycle — this is
        the only cycle check in the codebase, so `levels()` (or
        `topological_order()`, which flattens it) is how you validate a graph.
        """
        indeg = {nid: len(self._pred[nid]) for nid in self._nodes}
        wave = sorted(nid for nid, count in indeg.items() if count == 0)
        waves: list[list[str]] = []
        seen = 0
        while wave:
            waves.append(wave)
            seen += len(wave)
            nxt: list[str] = []
            for u in wave:
                for v in self._succ[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        nxt.append(v)
            wave = sorted(nxt)
        if seen != len(self._nodes):
            raise ValueError("graph contains a cycle")
        return waves

    def topological_order(self) -> list[str]:
        return [node_id for wave in self.levels() for node_id in wave]

    def to_text(self, label=None, width: int = 44) -> str:
        """Render the graph as a vertical stack of rectangular boxes."""
        render = label or (lambda data: str(data))

        def box(text: str) -> list[str]:
            text = text.replace("\n", " ")
            if len(text) > width:
                text = text[: width - 3] + "..."
            border = "─" * (width + 2)
            return [f"┌{border}┐", f"│ {text.ljust(width)} │", f"└{border}┘"]

        order = self.topological_order()
        connector = " " * ((width + 4) // 2) + "│"
        lines: list[str] = []
        for i, node_id in enumerate(order):
            lines += box(render(self._nodes[node_id].data))
            if i < len(order) - 1:
                lines.append(connector)
        return "\n".join(lines)

    def _require(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"unknown node id: {node_id!r}")
