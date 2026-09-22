from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from graph import Graph
from harness.introspection import (
    Capability,
    HarnessInspection,
    PluginInfo,
    inspect_registry,
)
from harness.registry import Registry
from protocols.plugin import Plugin

# A plugin lookup key: the instance, its PluginInfo, or its class name.
PluginRef = Plugin | PluginInfo | str


class DependencyCycleError(ValueError):
    """Raised when the plugin dependency graph cannot be ordered.

    A cycle means the plugins mutually require one another, so no start order
    exists. ``cycle`` names the plugins that participate in the cycle.

    Attributes:
        cycle: The names of the plugins caught in the dependency cycle.
    """

    def __init__(self, cycle: tuple[str, ...]) -> None:
        self.cycle = cycle
        # A set of members, not an ordered path, so joined by commas.
        joined = ", ".join(cycle) if cycle else "?"
        super().__init__(f"dependency cycle among plugins: {joined}")


@dataclass(frozen=True)
class GraphDiff:
    """The structural difference between two harness graphs.

    Names are class names, so replacing a plugin with a different class shows up
    as one removal and one addition — the shape of a plugin swap.

    Attributes:
        added_plugins: Plugins present in the other graph but not this one.
        removed_plugins: Plugins present in this graph but not the other.
        added_dependencies: ``(provider, dependent, via)`` edges only the other
            graph has.
        removed_dependencies: ``(provider, dependent, via)`` edges only this
            graph has.
    """

    added_plugins: tuple[str, ...]
    removed_plugins: tuple[str, ...]
    added_dependencies: tuple[tuple[str, str, str], ...]
    removed_dependencies: tuple[tuple[str, str, str], ...]

    def is_empty(self) -> bool:
        """Whether the two graphs are structurally identical."""
        return not (
            self.added_plugins
            or self.removed_plugins
            or self.added_dependencies
            or self.removed_dependencies
        )


class HarnessGraph:
    """The harness's plugins and capabilities as a first-class dependency graph.

    Nodes are the registered plugins; a directed edge ``provider -> dependent``
    means ``dependent`` declares a ``requires`` that ``provider`` satisfies (by
    being an instance of the required type). This is the same "another plugin
    satisfies it" rule the harness validates against, so the graph is a faithful
    model of what must be wired for the composition to run.

    From that model it answers the questions a composition system needs:
    startup ordering (:meth:`startup_order`), cycle detection
    (:meth:`has_cycle`), impact analysis (:meth:`impact_of`), dependency
    validation (:meth:`unmet_requirements`), and architecture comparison
    (:meth:`diff`). It is descriptive: building it never starts plugins.
    """

    def __init__(self, inspection: HarnessInspection) -> None:
        self._inspection = inspection
        self._graph = Graph()
        # Key by instance identity: two instances of a class share a qualified name.
        self._ids: dict[int, str] = {}
        self._infos: dict[str, PluginInfo] = {}
        for index, info in enumerate(inspection.plugins):
            node_id = f"{index}:{info.qualified_name}"
            self._ids[id(info.instance)] = node_id
            self._infos[node_id] = info
            self._graph.add_node(node_id, data=info)
        self._wire_dependencies()

    def _wire_dependencies(self) -> None:
        infos = self._inspection.plugins
        for info in infos:
            for dep in info.instance.requires:
                # A requirement is satisfied by another plugin, never itself.
                for other in infos:
                    if other.instance is not info.instance and isinstance(
                        other.instance, dep
                    ):
                        self._graph.add_edge(
                            self._ids[id(other.instance)],
                            self._ids[id(info.instance)],
                            role="requires",
                            via=dep.__name__,
                        )

    @property
    def name(self) -> str:
        """The name of the harness this graph describes."""
        return self._inspection.name

    def inspection(self) -> HarnessInspection:
        """The underlying composition snapshot backing this graph."""
        return self._inspection

    def plugins(self) -> tuple[PluginInfo, ...]:
        """Every registered plugin, in registration order."""
        return self._inspection.plugins

    def capabilities(self) -> tuple[Capability, ...]:
        """Every provided capability, each with its providers."""
        return self._inspection.capabilities

    def providers_of(self, capability: type[Plugin] | str) -> tuple[PluginInfo, ...]:
        """The plugins that provide ``capability`` (a protocol type or its name)."""
        return self._inspection.providers_of(capability)

    def dependencies_of(self, plugin: PluginRef) -> tuple[PluginInfo, ...]:
        """The plugins ``plugin`` directly depends on to satisfy its ``requires``."""
        node_id = self._resolve(plugin)
        return self._as_infos(self._graph.predecessors(node_id))

    def dependents_of(self, plugin: PluginRef) -> tuple[PluginInfo, ...]:
        """The plugins that directly depend on ``plugin``."""
        node_id = self._resolve(plugin)
        return self._as_infos(self._graph.successors(node_id))

    def impact_of(self, plugin: PluginRef) -> tuple[PluginInfo, ...]:
        """The plugins affected if ``plugin`` is replaced or removed.

        Returns every plugin that depends on ``plugin`` directly or transitively
        — the blast radius of changing it — excluding ``plugin`` itself, in a
        stable order.
        """
        start = self._resolve(plugin)
        affected: set[str] = set()
        stack = list(self._graph.successors(start))
        while stack:
            node_id = stack.pop()
            if node_id in affected:
                continue
            affected.add(node_id)
            stack.extend(self._graph.successors(node_id))
        affected.discard(start)  # a cycle can lead the walk back to the plugin
        return self._as_infos(self._ordered(affected))

    def startup_order(self) -> tuple[PluginInfo, ...]:
        """The plugins in a valid initialization order — providers before dependents.

        Raises:
            DependencyCycleError: If the plugins mutually depend on one another,
                so no order exists.
        """
        try:
            order = self._graph.topological_order()
        except ValueError as exc:
            raise DependencyCycleError(self._cycle_members()) from exc
        return self._as_infos(order)

    def has_cycle(self) -> bool:
        """Whether the plugins mutually depend on one another."""
        return bool(self._cycle_members())

    def unmet_requirements(self) -> tuple[tuple[str, str], ...]:
        """The ``(plugin, capability)`` pairs whose dependency no plugin satisfies."""
        infos = self._inspection.plugins
        unmet: list[tuple[str, str]] = []
        for info in infos:
            others = [o.instance for o in infos if o.instance is not info.instance]
            for dep in info.instance.requires:
                if not any(isinstance(o, dep) for o in others):
                    unmet.append((info.name, dep.__name__))
        return tuple(unmet)

    def is_valid(self) -> bool:
        """Whether every declared dependency is satisfied by another plugin."""
        return not self.unmet_requirements()

    def diff(self, other: HarnessGraph) -> GraphDiff:
        """Compare this graph's architecture against ``other``'s.

        Compares by plugin class name and by ``(provider, dependent, via)``
        dependency edges, so the result reads as what was added or removed going
        from this graph to ``other``.
        """
        mine, theirs = set(self._plugin_names()), set(other._plugin_names())
        my_edges, their_edges = set(self._edge_triples()), set(other._edge_triples())
        return GraphDiff(
            added_plugins=tuple(sorted(theirs - mine)),
            removed_plugins=tuple(sorted(mine - theirs)),
            added_dependencies=tuple(sorted(their_edges - my_edges)),
            removed_dependencies=tuple(sorted(my_edges - their_edges)),
        )

    def to_dict(self) -> dict:
        """A JSON-friendly snapshot of the graph — plugins, capabilities, edges."""
        return {
            "name": self.name,
            "plugins": [
                {
                    "name": info.name,
                    "qualified_name": info.qualified_name,
                    "status": info.status.name,
                    "provides": list(info.provides),
                    "requires": list(info.requires),
                }
                for info in self._inspection.plugins
            ],
            "capabilities": [
                {
                    "protocol": cap.protocol,
                    "providers": [p.name for p in cap.providers],
                    "selected": cap.selected.name if cap.selected else None,
                }
                for cap in self._inspection.capabilities
            ],
            "dependencies": [
                {"provider": provider, "dependent": dependent, "via": via}
                for provider, dependent, via in self._edge_triples()
            ],
        }

    def render(self) -> str:
        """Render the architecture as a dependency tree rooted at its entry points.

        Roots are the plugins nothing depends on (typically the loop). Each
        plugin's required capabilities are its children, and each capability's
        providers nest beneath it, recursing into their own requirements.
        """
        roots = [
            nid for nid in self._graph.node_ids() if not self._graph.successors(nid)
        ]
        if not roots and self._infos:  # every plugin is in a cycle
            roots = self._ordered(set(self._infos))
        lines = [self.name]
        for node_id in roots:
            lines.append(self._infos[node_id].name)
            self._render_requirements(node_id, "", set(), lines)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def _render_requirements(
        self, node_id: str, prefix: str, path: set[str], lines: list[str]
    ) -> None:
        """Append the plugin's required capabilities and providers, recursively."""
        if node_id in path:  # a cycle: stop before recursing into it again
            return
        path = path | {node_id}
        reqs = self._infos[node_id].requires
        for i, capability in enumerate(reqs):
            cap_glyph = "└──" if i == len(reqs) - 1 else "├──"
            cap_prefix = prefix + ("    " if i == len(reqs) - 1 else "│   ")
            # Read providers from wired edges: a requirement may name a concrete class.
            providers = [
                edge.source
                for edge in self._graph.in_edges(node_id)
                if edge.metadata.get("via") == capability
            ]
            # Collapse a concrete-class requirement: provider name == capability.
            if len(providers) == 1 and self._infos[providers[0]].name == capability:
                lines.append(f"{prefix}{cap_glyph} {capability}")
                self._render_requirements(providers[0], cap_prefix, path, lines)
                continue
            lines.append(f"{prefix}{cap_glyph} {capability}")
            for j, pid in enumerate(providers):
                prov_last = j == len(providers) - 1
                prov_glyph = "└──" if prov_last else "├──"
                lines.append(f"{cap_prefix}{prov_glyph} {self._infos[pid].name}")
                child_prefix = cap_prefix + ("    " if prov_last else "│   ")
                self._render_requirements(pid, child_prefix, path, lines)

    def _resolve(self, plugin: PluginRef) -> str:
        if isinstance(plugin, PluginInfo):
            plugin = plugin.instance
        if isinstance(plugin, str):
            for node_id, info in self._infos.items():
                if plugin in (info.qualified_name, info.name):
                    return node_id
            raise KeyError(f"no plugin named {plugin!r}")
        resolved = self._ids.get(id(plugin))
        if resolved is None:
            raise KeyError(f"plugin not registered: {type(plugin).__name__}")
        return resolved

    def _as_infos(self, node_ids: Iterable[str]) -> tuple[PluginInfo, ...]:
        return tuple(self._infos[node_id] for node_id in node_ids)

    def _ordered(self, node_ids: set[str]) -> list[str]:
        """The given nodes in startup order, falling back to node order on a cycle."""
        try:
            order = self._graph.topological_order()
        except ValueError:
            order = self._graph.node_ids()
        return [node_id for node_id in order if node_id in node_ids]

    def _cycle_members(self) -> tuple[str, ...]:
        """The names of plugins on a dependency cycle, empty when acyclic.

        A plugin is on a cycle only if it belongs to a strongly-connected
        component of more than one node (or has a self-loop): every node in such
        a component can reach every other and return. Bridge nodes that merely
        lie on a path connecting two separate cycles — with a live predecessor
        and successor but no way back to themselves — are correctly excluded, as
        are plugins merely up- or downstream of a cycle.
        """
        cyclic = {nid for component in self._strongly_connected() for nid in component}
        return tuple(
            self._infos[nid].name for nid in self._graph.node_ids() if nid in cyclic
        )

    def _strongly_connected(self) -> list[set[str]]:
        """Strongly-connected components that form a cycle (size >= 2 or self-loop).

        Uses an iterative Tarjan's algorithm so a deep dependency graph cannot
        overflow the recursion stack.
        """
        index_of: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        counter = 0
        components: list[set[str]] = []

        for root in self._graph.node_ids():
            if root in index_of:
                continue
            # work stack of (node, iterator over its successors)
            work: list[tuple[str, list[str]]] = []
            index_of[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            work.append((root, list(self._graph.successors(root))))
            while work:
                node_id, succ = work[-1]
                if succ:
                    nxt = succ.pop()
                    if nxt not in index_of:
                        index_of[nxt] = low[nxt] = counter
                        counter += 1
                        stack.append(nxt)
                        on_stack.add(nxt)
                        work.append((nxt, list(self._graph.successors(nxt))))
                    elif nxt in on_stack:
                        low[node_id] = min(low[node_id], index_of[nxt])
                else:
                    work.pop()
                    if low[node_id] == index_of[node_id]:
                        component: set[str] = set()
                        while True:
                            member = stack.pop()
                            on_stack.discard(member)
                            component.add(member)
                            if member == node_id:
                                break
                        is_cycle = len(component) > 1 or (
                            node_id in self._graph.successors(node_id)
                        )
                        if is_cycle:
                            components.append(component)
                    if work:
                        parent = work[-1][0]
                        low[parent] = min(low[parent], low[node_id])
        return components

    def _plugin_names(self) -> list[str]:
        return [info.name for info in self._inspection.plugins]

    def _edge_triples(self) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        for edge in self._graph.edges():
            provider = self._infos[edge.source].name
            dependent = self._infos[edge.target].name
            triples.append((provider, dependent, edge.metadata.get("via", "")))
        return triples


def build_graph(registry: Registry, *, name: str = "NexusAIHarness") -> HarnessGraph:
    """Build a :class:`HarnessGraph` describing ``registry``'s dependency structure.

    Args:
        registry: The registry whose plugins and dependencies to model.
        name: The name to show as the root of the rendered tree.

    Returns:
        A dependency graph over the registered plugins and their capabilities.
    """
    return HarnessGraph(inspect_registry(registry, name=name))
