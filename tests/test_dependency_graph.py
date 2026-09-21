from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest

from core.response import Response
from harness import (
    DependencyCycleError,
    GraphDiff,
    HarnessGraph,
    NexusAIHarness,
    PluginStatus,
)
from harness.graph import build_graph
from harness.registry import Registry
from plugins.guards import BudgetGuard
from plugins.hooks import CostCounter
from plugins.loops import AgenticLoop
from protocols.model import Model
from protocols.plugin import Plugin
from protocols.tool import Tool
from tests.conftest import RecordingTool, ScriptedModel


def _registry(*plugins: object) -> Registry:
    reg = Registry()
    for plugin in plugins:
        reg.add(plugin)
    return reg


class _NamedTool(Tool):
    """A minimal tool whose class stands in for a capability in dependency chains."""

    def __init__(self, name: str = "t") -> None:
        self.name = name
        self.description = ""
        self.parameters = {}

    def run(self, arguments, ctx):  # pragma: no cover - never executed
        return ""


class Leaf(_NamedTool):
    pass


class Mid(_NamedTool):
    requires: ClassVar[tuple[type[Plugin], ...]] = (Leaf,)


class Top(_NamedTool):
    requires: ClassVar[tuple[type[Plugin], ...]] = (Mid,)


def _chain() -> HarnessGraph:
    """A Leaf <- Mid <- Top dependency chain."""
    return build_graph(_registry(Top(), Mid(), Leaf()))


# --- construction / delegation -------------------------------------------------


def test_harness_graph_is_named_after_the_harness_and_starts_nothing():
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="x")))
    graph = h.graph()
    assert isinstance(graph, HarnessGraph)
    assert graph.name == "NexusAIHarness"
    assert all(p.status is PluginStatus.REGISTERED for p in graph.plugins())


def test_plugins_and_capabilities_delegate_to_the_inspection():
    graph = build_graph(_registry(AgenticLoop(), ScriptedModel(Response(text="x"))))
    assert [p.name for p in graph.plugins()] == ["AgenticLoop", "ScriptedModel"]
    assert [c.protocol for c in graph.capabilities()] == ["Loop", "Model"]
    assert [p.name for p in graph.providers_of(Model)] == ["ScriptedModel"]


def test_inspection_returns_the_backing_snapshot():
    from harness.introspection import HarnessInspection, inspect_registry

    inspection = inspect_registry(_registry(AgenticLoop()))
    assert HarnessGraph(inspection).inspection() is inspection
    assert isinstance(
        build_graph(_registry(AgenticLoop())).inspection(), HarnessInspection
    )


# --- dependency edges ----------------------------------------------------------


def test_direct_dependencies_and_dependents():
    graph = _chain()
    assert [p.name for p in graph.dependencies_of("Top")] == ["Mid"]
    assert [p.name for p in graph.dependents_of("Leaf")] == ["Mid"]
    # Endpoints have no dependency on / dependent below them.
    assert graph.dependencies_of("Leaf") == ()
    assert graph.dependents_of("Top") == ()


def test_edges_use_the_another_plugin_rule_from_validation():
    # AgenticLoop requires Model; ScriptedModel provides it.
    graph = build_graph(_registry(AgenticLoop(), ScriptedModel(Response(text="x"))))
    assert [p.name for p in graph.dependencies_of("AgenticLoop")] == ["ScriptedModel"]


def test_dependency_on_a_concrete_plugin_class_is_wired():
    # BudgetGuard requires the concrete CostCounter, not a protocol.
    graph = build_graph(_registry(BudgetGuard(5.0), CostCounter()))
    assert [p.name for p in graph.dependents_of("CostCounter")] == ["BudgetGuard"]
    assert [p.name for p in graph.dependencies_of("BudgetGuard")] == ["CostCounter"]


# --- impact analysis -----------------------------------------------------------


def test_impact_is_the_transitive_set_of_dependents():
    graph = _chain()
    # Replacing Leaf ripples up to Mid and Top; Top affects nothing.
    assert [p.name for p in graph.impact_of("Leaf")] == ["Mid", "Top"]
    assert [p.name for p in graph.impact_of("Mid")] == ["Top"]
    assert graph.impact_of("Top") == ()


def test_unrelated_plugin_has_no_impact_on_the_chain():
    graph = build_graph(_registry(Top(), Mid(), Leaf(), RecordingTool("free")))
    assert graph.impact_of("RecordingTool") == ()


# --- startup ordering ----------------------------------------------------------


def test_startup_order_places_providers_before_dependents():
    graph = _chain()
    order = [p.name for p in graph.startup_order()]
    assert order.index("Leaf") < order.index("Mid") < order.index("Top")


def test_startup_order_includes_independent_plugins():
    graph = build_graph(_registry(Top(), Mid(), Leaf(), RecordingTool("free")))
    names = {p.name for p in graph.startup_order()}
    assert names == {"Leaf", "Mid", "Top", "RecordingTool"}


# --- cycle detection -----------------------------------------------------------


class Ping(_NamedTool):
    pass


class Pong(_NamedTool):
    pass


class Downstream(_NamedTool):
    """Depends on a cycle member without being part of the cycle."""

    requires: ClassVar[tuple[type[Plugin], ...]] = (Ping,)


Ping.requires = (Pong,)
Pong.requires = (Ping,)


def test_has_cycle_is_false_for_an_acyclic_graph():
    assert _chain().has_cycle() is False


def test_cycle_is_detected_and_named():
    graph = build_graph(_registry(Ping(), Pong()))
    assert graph.has_cycle() is True
    with pytest.raises(DependencyCycleError) as exc:
        graph.startup_order()
    assert set(exc.value.cycle) == {"Ping", "Pong"}


def test_cycle_members_exclude_plugins_merely_downstream():
    # Downstream depends on Ping but is not on the Ping<->Pong cycle.
    graph = build_graph(_registry(Ping(), Pong(), Downstream()))
    with pytest.raises(DependencyCycleError) as exc:
        graph.startup_order()
    assert set(exc.value.cycle) == {"Ping", "Pong"}


def test_impact_excludes_the_plugin_itself_on_a_cycle():
    graph = build_graph(_registry(Ping(), Pong()))
    # A cycle leads the walk back to the start; it must not be its own dependent.
    impacted = {p.name for p in graph.impact_of("Ping")}
    assert impacted == {"Pong"}


# --- validation ----------------------------------------------------------------


def test_unmet_requirements_reports_missing_providers():
    graph = build_graph(_registry(AgenticLoop()))  # no Model registered
    assert graph.unmet_requirements() == (("AgenticLoop", "Model"),)
    assert graph.is_valid() is False


def test_unmet_requirements_accumulates_multiple_gaps_for_one_plugin():
    class NeedsTwo(_NamedTool):
        requires: ClassVar[tuple[type[Plugin], ...]] = (Model, Leaf)

    graph = build_graph(_registry(NeedsTwo()))  # neither Model nor Leaf present
    assert set(graph.unmet_requirements()) == {
        ("NeedsTwo", "Model"),
        ("NeedsTwo", "Leaf"),
    }


def test_a_satisfied_composition_is_valid():
    graph = build_graph(_registry(AgenticLoop(), ScriptedModel(Response(text="x"))))
    assert graph.unmet_requirements() == ()
    assert graph.is_valid() is True


# --- architecture comparison ---------------------------------------------------


def test_diff_reports_swapped_plugins_and_edges():
    before = build_graph(_registry(AgenticLoop(), ScriptedModel(Response(text="x"))))
    after = build_graph(_registry(AgenticLoop(), _NamedModel()))
    diff = before.diff(after)
    assert isinstance(diff, GraphDiff)
    assert diff.added_plugins == ("_NamedModel",)
    assert diff.removed_plugins == ("ScriptedModel",)
    # The Model dependency now flows from a different provider.
    assert ("_NamedModel", "AgenticLoop", "Model") in diff.added_dependencies
    assert ("ScriptedModel", "AgenticLoop", "Model") in diff.removed_dependencies


def test_diff_of_identical_graphs_is_empty():
    model = ScriptedModel(Response(text="x"))
    before = build_graph(_registry(AgenticLoop(), model))
    after = build_graph(_registry(AgenticLoop(), model))
    assert before.diff(after).is_empty() is True


def test_diff_is_not_empty_when_graphs_differ():
    before = build_graph(_registry(AgenticLoop(), ScriptedModel(Response(text="x"))))
    after = build_graph(_registry(AgenticLoop(), _NamedModel()))
    assert before.diff(after).is_empty() is False


# --- serialization -------------------------------------------------------------


def test_to_dict_captures_plugins_capabilities_and_edges():
    graph = build_graph(_registry(AgenticLoop(), ScriptedModel(Response(text="x"))))
    data = graph.to_dict()
    assert data["name"] == "NexusAIHarness"
    assert {p["name"] for p in data["plugins"]} == {"AgenticLoop", "ScriptedModel"}
    edge = {"provider": "ScriptedModel", "dependent": "AgenticLoop", "via": "Model"}
    assert edge in data["dependencies"]
    model_cap = next(c for c in data["capabilities"] if c["protocol"] == "Model")
    assert model_cap["selected"] == "ScriptedModel"


def test_to_dict_reports_no_selection_for_a_multi_select_capability():
    # Tools are multi-select, so no single provider is resolved.
    data = build_graph(_registry(RecordingTool("echo"))).to_dict()
    tool_cap = next(c for c in data["capabilities"] if c["protocol"] == "Tool")
    assert tool_cap["selected"] is None


# --- rendering -----------------------------------------------------------------


def test_render_draws_a_tree_of_capabilities_and_providers():
    graph = build_graph(
        _registry(AgenticLoop(), ScriptedModel(Response(text="x"))), name="MyHarness"
    )
    rendered = graph.render()
    assert rendered == str(graph)
    lines = rendered.splitlines()
    assert lines[0] == "MyHarness"
    assert "AgenticLoop" in rendered
    # AgenticLoop requires Model, provided by ScriptedModel.
    assert "└── Model" in rendered
    assert "    └── ScriptedModel" in rendered


def test_render_recurses_into_a_providers_own_requirements():
    graph = _chain()
    rendered = graph.render()
    # Top -> Mid -> Leaf nests, capability names come from `requires`.
    assert "Top" in rendered
    assert "├── Mid" in rendered or "└── Mid" in rendered
    assert "Leaf" in rendered
    assert rendered.index("Mid") < rendered.index("Leaf")


def test_render_collapses_concrete_class_requirements():
    # BudgetGuard requires the concrete CostCounter, so the capability name and
    # provider name coincide and render as a single node, not "CostCounter" twice.
    graph = build_graph(_registry(BudgetGuard(5.0), CostCounter()))
    rendered = graph.render()
    assert rendered.count("CostCounter") == 1


def test_render_survives_a_cycle():
    graph = build_graph(_registry(Ping(), Pong()))
    rendered = graph.render()  # must not recurse forever
    assert "Ping" in rendered
    assert "Pong" in rendered


# --- lookups -------------------------------------------------------------------


def test_lookups_accept_instance_info_and_name():
    leaf = Leaf()
    graph = build_graph(_registry(Top(), Mid(), leaf))
    info = next(p for p in graph.plugins() if p.name == "Leaf")
    assert graph.dependents_of(leaf) == graph.dependents_of("Leaf")
    assert graph.dependents_of(info) == graph.dependents_of("Leaf")
    # The module-qualified name resolves the same node as the short name.
    assert graph.dependents_of(info.qualified_name) == graph.dependents_of("Leaf")


def test_unknown_plugin_lookup_raises():
    graph = _chain()
    with pytest.raises(KeyError):
        graph.dependents_of("Nope")


def test_repeated_plugin_class_yields_distinct_nodes():
    a = RecordingTool("a")
    b = RecordingTool("b")
    graph = build_graph(_registry(a, b))
    # Same class, same qualified name, but two separate nodes.
    assert len(graph.startup_order()) == 2
    assert graph.dependents_of(a) == ()
    assert graph.dependents_of(b) == ()


def test_graph_never_starts_plugins_even_after_start():
    async def run() -> HarnessGraph:
        h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="x")))
        await h.start()
        try:
            return h.graph()
        finally:
            await h.stop()

    graph = asyncio.run(run())
    assert all(p.status is PluginStatus.REGISTERED for p in graph.plugins())


class _NamedModel(Model):
    """A distinct Model implementation, to exercise plugin-swap diffs."""

    async def complete(self, history, ctx) -> Response:  # pragma: no cover
        return Response(text="named")
