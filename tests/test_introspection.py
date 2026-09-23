from __future__ import annotations

from typing import ClassVar

from core.response import Response
from harness import (
    Capability,
    HarnessInspection,
    NexusAIHarness,
    PluginInfo,
    PluginStatus,
)
from harness.introspection import inspect_registry
from harness.registry import Registry
from plugins.loops import AgenticLoop, ChatLoop
from protocols.context_manager import ContextManager
from protocols.interceptor import Interceptor
from protocols.lifecycle import Lifecycle
from protocols.loop import Loop
from protocols.memory import MemoryStore
from protocols.model import Model
from protocols.plugin import Plugin
from protocols.router import Router
from protocols.tool import Tool
from tests.conftest import RecordingTool, ScriptedModel


def _registry(*plugins: object) -> Registry:
    reg = Registry()
    for plugin in plugins:
        reg.add(plugin)
    return reg


class SecondModel(ScriptedModel):
    """A distinct Model subclass, to exercise single-select shadowing."""


class NeedsModel(Loop):
    requires: ClassVar[tuple[type[Plugin], ...]] = (Model,)

    def run(self, ctx):  # pragma: no cover - never executed
        return ""


def test_inspect_returns_structured_snapshot_named_after_the_harness():
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="x")))
    snapshot = h.graph().inspection()
    assert isinstance(snapshot, HarnessInspection)
    assert snapshot.name == "NexusAIHarness"


def test_inspect_never_starts_plugins():
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="x")))
    snapshot = h.graph().inspection()
    assert all(p.status is PluginStatus.REGISTERED for p in snapshot.plugins)


def test_plugins_listed_in_registration_order_with_facts():
    loop = AgenticLoop()
    model = ScriptedModel(Response(text="x"))
    snapshot = inspect_registry(_registry(loop, model))
    assert [p.name for p in snapshot.plugins] == ["AgenticLoop", "ScriptedModel"]

    loop_info = snapshot.plugin("AgenticLoop")
    assert loop_info is not None
    assert isinstance(loop_info, PluginInfo)
    assert loop_info.instance is loop
    assert loop_info.provides == ("Loop",)
    # AgenticLoop declares requires=(Model,).
    assert loop_info.requires == ("Model",)
    assert loop_info.qualified_name == "plugins.loops.agentic.AgenticLoop"
    assert loop_info.target is None  # not an interceptor


def test_capabilities_group_providers_and_omit_the_unprovided():
    snapshot = inspect_registry(
        _registry(AgenticLoop(), ScriptedModel(Response(text="x")))
    )
    protocols = [c.protocol for c in snapshot.capabilities]
    assert protocols == ["Loop", "Model"]
    assert all(isinstance(c, Capability) for c in snapshot.capabilities)
    # A capability nobody provides is absent, not an empty group.
    assert snapshot.capability(Tool) is None


def test_single_select_capability_resolves_the_last_registered_provider():
    first = ScriptedModel(Response(text="first"))
    second = SecondModel(Response(text="second"))
    snapshot = inspect_registry(_registry(first, second))

    model_cap = snapshot.capability(Model)
    assert model_cap is not None
    assert model_cap.selects_one is True
    assert [p.instance for p in model_cap.providers] == [first, second]
    # ctx.get returns the last matching registration, so the last one wins.
    assert model_cap.selected is not None
    assert model_cap.selected.instance is second
    assert snapshot.provider_of(Model) is model_cap.selected


def test_multi_select_capability_has_no_single_selection():
    a = RecordingTool("a")
    b = RecordingTool("b")
    snapshot = inspect_registry(_registry(a, b))

    tools = snapshot.capability(Tool)
    assert tools is not None
    assert tools.protocol == "Tool"
    assert tools.label == "Tools"  # display label differs from the protocol name
    assert tools.selects_one is False
    assert tools.selected is None
    assert snapshot.provider_of(Tool) is None
    # Multi-select providers keep registration order.
    assert [p.instance for p in tools.providers] == [a, b]
    assert snapshot.providers_of(Tool) == tools.providers


def test_context_manager_is_a_multi_provider_capability():
    from plugins.context_manager import SummarizingContextManager

    snapshot = inspect_registry(_registry(SummarizingContextManager()))
    cm = snapshot.capability(ContextManager)
    assert cm is not None
    assert cm.selects_one is False
    assert cm.selected is None


def test_dependents_of_finds_plugins_requiring_a_capability():
    snapshot = inspect_registry(
        _registry(AgenticLoop(), ScriptedModel(Response(text="x")))
    )
    dependents = snapshot.dependents_of(Model)
    assert [p.name for p in dependents] == ["AgenticLoop"]
    # Accepts the protocol name as a string too.
    assert snapshot.dependents_of("Model") == dependents
    assert snapshot.dependents_of(Tool) == ()


def test_capability_and_provider_lookups_accept_type_or_name():
    snapshot = inspect_registry(
        _registry(AgenticLoop(), ScriptedModel(Response(text="x")))
    )
    assert snapshot.capability("Loop") is snapshot.capability(Loop)
    assert snapshot.provider_of("Model") is snapshot.provider_of(Model)


def test_lookups_are_empty_for_unknown_capabilities():
    snapshot = inspect_registry(_registry(AgenticLoop()))
    assert snapshot.capability(MemoryStore) is None
    assert snapshot.provider_of(MemoryStore) is None
    assert snapshot.providers_of(MemoryStore) == ()
    assert snapshot.plugin("Nope") is None


class _StartablePlugin(Plugin, Lifecycle):
    """A Lifecycle plugin that reaches STARTED once the harness starts."""


def test_status_reflects_started_plugins():
    startable = _StartablePlugin()

    async def run() -> HarnessInspection:
        h = (
            NexusAIHarness()
            .use(AgenticLoop())
            .use(ScriptedModel(Response(text="x")))
            .use(startable)
        )
        await h.start()
        try:
            return h.graph().inspection()
        finally:
            await h.stop()

    import asyncio

    snapshot = asyncio.run(run())
    status = {p.instance: p.status for p in snapshot.plugins}
    # The Lifecycle plugin was started; non-Lifecycle plugins stay REGISTERED.
    assert status[startable] is PluginStatus.STARTED
    assert all(
        s is PluginStatus.REGISTERED
        for inst, s in status.items()
        if inst is not startable
    )


def test_render_draws_a_tree_rooted_at_the_harness():
    snapshot = inspect_registry(
        _registry(AgenticLoop(), ScriptedModel(Response(text="x"))), name="MyHarness"
    )
    rendered = snapshot.render()
    assert rendered == str(snapshot)
    lines = rendered.splitlines()
    assert lines[0] == "MyHarness"
    assert "├── Loop" in rendered
    assert "└── Model" in rendered  # last capability uses the closing glyph
    assert "    └── ScriptedModel" in rendered
    # A non-last capability's providers carry the continuation prefix.
    assert "│   └── AgenticLoop" in rendered


def test_render_marks_the_selected_provider_when_providers_compete():
    first = ScriptedModel(Response(text="first"))
    second = SecondModel(Response(text="second"))
    rendered = inspect_registry(_registry(first, second)).render()
    # The shadowed loser carries no marker; the resolved winner is named.
    assert "├── ScriptedModel\n" in rendered + "\n"
    assert "└── SecondModel  (selected)" in rendered


def test_render_omits_the_marker_for_a_sole_provider():
    rendered = inspect_registry(
        _registry(AgenticLoop(), ScriptedModel(Response(text="x")))
    ).render()
    assert "(selected)" not in rendered


def test_multiple_providers_of_a_multi_capability_all_render():
    rendered = inspect_registry(
        _registry(RecordingTool("a"), RecordingTool("b"))
    ).render()
    # The display label differs from the protocol name (Tool -> "Tools").
    assert "└── Tools" in rendered
    assert "├── RecordingTool" in rendered
    assert "└── RecordingTool" in rendered
    assert "(selected)" not in rendered  # nothing is shadowed


def test_empty_harness_renders_just_its_name():
    snapshot = inspect_registry(Registry())
    assert snapshot.plugins == ()
    assert snapshot.capabilities == ()
    assert snapshot.render() == "NexusAIHarness"


def test_default_harness_inspection_covers_its_composition(tmp_path):
    from plugins import default_harness

    h = default_harness(ScriptedModel(Response(text="x")), memory_dir=str(tmp_path))
    snapshot = h.graph().inspection()

    # The loop, model, and memory are single-select; tools are multi.
    assert snapshot.provider_of(Loop).name == "AgenticLoop"
    assert snapshot.provider_of(Model).name == "ScriptedModel"
    assert snapshot.capability(Tool).selects_one is False
    tool_names = {p.name for p in snapshot.providers_of(Tool)}
    assert {"Remember", "Recall", "Subagent"} <= tool_names
    # Memory tools depend on the store, so they surface as its dependents.
    dependent_names = {p.name for p in snapshot.dependents_of(MemoryStore)}
    assert {"Remember", "Recall"} <= dependent_names


def test_chat_loop_also_provides_the_loop_capability():
    snapshot = inspect_registry(_registry(ChatLoop()))
    assert snapshot.provider_of(Loop).name == "ChatLoop"


def test_requires_reports_concrete_plugin_dependencies_by_name():
    from plugins.guards import BudgetGuard
    from plugins.hooks import CostCounter

    snapshot = inspect_registry(_registry(BudgetGuard(5.0), CostCounter()))
    guard = snapshot.plugin("BudgetGuard")
    assert guard is not None
    # BudgetGuard requires the concrete CostCounter, reported by its name.
    assert guard.requires == ("CostCounter",)
    assert [p.name for p in snapshot.dependents_of("CostCounter")] == ["BudgetGuard"]


def test_needs_model_loop_is_reported_as_a_model_dependent():
    snapshot = inspect_registry(
        _registry(NeedsModel(), ScriptedModel(Response(text="x")))
    )
    assert [p.name for p in snapshot.dependents_of(Model)] == ["NeedsModel"]


class PinnedRouter(Router):
    """A Router that always returns the last registered model."""

    def route(self, history, ctx):  # pragma: no cover - never executed here
        return ctx.get(Model)


def test_model_has_no_static_selection_when_a_router_is_present():
    first = ScriptedModel(Response(text="first"))
    second = SecondModel(Response(text="second"))
    snapshot = inspect_registry(_registry(PinnedRouter(), first, second))

    model_cap = snapshot.capability(Model)
    assert model_cap is not None
    # One model runs per turn, but the router chooses it dynamically.
    assert model_cap.selects_one is True
    assert [p.instance for p in model_cap.providers] == [first, second]
    assert model_cap.selected is None
    assert snapshot.provider_of(Model) is None
    # Other single-select capabilities are unaffected by the router.
    assert snapshot.provider_of(Router).name == "PinnedRouter"


def test_model_is_statically_selected_without_a_router():
    first = ScriptedModel(Response(text="first"))
    second = SecondModel(Response(text="second"))
    snapshot = inspect_registry(_registry(first, second))
    assert snapshot.provider_of(Model).instance is second


class DecoyTarget(Tool):
    """A non-interceptor that coincidentally has a `target` class attribute."""

    target = Model

    def run(self, arguments, ctx):  # pragma: no cover
        return ""


def test_target_is_none_for_non_interceptors():
    snapshot = inspect_registry(_registry(DecoyTarget()))
    info = snapshot.plugin("DecoyTarget")
    assert info is not None
    assert info.target is None  # only interceptors report a wrapping target


class TimingInterceptor(Interceptor):
    target = Model


def test_interceptor_target_is_surfaced_as_a_plugin_fact():
    snapshot = inspect_registry(_registry(TimingInterceptor()))
    info = snapshot.plugin("TimingInterceptor")
    assert info is not None
    # The wrapping edge is exposed even though `requires` does not carry it.
    assert info.target == "Model"
    assert info.provides == ("Interceptor",)
    assert snapshot.capability(Interceptor).selects_one is False


def _dup_tool(module: str) -> Tool:
    """A Tool subclass named ``Dup`` that reports ``module`` as its module."""

    class Dup(Tool):
        def run(self, arguments, ctx):  # pragma: no cover
            return ""

    Dup.__module__ = module
    Dup.__qualname__ = "Dup"
    return Dup()


def test_qualified_name_disambiguates_same_named_plugins():
    a = _dup_tool("pkg_a")
    b = _dup_tool("pkg_b")
    infos = inspect_registry(_registry(a, b)).plugins
    # Same class name, different modules -> distinct, module-qualified names.
    assert [i.name for i in infos] == ["Dup", "Dup"]
    assert {i.qualified_name for i in infos} == {"pkg_a.Dup", "pkg_b.Dup"}
