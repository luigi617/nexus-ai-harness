from __future__ import annotations

from typing import ClassVar

import pytest

from core.response import Response
from harness import MissingDependencyError, NexusAIHarness
from harness.registry import Registry
from harness.validation import (
    describe_registry,
    validate_registry,
)
from plugins.loops import AgenticLoop, ChatLoop
from protocols.context_manager import ContextManager
from protocols.hook import Hook
from protocols.loop import Loop
from protocols.model import Model
from protocols.plugin import Plugin
from protocols.router import Router
from protocols.tool import Tool
from tests.conftest import RecordingTool, ScriptedModel


class NeedsModelAndTool(Loop):
    """A plugin declaring dependencies, used to exercise validation directly."""

    requires: ClassVar[tuple[type[Plugin], ...]] = (Model, Tool)

    def run(self, ctx):  # pragma: no cover - never executed here
        return ""


class NeedsNothing(Hook):
    # Inherits no `requires`; validation must treat it as satisfied.

    def on(self, event, ctx):  # pragma: no cover
        pass


def _registry(*plugins: object) -> Registry:
    reg = Registry()
    for plugin in plugins:
        reg.add(plugin)
    return reg


def test_plugin_without_requires_is_treated_as_no_deps():
    # Concrete plugins that don't declare `requires` are read as no-deps and so
    # never contribute a missing dependency.
    validate_registry(_registry(RecordingTool()))  # must not raise


def test_loops_declare_model_dependency():
    assert Model in AgenticLoop.requires
    assert Model in ChatLoop.requires


def test_validate_passes_when_dependencies_present():
    reg = _registry(
        NeedsModelAndTool(), ScriptedModel(Response(text="x")), RecordingTool()
    )
    assert validate_registry(reg) is None


def test_validate_no_requires_is_noop():
    validate_registry(_registry(NeedsNothing()))  # must not raise


def test_validate_empty_registry_is_noop():
    validate_registry(Registry())  # must not raise


def test_validate_raises_on_missing_dependency():
    reg = _registry(NeedsModelAndTool(), ScriptedModel(Response(text="x")))
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(reg)
    err = exc.value
    assert err.missing == [("NeedsModelAndTool", "Tool")]
    # The rendered tree marks the satisfied and missing deps and names the gap.
    assert "NeedsModelAndTool" in err.report
    assert "Model ✓" in err.report
    assert "Tool ✗" in err.report
    assert "Missing dependency: NeedsModelAndTool requires Tool" in err.report


def test_validate_reports_all_missing_across_plugins():
    reg = _registry(NeedsModelAndTool())  # both Model and Tool absent
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(reg)
    assert exc.value.missing == [
        ("NeedsModelAndTool", "Model"),
        ("NeedsModelAndTool", "Tool"),
    ]


def test_tree_uses_branch_glyphs():
    reg = _registry(NeedsModelAndTool())
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(reg)
    report = exc.value.report
    assert "├── Model" in report  # non-last branch
    assert "└── Tool" in report  # last branch


def test_single_requirement_renders_satisfied_last_branch():
    # One requirement: its only row is the last, so it uses the last-branch glyph.
    reg = _registry(AgenticLoop(), ScriptedModel(Response(text="x")))
    assert describe_registry(reg) == "AgenticLoop\n└── Model ✓"


class NeedsModel(Router):
    requires: ClassVar[tuple[type[Plugin], ...]] = (Model,)

    def route(self, history, ctx):  # pragma: no cover
        return None


def test_validate_filters_to_only_unsatisfied_plugin_trees():
    # Only the unsatisfied plugin's tree renders; missing tuples in registry order.
    reg = _registry(
        NeedsModel(), NeedsModelAndTool(), ScriptedModel(Response(text="x"))
    )
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(reg)
    report = exc.value.report
    assert exc.value.missing == [("NeedsModelAndTool", "Tool")]
    assert "NeedsModel\n" not in report  # satisfied plugin's tree omitted
    assert "NeedsModelAndTool" in report


def test_validate_missing_ordering_across_plugins():
    # Neither Model nor Tool present: both plugins report, in registry order.
    reg = _registry(NeedsModel(), NeedsModelAndTool())
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(reg)
    assert exc.value.missing == [
        ("NeedsModel", "Model"),
        ("NeedsModelAndTool", "Model"),
        ("NeedsModelAndTool", "Tool"),
    ]


def test_describe_registry_joins_multiple_trees():
    reg = _registry(NeedsModel(), AgenticLoop(), ScriptedModel(Response(text="x")))
    desc = describe_registry(reg)
    # Two trees, blank-line separated, in registry order.
    assert desc == "NeedsModel\n└── Model ✓\n\nAgenticLoop\n└── Model ✓"


def test_plugin_requiring_its_own_type_needs_another_provider():
    # A plugin that is itself a Model but requires one is NOT self-satisfying.
    class SelfWrapModel(Model):
        requires: ClassVar[tuple[type[Plugin], ...]] = (Model,)

        def complete(self, history, ctx):  # pragma: no cover
            return Response(text="")

    # Only the wrapper is registered → its Model requirement is unmet.
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(_registry(SelfWrapModel()))
    assert exc.value.missing == [("SelfWrapModel", "Model")]

    # Add a second, distinct Model → the requirement is now satisfied.
    validate_registry(_registry(SelfWrapModel(), ScriptedModel(Response(text="x"))))


def test_harness_validate_chains_and_passes():
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))
    assert h.validate() is h  # returns self for chaining


def test_harness_validate_raises_when_model_missing():
    h = NexusAIHarness().use(AgenticLoop())
    with pytest.raises(MissingDependencyError) as exc:
        h.validate()
    assert exc.value.missing == [("AgenticLoop", "Model")]


def test_run_validates_before_executing():
    # run() validates on start, so an invalid harness fails up front, not lazily.
    h = NexusAIHarness().use(AgenticLoop())
    with pytest.raises(MissingDependencyError):
        h.run_sync("q")


def test_describe_dependencies_lists_all_trees():
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))
    desc = h.describe_dependencies()
    assert "AgenticLoop" in desc
    assert "Model ✓" in desc


def test_describe_registry_empty_when_no_requires():
    assert describe_registry(_registry(NeedsNothing())) == ""


def test_concrete_class_requirement_needs_the_exact_plugin():
    from plugins.guards import BudgetGuard
    from plugins.hooks import CostCounter, IterationCounter

    # BudgetGuard requires the concrete CostCounter; a different hook won't satisfy it.
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(_registry(BudgetGuard(5.0), IterationCounter()))
    assert exc.value.missing == [("BudgetGuard", "CostCounter")]

    # The exact plugin satisfies it.
    validate_registry(_registry(BudgetGuard(5.0), CostCounter()))


def test_concrete_class_requirement_matches_subclass():
    from plugins.guards import BudgetGuard
    from plugins.hooks import CostCounter

    class TieredCostCounter(CostCounter):
        pass

    # A subclass of the required plugin still satisfies the dependency.
    validate_registry(_registry(BudgetGuard(5.0), TieredCostCounter()))


def test_base_requirement_matches_any_subclass():
    # A base dependency is satisfied by ANY subclass, unlike a concrete-class one.
    validate_registry(_registry(NeedsModel(), ScriptedModel(Response(text="x"))))


def test_memory_tools_require_a_memory_store(tmp_path):
    from plugins.memory import FileMemoryStore
    from plugins.tools import Recall

    # Recall declares requires=(MemoryStore,): validation flags a missing store.
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(_registry(Recall()))
    assert exc.value.missing == [("Recall", "MemoryStore")]

    # A registered store satisfies it.
    validate_registry(_registry(Recall(), FileMemoryStore(str(tmp_path))))


def test_default_harness_validates(tmp_path):
    from plugins import default_harness

    # The default harness wires every required hook and model, so validation passes.
    h = default_harness(ScriptedModel(Response(text="x")), memory_dir=str(tmp_path))
    assert h.validate() is h


def test_context_manager_dependency_scenario():
    # Mirrors the issue's example: a loop-like plugin also needing a ContextManager.
    class NeedsCM(Loop):
        requires: ClassVar[tuple[type[Plugin], ...]] = (Model, ContextManager)

        def run(self, ctx):  # pragma: no cover
            return ""

    reg = _registry(NeedsCM(), ScriptedModel(Response(text="x")))
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(reg)
    assert exc.value.missing == [("NeedsCM", "ContextManager")]
    assert "Model ✓" in exc.value.report
    assert "ContextManager ✗" in exc.value.report
