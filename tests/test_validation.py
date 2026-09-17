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
from protocols.model import Model
from protocols.plugin import Plugin
from protocols.tool import Tool
from tests.conftest import RecordingTool, ScriptedModel


class NeedsModelAndTool:
    """A plugin declaring dependencies, used to exercise validation directly."""

    kind: ClassVar[str] = "loop"
    requires: ClassVar[tuple[type[Plugin], ...]] = (Model, Tool)

    def run(self, ctx):  # pragma: no cover - never executed here
        return ""


class NeedsNothing:
    kind: ClassVar[str] = "hook"
    # inherits no `requires`; validation must treat it as satisfied

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
    # Returns None and does not raise.
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


def test_harness_validate_chains_and_passes():
    h = (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(ScriptedModel(Response(text="hi")))
    )
    assert h.validate() is h  # returns self for chaining


def test_harness_validate_raises_when_model_missing():
    h = NexusAIHarness().use(AgenticLoop())
    with pytest.raises(MissingDependencyError) as exc:
        h.validate()
    assert exc.value.missing == [("AgenticLoop", "Model")]


def test_harness_validate_does_not_change_run_behavior():
    # An invalid harness still resolves lazily at run time (old behavior): a
    # loop with no model raises LookupError from run(), not the validation path.
    h = NexusAIHarness().use(AgenticLoop())
    with pytest.raises(LookupError):
        h.run_sync("q")


def test_describe_dependencies_lists_all_trees():
    h = (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(ScriptedModel(Response(text="hi")))
    )
    desc = h.describe_dependencies()
    assert "AgenticLoop" in desc
    assert "Model ✓" in desc


def test_describe_registry_empty_when_no_requires():
    assert describe_registry(_registry(NeedsNothing())) == ""


def test_context_manager_dependency_scenario():
    # Mirrors the issue's example: a loop-like plugin that also needs a
    # ContextManager surfaces the missing one.
    class NeedsCM:
        kind: ClassVar[str] = "loop"
        requires: ClassVar[tuple[type[Plugin], ...]] = (Model, ContextManager)

        def run(self, ctx):  # pragma: no cover
            return ""

    reg = _registry(NeedsCM(), ScriptedModel(Response(text="x")))
    with pytest.raises(MissingDependencyError) as exc:
        validate_registry(reg)
    assert exc.value.missing == [("NeedsCM", "ContextManager")]
    assert "Model ✓" in exc.value.report
    assert "ContextManager ✗" in exc.value.report
