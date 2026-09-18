from __future__ import annotations

import asyncio

import pytest

from core.phase import Phase
from core.response import Response
from harness.harness import NexusAIHarness
from plugins.loops import AgenticLoop
from plugins.permissions import AllowList, AutoApprove
from protocols.context import Context
from protocols.interceptor import Interceptor
from protocols.model import Model
from protocols.tool import Tool
from services.invoke import InterceptorBinding, invoke
from tests.conftest import RecordingTool, ScriptedModel, make_ctx


class Mark(Interceptor):
    """Append a label to a shared list each time it runs."""

    kind = "interceptor"

    def __init__(self, label: str, sink: list[str]) -> None:
        self._label = label
        self._sink = sink

    def run(self, ctx: Context) -> None:
        self._sink.append(self._label)


def _harness(model, *plugins) -> NexusAIHarness:
    harness = NexusAIHarness().use(AgenticLoop()).use(model)
    for plugin in plugins:
        harness.use(plugin)
    return harness


def test_use_before_and_use_after_return_self_for_chaining():
    harness = NexusAIHarness()
    assert harness.use_before(Model, Mark("a", [])) is harness
    assert harness.use_after(Model, Mark("b", [])) is harness


def test_before_runs_ahead_of_target_after_runs_behind():
    order: list[str] = []
    harness = _harness(ScriptedModel(Response(text="hi")))
    harness.use_before(Model, Mark("before-model", order))
    harness.use_after(Model, Mark("after-model", order))
    harness.run_sync("go")
    # the model call itself sits between the two interceptors
    assert order == ["before-model", "after-model"]


def test_interceptor_fires_per_invocation():
    order: list[str] = []
    model = ScriptedModel(
        Response(text="", tool_calls=[{"name": "echo", "id": "1", "arguments": {}}]),
        Response(text="done"),
    )
    harness = _harness(
        model, RecordingTool("echo", "ok"), AllowList(["echo"]), AutoApprove()
    )
    harness.use_after(Tool, Mark("tool", order))
    harness.run_sync("go")
    assert order == ["tool"]  # one echo call -> one firing


def test_interceptor_only_fires_for_its_target():
    order: list[str] = []
    harness = _harness(ScriptedModel(Response(text="hi")))
    harness.use_before(Tool, Mark("tool", order))  # no tool is ever invoked
    harness.run_sync("go")
    assert order == []


def test_binding_to_a_concrete_class_ignores_a_sibling_implementation():
    class OtherModel(ScriptedModel):
        pass

    order: list[str] = []
    harness = _harness(ScriptedModel(Response(text="hi")))  # not OtherModel
    harness.use_before(OtherModel, Mark("other", order))
    harness.run_sync("go")
    assert order == []  # the registered model is not an OtherModel


def test_binding_to_a_protocol_matches_any_implementer():
    order: list[str] = []
    harness = _harness(ScriptedModel(Response(text="hi")))
    harness.use_before(Model, Mark("model", order))  # protocol, not a class
    harness.run_sync("go")
    assert order == ["model"]


def test_async_interceptor_is_awaited():
    order: list[str] = []

    class AsyncMark(Interceptor):
        kind = "interceptor"

        async def run(self, ctx: Context) -> None:
            order.append("async")

    harness = _harness(ScriptedModel(Response(text="hi")))
    harness.use_before(Model, AsyncMark())
    harness.run_sync("go")
    assert order == ["async"]


class _Widget:
    """A minimal target plugin for exercising invoke() directly."""

    kind = "widget"

    async def go(self) -> str:
        return "ok"

    async def boom(self) -> str:
        raise ValueError("boom")


def test_after_interceptors_run_even_when_invocation_raises():
    order: list[str] = []
    widget = _Widget()
    ctx = make_ctx(InterceptorBinding(_Widget, Phase.AFTER, Mark("after", order)))
    with pytest.raises(ValueError):
        asyncio.run(invoke(ctx, widget.boom))
    assert order == ["after"]


def test_before_and_after_both_run_in_registration_order():
    order: list[str] = []
    widget = _Widget()
    ctx = make_ctx(
        InterceptorBinding(_Widget, Phase.BEFORE, Mark("before-A", order)),
        InterceptorBinding(_Widget, Phase.AFTER, Mark("after-A", order)),
        InterceptorBinding(_Widget, Phase.BEFORE, Mark("before-B", order)),
        InterceptorBinding(_Widget, Phase.AFTER, Mark("after-B", order)),
    )
    asyncio.run(invoke(ctx, widget.go))
    assert order == ["before-A", "before-B", "after-A", "after-B"]


def test_binding_kind_is_distinct_from_interceptor():
    ctx = make_ctx(InterceptorBinding(_Widget, Phase.BEFORE, Mark("x", [])))
    assert ctx.get(Interceptor) is None  # a binding is not resolvable as Interceptor
    assert len(ctx.all(InterceptorBinding)) == 1


def test_a_plugin_invoking_another_plugin_is_also_intercepted():
    order: list[str] = []

    class Inner:
        kind = "inner"

        async def go(self) -> str:
            return "inner"

    class Outer:
        kind = "outer"

        def __init__(self, inner: Inner) -> None:
            self._inner = inner

        async def go(self, ctx: Context) -> str:  # a plugin using another plugin
            return await invoke(ctx, self._inner.go)

    outer = Outer(Inner())
    ctx = make_ctx(InterceptorBinding(Inner, Phase.BEFORE, Mark("inner", order)))
    asyncio.run(invoke(ctx, outer.go, ctx))
    assert order == ["inner"]  # fires for the nested call, not just top-level
