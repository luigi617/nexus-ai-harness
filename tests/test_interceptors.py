from __future__ import annotations

import asyncio

import pytest

from core.phase import Phase
from core.response import Response
from harness.context import RunContext
from harness.harness import NexusAIHarness
from harness.registry import Registry
from harness.session import Session
from plugins.loops import AgenticLoop
from plugins.permissions import AllowList, AutoApprove
from protocols.context import Context
from protocols.interceptor import Interceptor
from protocols.loop import Loop
from protocols.model import Model
from protocols.tool import Tool
from tests.conftest import RecordingTool, ScriptedModel


def _ctx_with(*bindings: tuple[type, Phase, Interceptor]) -> RunContext:
    """A RunContext whose registry holds the given (target, phase, interceptor)s."""
    registry = Registry()
    for target, phase, interceptor in bindings:
        registry.add_interceptor(target, phase, interceptor)
    return RunContext(Session(), registry)


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


def test_target_protocol_does_not_leak_across_protocols_sharing_a_method():
    # Tool structurally satisfies the minimal Loop protocol (both have `run`),
    # so a Loop-bound interceptor must not fire on tool calls.
    order: list[str] = []
    model = ScriptedModel(
        Response(text="", tool_calls=[{"name": "echo", "id": "1", "arguments": {}}]),
        Response(text="done"),
    )
    harness = _harness(
        model, RecordingTool("echo", "ok"), AllowList(["echo"]), AutoApprove()
    )
    harness.use_before(Loop, Mark("loop", order))
    harness.run_sync("go")
    assert order == ["loop"]  # once for the loop, not also per tool call


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
    ctx = _ctx_with((_Widget, Phase.AFTER, Mark("after", order)))
    with pytest.raises(ValueError):
        asyncio.run(ctx.invoke(widget.boom))
    assert order == ["after"]


def test_before_and_after_both_run_in_registration_order():
    order: list[str] = []
    widget = _Widget()
    ctx = _ctx_with(
        (_Widget, Phase.BEFORE, Mark("before-A", order)),
        (_Widget, Phase.AFTER, Mark("after-A", order)),
        (_Widget, Phase.BEFORE, Mark("before-B", order)),
        (_Widget, Phase.AFTER, Mark("after-B", order)),
    )
    asyncio.run(ctx.invoke(widget.go))
    assert order == ["before-A", "before-B", "after-A", "after-B"]


def test_invoke_passes_through_args_and_kwargs():
    class Adder:
        kind = "adder"

        async def add(self, a: int, b: int = 0) -> int:
            return a + b

    ctx = _ctx_with()
    assert asyncio.run(ctx.invoke(Adder().add, 1, b=2)) == 3


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
            return await ctx.invoke(self._inner.go)

    outer = Outer(Inner())
    ctx = _ctx_with((Inner, Phase.BEFORE, Mark("inner", order)))
    asyncio.run(ctx.invoke(outer.go, ctx))
    assert order == ["inner"]  # fires for the nested call, not just top-level
