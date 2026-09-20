from __future__ import annotations

import asyncio

import pytest

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
from protocols.plugin import Plugin
from protocols.tool import Tool
from tests.conftest import RecordingTool, ScriptedModel


class BeforeMark(Interceptor):
    """Appends a label before each invocation of ``target``."""

    def __init__(self, target: type[Plugin], label: str, sink: list[str]) -> None:
        self.target = target
        self._label = label
        self._sink = sink

    def before(self, ctx: Context) -> None:
        self._sink.append(self._label)


class AfterMark(Interceptor):
    """Appends a label after each invocation of ``target``."""

    def __init__(self, target: type[Plugin], label: str, sink: list[str]) -> None:
        self.target = target
        self._label = label
        self._sink = sink

    def after(self, ctx: Context) -> None:
        self._sink.append(self._label)


def _ctx_with(*interceptors: Interceptor) -> RunContext:
    """A RunContext whose registry holds the given interceptors."""
    registry = Registry()
    for interceptor in interceptors:
        registry.add(interceptor)
    return RunContext(Session(), registry)


def _harness(model, *plugins) -> NexusAIHarness:
    harness = NexusAIHarness().use(AgenticLoop()).use(model)
    for plugin in plugins:
        harness.use(plugin)
    return harness


def test_use_returns_self_for_chaining():
    harness = NexusAIHarness()
    assert harness.use(BeforeMark(Model, "a", [])) is harness


def test_registering_an_interceptor_without_target_is_rejected():
    class Bare(Interceptor):
        def before(self, ctx: Context) -> None: ...

    with pytest.raises(TypeError, match="declares no 'target'"):
        NexusAIHarness().use(Bare())


def test_before_runs_ahead_of_target_after_runs_behind():
    order: list[str] = []
    harness = _harness(
        ScriptedModel(Response(text="hi")),
        BeforeMark(Model, "before-model", order),
        AfterMark(Model, "after-model", order),
    )
    harness.run_sync("go")
    # the model call itself sits between the two interceptors
    assert order == ["before-model", "after-model"]


def test_one_interceptor_can_wrap_both_phases():
    order: list[str] = []

    class Around(Interceptor):
        target = Model

        def before(self, ctx: Context) -> None:
            order.append("before")

        def after(self, ctx: Context) -> None:
            order.append("after")

    _harness(ScriptedModel(Response(text="hi")), Around()).run_sync("go")
    assert order == ["before", "after"]


def test_interceptor_fires_per_invocation():
    order: list[str] = []
    model = ScriptedModel(
        Response(text="", tool_calls=[{"name": "echo", "id": "1", "arguments": {}}]),
        Response(text="done"),
    )
    harness = _harness(
        model,
        RecordingTool("echo", "ok"),
        AllowList(["echo"]),
        AutoApprove(),
        AfterMark(Tool, "tool", order),
    )
    harness.run_sync("go")
    assert order == ["tool"]  # one echo call -> one firing


def test_interceptor_only_fires_for_its_target():
    order: list[str] = []
    harness = _harness(
        ScriptedModel(Response(text="hi")),
        BeforeMark(Tool, "tool", order),  # no tool is ever invoked
    )
    harness.run_sync("go")
    assert order == []


def test_target_type_does_not_leak_across_types_sharing_a_method():
    # Tool and Loop both expose `run`, but a Tool does not subclass Loop, so a
    # Loop-bound interceptor must not fire on tool calls.
    order: list[str] = []
    model = ScriptedModel(
        Response(text="", tool_calls=[{"name": "echo", "id": "1", "arguments": {}}]),
        Response(text="done"),
    )
    harness = _harness(
        model,
        RecordingTool("echo", "ok"),
        AllowList(["echo"]),
        AutoApprove(),
        BeforeMark(Loop, "loop", order),
    )
    harness.run_sync("go")
    assert order == ["loop"]  # once for the loop, not also per tool call


def test_wrapping_a_concrete_class_ignores_a_sibling_implementation():
    class OtherModel(ScriptedModel):
        pass

    order: list[str] = []
    harness = _harness(
        ScriptedModel(Response(text="hi")),  # not OtherModel
        BeforeMark(OtherModel, "other", order),
    )
    harness.run_sync("go")
    assert order == []  # the registered model is not an OtherModel


def test_wrapping_a_protocol_matches_any_implementer():
    order: list[str] = []
    harness = _harness(
        ScriptedModel(Response(text="hi")),
        BeforeMark(Model, "model", order),  # protocol, not a class
    )
    harness.run_sync("go")
    assert order == ["model"]


def test_async_interceptor_is_awaited():
    order: list[str] = []

    class AsyncMark(Interceptor):
        target = Model

        async def before(self, ctx: Context) -> None:
            order.append("async")

    _harness(ScriptedModel(Response(text="hi")), AsyncMark()).run_sync("go")
    assert order == ["async"]


def test_unuse_stops_an_interceptor_wrapping_its_target():
    order: list[str] = []
    interceptor = BeforeMark(Model, "model", order)
    harness = _harness(ScriptedModel(Response(text="hi")), interceptor)

    async def go() -> None:
        await harness.run("go")
        await harness.unuse(interceptor)
        await harness.run("go")  # binding gone, so it must not fire again

    asyncio.run(go())
    assert order == ["model"]


class _Widget(Plugin):
    """A minimal target plugin for exercising invoke() directly."""

    async def go(self) -> str:
        return "ok"

    async def boom(self) -> str:
        raise ValueError("boom")


def test_after_interceptors_run_even_when_invocation_raises():
    order: list[str] = []
    widget = _Widget()
    ctx = _ctx_with(AfterMark(_Widget, "after", order))
    with pytest.raises(ValueError):
        asyncio.run(ctx.invoke(widget.boom))
    assert order == ["after"]


def test_before_and_after_both_run_in_registration_order():
    order: list[str] = []
    widget = _Widget()
    ctx = _ctx_with(
        BeforeMark(_Widget, "before-A", order),
        AfterMark(_Widget, "after-A", order),
        BeforeMark(_Widget, "before-B", order),
        AfterMark(_Widget, "after-B", order),
    )
    asyncio.run(ctx.invoke(widget.go))
    assert order == ["before-A", "before-B", "after-A", "after-B"]


def test_invoke_passes_through_args_and_kwargs():
    class Adder(Plugin):
        async def add(self, a: int, b: int = 0) -> int:
            return a + b

    ctx = _ctx_with()
    assert asyncio.run(ctx.invoke(Adder().add, 1, b=2)) == 3


def test_a_plugin_invoking_another_plugin_is_also_intercepted():
    order: list[str] = []

    class Inner(Plugin):
        async def go(self) -> str:
            return "inner"

    class Outer(Plugin):
        def __init__(self, inner: Inner) -> None:
            self._inner = inner

        async def go(self, ctx: Context) -> str:  # a plugin using another plugin
            return await ctx.invoke(self._inner.go)

    outer = Outer(Inner())
    ctx = _ctx_with(BeforeMark(Inner, "inner", order))
    asyncio.run(ctx.invoke(outer.go, ctx))
    assert order == ["inner"]  # fires for the nested call, not just top-level
