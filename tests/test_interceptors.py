from __future__ import annotations

from core.response import Response
from harness.harness import NexusAIHarness
from plugins.loops import AgenticLoop
from plugins.permissions import AllowList, AutoApprove
from protocols.context import Context
from protocols.interceptor import Interceptor
from protocols.model import Model
from protocols.tool import Tool
from tests.conftest import RecordingTool, ScriptedModel


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
