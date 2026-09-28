from __future__ import annotations

import importlib

import pytest

from nexus_ai_harness.core.response import Response
from nexus_ai_harness.harness.context import RunContext
from nexus_ai_harness.harness.registry import Registry
from nexus_ai_harness.harness.session import Session
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.tool import Tool


def tau2_importable() -> bool:
    """Whether tau2 can actually be imported, not merely located."""
    try:
        importlib.import_module("tau2")
    except ImportError:
        return False
    return True


if not tau2_importable():
    collect_ignore_glob = ["test_bench_tau2.py"]


def make_ctx(*plugins: object) -> RunContext:
    """A RunContext over a registry holding the given plugins."""
    registry = Registry()
    for plugin in plugins:
        registry.add(plugin)
    return RunContext(Session(), registry)


class ScriptedModel(Model):
    """Returns queued Responses in order; the last repeats once exhausted.

    Records the histories it was called with, for assertions.
    """

    def __init__(self, *responses: Response) -> None:
        self._responses = list(responses) or [Response(text="done")]
        self.calls: list[list] = []

    async def complete(self, history, tools, ctx) -> Response:
        self.calls.append(list(history))
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


class RecordingTool(Tool):
    """A sync tool that records its invocations and returns a fixed result."""

    def __init__(self, name: str = "echo", result: str = "ok") -> None:
        self.name = name
        self.description = ""
        self.parameters = {}
        self._result = result
        self.calls: list[dict] = []

    def run(self, arguments: dict, ctx) -> str:
        self.calls.append(arguments)
        return self._result


@pytest.fixture
def response():
    return Response
