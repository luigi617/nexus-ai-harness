from __future__ import annotations

import asyncio

from core.events import Event, ModelCallStarted
from core.response import Response
from harness import NexusAIHarness
from plugins.loops import AgenticLoop
from protocols.context import Context
from protocols.interceptor import Interceptor
from protocols.lifecycle import Lifecycle
from protocols.model import Model
from protocols.plugin import Plugin
from tests.conftest import ScriptedModel


class TracingPlugin(Plugin, Lifecycle):
    """Subscribes to model calls at start and records that it was stopped."""

    def __init__(self) -> None:
        self.seen: list[Event] = []
        self.stopped = False

    async def start(self, ctx: Context) -> None:
        ctx.on(ModelCallStarted, self.before_model)

    async def stop(self) -> None:
        self.stopped = True

    def before_model(self, event: Event, ctx: Context) -> None:
        self.seen.append(event)


class CountingInterceptor(Interceptor):
    target = Model

    def __init__(self) -> None:
        self.runs = 0

    def before(self, ctx: Context) -> None:
        self.runs += 1


def _harness(*plugins: object) -> NexusAIHarness:
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))
    for plugin in plugins:
        h.use(plugin)
    return h


def test_owned_subscription_fires_while_registered():
    tracer = TracingPlugin()
    h = _harness(tracer)
    h.run_sync("q")
    assert len(tracer.seen) == 1


def test_unuse_removes_owned_subscriptions():
    tracer = TracingPlugin()
    h = _harness(tracer)

    async def go() -> None:
        await h.run("q")
        await h.unuse(tracer)
        await h.run("q")  # the tracer's handler must no longer fire

    asyncio.run(go())
    assert len(tracer.seen) == 1


def test_unuse_stops_a_started_lifecycle_plugin():
    tracer = TracingPlugin()
    h = _harness(tracer)

    async def go() -> None:
        await h.start()
        await h.unuse(tracer)

    asyncio.run(go())
    assert tracer.stopped is True


def test_unuse_before_start_does_not_stop():
    tracer = TracingPlugin()
    h = _harness(tracer)
    asyncio.run(h.unuse(tracer))
    assert tracer.stopped is False


def test_unuse_removes_the_plugin_from_the_registry():
    tracer = TracingPlugin()
    h = _harness(tracer)
    asyncio.run(h.unuse(tracer))
    assert tracer not in h._registry.plugins()


def test_unuse_removes_owned_interceptor_bindings():
    interceptor = CountingInterceptor()
    h = _harness()
    h.use(interceptor)

    async def go() -> None:
        await h.run("q")
        assert interceptor.runs == 1
        await h.unuse(interceptor)
        await h.run("q")  # binding gone, so it must not run again

    asyncio.run(go())
    assert interceptor.runs == 1


def test_unuse_is_chainable_and_noop_for_absent_plugin():
    h = _harness()
    absent = TracingPlugin()

    async def go() -> NexusAIHarness:
        return await h.unuse(absent)

    assert asyncio.run(go()) is h
