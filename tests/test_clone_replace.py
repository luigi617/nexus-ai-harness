from __future__ import annotations

import asyncio

import pytest

from core.events import Event, ModelCallStarted
from core.response import Response
from harness import NexusAIHarness, PluginStatus
from harness.registry import Registry
from plugins.loops import AgenticLoop
from protocols.context import Context
from protocols.lifecycle import Lifecycle
from protocols.loop import Loop
from protocols.model import Model
from protocols.plugin import Plugin
from tests.conftest import ScriptedModel


class MarkerLoop(Loop):
    """A loop that returns a fixed marker, standing in for an alternate loop."""

    requires = (Model,)

    def __init__(self, marker: str = "variant") -> None:
        self.marker = marker

    def run(self, ctx: Context) -> str:
        return self.marker


class StartTracker(Plugin, Lifecycle):
    """Records how many times start() ran, to prove instances are shared."""

    def __init__(self) -> None:
        self.starts = 0

    async def start(self, ctx: Context) -> None:
        self.starts += 1

    async def stop(self) -> None:
        pass


def _base() -> NexusAIHarness:
    return NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))


def test_clone_registry_is_independent():
    base = _base()
    twin = base.clone()
    twin.use(MarkerLoop())
    assert len(twin._registry.plugins()) == 3
    assert len(base._registry.plugins()) == 2


def test_clone_shares_plugin_instances():
    base = _base()
    model = base._registry.get(Model)
    twin = base.clone()
    assert twin._registry.get(Model) is model


def test_clone_is_unstarted_with_fresh_status():
    base = _base()

    async def go() -> NexusAIHarness:
        await base.start()
        return base.clone()

    twin = asyncio.run(go())
    assert twin._started is False
    for plugin in twin._registry.plugins():
        assert twin._registry.status_of(plugin) is PluginStatus.REGISTERED


def test_clone_of_started_harness_reruns_lifecycle_on_shared_instance():
    tracker = StartTracker()
    base = _base().use(tracker)

    async def go() -> None:
        await base.start()
        assert tracker.starts == 1
        twin = base.clone()
        await twin.start()  # same instance, fresh registry -> starts again

    asyncio.run(go())
    assert tracker.starts == 2


def test_replace_swaps_provider_and_leaves_base_untouched():
    base = _base()
    variant = base.clone().replace(Loop, MarkerLoop(marker="tree"))
    assert variant.run_sync("q").output == "tree"
    assert base.run_sync("q").output == "hi"  # base still runs the original loop


def test_replace_preserves_registration_slot_for_single_select():
    base = _base()
    variant = base.clone().replace(Loop, MarkerLoop(marker="tree"))
    loops = variant._registry.all(Loop)
    assert len(loops) == 1
    assert isinstance(loops[0], MarkerLoop)


def test_replace_drops_additional_matches():
    variant = _base().use(MarkerLoop(marker="second")).clone()  # two loops registered
    replacement = MarkerLoop(marker="only")
    variant.replace(Loop, replacement)
    assert variant._registry.all(Loop) == [replacement]


def test_replace_returns_self_for_chaining():
    twin = _base().clone()
    assert twin.replace(Loop, MarkerLoop()) is twin


def test_replace_raises_when_no_provider_matches():
    twin = NexusAIHarness().use(ScriptedModel(Response(text="hi"))).clone()
    with pytest.raises(LookupError):
        twin.replace(Loop, MarkerLoop())


def test_replace_rejects_non_plugin():
    twin = _base().clone()
    with pytest.raises(TypeError):
        twin.replace(Loop, object())


def test_replace_rejects_plugin_that_does_not_provide_capability():
    # Replacing with a plugin that isn't a Loop would strip the loop and defer
    # the failure to run time; reject it up front instead.
    twin = _base().clone()
    with pytest.raises(TypeError):
        twin.replace(Loop, ScriptedModel(Response(text="x")))
    assert len(twin._registry.all(Loop)) == 1  # original loop untouched


def test_replace_on_started_harness_raises():
    base = _base()

    async def go() -> None:
        await base.start()
        base.replace(Loop, MarkerLoop())

    with pytest.raises(RuntimeError):
        asyncio.run(go())


def test_clone_does_not_carry_subscriptions():
    seen: list[Event] = []

    class Tracer(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            ctx.on(ModelCallStarted, lambda event, _ctx: seen.append(event))

    base = _base().use(Tracer())

    async def go() -> None:
        await base.start()
        twin = base.clone()
        assert twin._registry.subscriptions() == []

    asyncio.run(go())


def test_registry_clone_copies_entries_without_status():
    registry = Registry()
    loop = AgenticLoop()
    registry.add(loop)
    registry.set_status(loop, PluginStatus.STARTED)
    copy = registry.clone()
    assert copy.plugins() == [loop]
    assert copy.status_of(loop) is PluginStatus.REGISTERED
