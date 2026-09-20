from __future__ import annotations

import asyncio

import pytest

from core.response import Response
from harness import NexusAIHarness
from plugins.loops import AgenticLoop
from protocols.context import Context
from protocols.lifecycle import Lifecycle
from protocols.model import Model
from protocols.plugin import Plugin
from tests.conftest import ScriptedModel


class Resource(Plugin, Lifecycle):
    """A registrable lifecycle plugin that records its calls into ``log``."""

    def __init__(self, name: str, log: list[str]) -> None:
        self._name = name
        self._log = log

    async def start(self, ctx: Context) -> None:
        self._log.append(f"start:{self._name}")

    async def stop(self) -> None:
        self._log.append(f"stop:{self._name}")


def build(*extra: object) -> NexusAIHarness:
    h = NexusAIHarness().use(AgenticLoop()).use(ScriptedModel(Response(text="hi")))
    for plugin in extra:
        h.use(plugin)
    return h


def test_membership_is_by_inheritance():
    assert isinstance(Resource("a", []), Lifecycle)


def test_run_starts_lifecycle_plugins():
    log: list[str] = []
    build(Resource("a", log)).run_sync("q")
    assert "start:a" in log


def test_start_returns_self_for_chaining():
    h = build(Resource("a", []))

    async def go() -> None:
        assert await h.start() is h
        async with build(Resource("b", [])) as entered:
            assert isinstance(entered, NexusAIHarness)

    asyncio.run(go())


def test_start_runs_in_registration_order_stop_in_reverse():
    log: list[str] = []
    h = build(Resource("a", log), Resource("b", log))

    async def go() -> None:
        async with h:
            pass

    asyncio.run(go())
    assert log == ["start:a", "start:b", "stop:b", "stop:a"]


def test_start_is_idempotent_across_runs():
    log: list[str] = []
    h = build(Resource("a", log))
    h.run_sync("q")
    h.run_sync("q")
    assert log.count("start:a") == 1


def test_stop_without_start_is_a_noop():
    log: list[str] = []
    asyncio.run(build(Resource("a", log)).stop())
    assert log == []


def test_second_stop_is_a_noop():
    log: list[str] = []
    h = build(Resource("a", log))

    async def go() -> None:
        await h.start()
        await h.stop()
        await h.stop()  # a second stop must not tear down again

    asyncio.run(go())
    assert log == ["start:a", "stop:a"]


def test_stop_re_enables_a_later_start():
    log: list[str] = []
    h = build(Resource("a", log))

    async def go() -> None:
        await h.start()
        await h.stop()
        await h.start()

    asyncio.run(go())
    assert log == ["start:a", "stop:a", "start:a"]


def test_supports_sync_start_and_stop():
    log: list[str] = []

    class SyncResource(Plugin, Lifecycle):
        def start(self, ctx: Context) -> None:
            log.append("start")

        def stop(self) -> None:
            log.append("stop")

    async def go() -> None:
        async with build(SyncResource()):
            pass

    asyncio.run(go())
    assert log == ["start", "stop"]


def test_partial_lifecycle_start_only_and_stop_only():
    log: list[str] = []

    class StartOnly(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            log.append("start")

    class StopOnly(Plugin, Lifecycle):
        async def stop(self) -> None:
            log.append("stop")

    async def go() -> None:
        async with build(StartOnly(), StopOnly()):
            pass

    asyncio.run(go())
    assert log == ["start", "stop"]


def test_start_context_can_resolve_other_plugins():
    seen: list[Model | None] = []

    class Inspector(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            seen.append(ctx.get(Model))

    asyncio.run(build(Inspector()).start())
    assert isinstance(seen[0], Model)


def test_every_stop_runs_and_first_executed_error_is_reraised():
    log: list[str] = []

    class Boom(Plugin, Lifecycle):
        def __init__(self, name: str) -> None:
            self._name = name

        async def stop(self) -> None:
            log.append(f"stop:{self._name}")
            raise ValueError(self._name)

    # Two failing stops: the earlier-executed one (last registered, since stop
    # runs in reverse) must win, proving first-of-many rather than last-wins.
    h = build(Resource("a", log), Boom("early"), Resource("b", log), Boom("late"))

    async def go() -> None:
        await h.start()
        await h.stop()

    with pytest.raises(ValueError, match="late"):
        asyncio.run(go())
    assert log == ["start:a", "start:b", "stop:late", "stop:b", "stop:early", "stop:a"]


def test_partial_start_failure_rolls_back_started_plugins():
    log: list[str] = []

    class FailStart(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            log.append("start:boom")
            raise RuntimeError("start failed")

    h = build(Resource("a", log), FailStart(), Resource("b", log))

    with pytest.raises(RuntimeError, match="start failed"):
        asyncio.run(h.start())
    # a started, FailStart raised before b; a must be torn down, b never started
    assert log == ["start:a", "start:boom", "stop:a"]
    assert h._started is False


def test_concurrent_starts_do_not_double_initialize():
    log: list[str] = []

    class SlowStart(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            await asyncio.sleep(0)  # yield so a racing start() can interleave
            log.append("start")

    h = build(SlowStart())

    async def go() -> None:
        await asyncio.gather(h.start(), h.start())

    asyncio.run(go())
    assert log == ["start"]


def test_stop_propagates_cancellation_immediately():
    log: list[str] = []

    class Cancels(Plugin, Lifecycle):
        async def stop(self) -> None:
            raise asyncio.CancelledError

    # A trailing plugin whose stop() must NOT run once cancellation propagates.
    h = build(Resource("a", log), Cancels())

    async def go() -> None:
        await h.start()
        await h.stop()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(go())
    assert "stop:a" not in log  # cancellation short-circuits remaining teardown


def test_aexit_stops_even_when_body_raises():
    log: list[str] = []
    h = build(Resource("a", log))

    async def go() -> None:
        async with h:
            raise RuntimeError("body failed")

    with pytest.raises(RuntimeError, match="body failed"):
        asyncio.run(go())
    assert log == ["start:a", "stop:a"]
