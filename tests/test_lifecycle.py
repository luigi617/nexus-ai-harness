from __future__ import annotations

import asyncio
import contextlib

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


def test_concurrent_run_waits_for_start_to_complete():
    # A concurrent run() must not execute its session until lifecycle init has
    # fully finished — not merely started.
    seen_ready: list[bool] = []

    class SlowResource(Plugin, Lifecycle):
        def __init__(self) -> None:
            self.ready = False

        async def start(self, ctx: Context) -> None:
            await asyncio.sleep(0.01)
            self.ready = True

    resource = SlowResource()

    class Probe(ScriptedModel):
        async def complete(self, history, ctx):
            seen_ready.append(resource.ready)
            return Response(text="ok")

    h = NexusAIHarness().use(AgenticLoop()).use(Probe()).use(resource)

    async def go() -> None:
        await asyncio.gather(h.run("a"), h.run("b"))

    asyncio.run(go())
    assert seen_ready == [True, True]  # both runs saw a fully-started resource


def test_rollback_suppresses_ordinary_teardown_error_and_surfaces_start_error():
    log: list[str] = []

    class NoisyStop(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            log.append("start:ns")

        async def stop(self) -> None:
            log.append("stop:ns")
            raise ValueError("teardown boom")  # ordinary error, suppressed

    class FailStart(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            raise RuntimeError("start failed")

    h = build(NoisyStop(), FailStart())

    # The original start error surfaces; the rollback teardown still ran.
    with pytest.raises(RuntimeError, match="start failed"):
        asyncio.run(h.start())
    assert log == ["start:ns", "stop:ns"]


def test_rollback_lets_cancellation_through():
    class CancelStop(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None: ...

        async def stop(self) -> None:
            raise asyncio.CancelledError  # cancellation must not be swallowed

    class FailStart(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            raise RuntimeError("start failed")

    h = build(CancelStop(), FailStart())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(h.start())


def test_use_after_start_is_rejected():
    h = build()
    asyncio.run(h.start())
    with pytest.raises(RuntimeError, match="after the harness has started"):
        h.use(Resource("late", []))


def test_baseexception_mid_stop_leaves_harness_started_for_retry():
    log: list[str] = []

    class Interrupted(BaseException):
        pass

    class Flaky(Plugin, Lifecycle):
        def __init__(self) -> None:
            self._fail = True

        async def stop(self) -> None:
            if self._fail:
                self._fail = False
                raise Interrupted  # BaseException interrupts the first sweep
            log.append("stop:flaky")

    class Res(Plugin, Lifecycle):
        async def stop(self) -> None:
            log.append("stop:res")

    # teardown order (reverse of registration): Flaky first, then Res
    h = build(Res(), Flaky())

    async def go() -> None:
        await h.start()
        with contextlib.suppress(Interrupted):
            await h.stop()  # Flaky interrupts; harness must stay started
        assert h._started is True
        await h.stop()  # retry resumes and completes teardown

    asyncio.run(go())
    assert h._started is False
    assert log == ["stop:flaky", "stop:res"]


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


def test_aexit_body_error_wins_over_stop_error():
    class FailStop(Plugin, Lifecycle):
        async def stop(self) -> None:
            raise ValueError("stop boom")

    h = build(FailStop())

    async def go() -> None:
        async with h:
            raise RuntimeError("body failed")

    # The body's exception propagates, not the teardown's.
    with pytest.raises(RuntimeError, match="body failed"):
        asyncio.run(go())
