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

    # Two failing stops: the earlier-executed (last registered) wins, not last-wins.
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
    # A started, FailStart raised before b; a must be torn down, b never started.
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
    # A concurrent run() waits for lifecycle init to fully finish, not just start.
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


def test_late_plugin_is_started_on_the_next_run():
    log: list[str] = []
    h = build()

    async def go() -> None:
        await h.start()
        h.use(Resource("late", log))  # added after the harness is running
        assert "start:late" not in log  # not started eagerly
        await h.run("q")

    asyncio.run(go())
    assert "start:late" in log


def test_late_plugin_started_once_across_further_runs():
    log: list[str] = []
    h = build()

    async def go() -> None:
        await h.start()
        h.use(Resource("late", log))
        await h.run("q")
        await h.run("q")

    asyncio.run(go())
    assert log.count("start:late") == 1


def test_late_plugin_is_stopped_on_shutdown():
    log: list[str] = []
    h = build()

    async def go() -> None:
        await h.start()
        h.use(Resource("late", log))
        await h.run("q")
        await h.stop()

    asyncio.run(go())
    assert log == ["start:late", "stop:late"]


def test_plugin_added_but_never_run_is_not_stopped():
    log: list[str] = []
    h = build()

    async def go() -> None:
        await h.start()
        h.use(Resource("late", log))  # registered but never started
        await h.stop()

    asyncio.run(go())
    assert log == []  # stop() must not stop a plugin whose start() never ran


def test_late_start_failure_keeps_already_running_plugins_up():
    log: list[str] = []
    a = Resource("a", log)
    h = build(a)

    class FailStart(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            raise RuntimeError("late boom")

    async def go() -> None:
        await h.start()  # a is up
        h.use(FailStart())
        with pytest.raises(RuntimeError, match="late boom"):
            await h.run("q")  # the late plugin's start() fails this pass
        assert h._registry.is_started(a) is True  # a must stay up
        assert h._started is True
        await h.stop()

    asyncio.run(go())
    assert log == ["start:a", "stop:a"]  # a started once, torn down once


def test_hot_swap_reuses_the_same_instance():
    log: list[str] = []
    plugin = Resource("swap", log)
    h = build()

    async def go() -> None:
        await h.start()
        h.use(plugin)
        await h.run("q")  # started
        await h.unuse(plugin)  # stopped and removed
        h.use(plugin)  # re-add the same instance
        await h.run("q")  # started again, not skipped as already-started
        await h.stop()

    asyncio.run(go())
    assert log == ["start:swap", "stop:swap", "start:swap", "stop:swap"]


def test_late_non_lifecycle_plugin_takes_effect_on_the_next_run():
    from protocols.interceptor import Interceptor

    order: list[str] = []

    class Mark(Interceptor):
        target = Model

        def before(self, ctx: Context) -> None:
            order.append("before")

    h = build()

    async def go() -> None:
        await h.start()
        h.use(Mark())  # non-lifecycle, added after start
        await h.run("q")

    asyncio.run(go())
    assert order == ["before"]  # active immediately, no start()/stop() needed


def test_interleaved_late_plugins_start_once_and_stop_in_reverse():
    log: list[str] = []
    a, b = Resource("a", log), Resource("b", log)
    h = build()

    async def go() -> None:
        await h.start()
        h.use(a)
        await h.run("q")  # a started
        h.use(b)
        await h.run("q")  # b started; a must NOT restart
        await h.stop()  # reverse registration order

    asyncio.run(go())
    assert log == ["start:a", "start:b", "stop:b", "stop:a"]


def test_restart_does_not_duplicate_in_start_subscriptions():
    from core.events import Event, ModelCallStarted

    seen: list[Event] = []

    class Subscriber(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            ctx.on(ModelCallStarted, self.record)

        def record(self, event: Event, ctx: Context) -> None:
            seen.append(event)

    h = build(Subscriber())

    async def go() -> None:
        await h.start()
        await h.stop()
        await h.start()  # re-subscribes; the first start()'s handler must be gone
        await h.run("q")

    asyncio.run(go())
    assert len(seen) == 1  # one run -> one event -> handler fires once, not twice


def test_rollback_interrupted_by_cancellation_reinitializes_on_retry():
    starts: list[str] = []

    class P1(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            starts.append("p1")

    class CancelStop(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            starts.append("cancel")

        async def stop(self) -> None:
            raise asyncio.CancelledError  # interrupts the rollback sweep

    class FailStart(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            starts.append("fail")
            raise RuntimeError("boom")

    # Order: P1, CancelStop, FailStart -> rollback runs CancelStop.stop first
    h = build(P1(), CancelStop(), FailStart())

    async def go() -> None:
        with contextlib.suppress(asyncio.CancelledError):
            await h.start()  # FailStart raises, CancelStop.stop cancels the rollback
        await h.start()  # retry

    asyncio.run(go())
    assert starts.count("p1") == 1  # never torn down, so not restarted
    assert starts.count("cancel") == 2  # un-tracked on interrupt, re-initialized
    assert starts.count("fail") == 1  # marked failed, not retried; harness recovers
    assert h._started is True


def test_failed_plugins_subscriptions_do_not_leak():
    from core.events import Event, ModelCallStarted

    seen: list[Event] = []

    class FailAfterSubscribe(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            ctx.on(ModelCallStarted, self.record)  # subscribes, then fails
            raise RuntimeError("boom")

        def record(self, event: Event, ctx: Context) -> None:
            seen.append(event)

    h = build()

    async def go() -> None:
        await h.start()
        h.use(FailAfterSubscribe())
        with pytest.raises(RuntimeError, match="boom"):
            await h.run("q")
        await h.run("q")  # a leaked handler would fire on this run's event

    asyncio.run(go())
    assert seen == []  # the failed plugin's subscription was cleaned up


def test_failed_late_plugin_is_skipped_on_later_runs():
    calls: list[int] = []

    class FailStart(Plugin, Lifecycle):
        async def start(self, ctx: Context) -> None:
            calls.append(1)
            raise RuntimeError("boom")

    h = build()

    async def go() -> None:
        await h.start()
        h.use(FailStart())
        with pytest.raises(RuntimeError, match="boom"):
            await h.run("q")  # first run: start() fails -> plugin marked failed
        await h.run("q")  # second run must succeed, the failed plugin skipped

    asyncio.run(go())
    assert calls == [1]  # start() attempted once, never retried


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

    # Teardown order (reverse of registration): Flaky first, then Res
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
