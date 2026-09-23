from __future__ import annotations

import asyncio
import functools
import inspect
import threading

import pytest

from core.invoke import call


def test_async_fn_is_awaited_on_the_loop():
    loop_tid = None

    async def go():
        nonlocal loop_tid
        loop_tid = threading.get_ident()

        async def af(x):
            return x, threading.get_ident()

        return await call(af, 5)

    value, tid = asyncio.run(go())
    assert value == 5
    assert tid == loop_tid  # async ran on the event-loop thread


def test_sync_fn_is_offloaded_to_a_thread():
    async def go():
        main = threading.get_ident()

        def sf(x):
            return x, threading.get_ident()

        return main, await call(sf, 7)

    main, (value, tid) = asyncio.run(go())
    assert value == 7
    assert tid != main  # sync ran off the event-loop thread


def test_sync_fn_never_blocks_the_loop():
    # While a slow sync call runs in a thread, other coroutines keep progressing.
    async def go():
        order = []

        def slow():
            import time

            time.sleep(0.1)
            order.append("slow")

        async def quick():
            order.append("quick")

        await asyncio.gather(call(slow), quick())
        return order

    order = asyncio.run(go())
    assert order == ["quick", "slow"]  # quick finished first despite slow being started


def test_kwargs_are_forwarded_to_a_sync_fn():
    async def go():
        def sf(a, b=0, *, c=0):
            return a + b + c

        return await call(sf, 1, b=2, c=3)

    assert asyncio.run(go()) == 6


def test_kwargs_are_forwarded_to_an_async_fn():
    async def go():
        async def af(a, b=0, *, c=0):
            return a + b + c

        return await call(af, 1, b=2, c=3)

    assert asyncio.run(go()) == 6


def test_exception_from_a_sync_fn_propagates_out_of_call():
    # ToolRunner and interceptor error handling rely on call() re-raising, not wrapping.
    async def go():
        def boom():
            raise ValueError("sync boom")

        return await call(boom)

    with pytest.raises(ValueError, match="sync boom"):
        asyncio.run(go())


def test_exception_from_an_async_fn_propagates_out_of_call():
    async def go():
        async def boom():
            raise KeyError("async boom")

        return await call(boom)

    with pytest.raises(KeyError, match="async boom"):
        asyncio.run(go())


def test_partial_wrapping_an_async_fn_is_awaited_not_left_as_a_coroutine():
    # iscoroutinefunction unwraps functools.partial here, so call() awaits the partial.
    async def go():
        async def af(a, b):
            return a + b

        result = await call(functools.partial(af, 3), 4)
        assert not inspect.iscoroutine(result)
        return result

    assert asyncio.run(go()) == 7


# --- async callable objects ----------------------------------------------


def test_call_awaits_async_callable_object():
    # An object whose __call__ is async must be awaited, not returned unawaited.
    class Doubler:
        async def __call__(self, x: int) -> int:
            return x * 2

    result = asyncio.run(call(Doubler(), 21))
    assert result == 42


def test_call_treats_a_class_as_a_thread_offloaded_factory():
    # Calling a class constructs an instance (sync); its async __call__ isn't awaited.
    class Widget:
        async def __call__(self):  # instances are async-callable
            return "called"

    result = asyncio.run(call(Widget))
    assert isinstance(result, Widget)
