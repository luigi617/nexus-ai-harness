from __future__ import annotations

import asyncio
import threading

from services.invoke import _invoke


def test_async_fn_is_awaited_on_the_loop():
    loop_tid = None

    async def go():
        nonlocal loop_tid
        loop_tid = threading.get_ident()

        async def af(x):
            return x, threading.get_ident()

        return await _invoke(af, 5)

    value, tid = asyncio.run(go())
    assert value == 5
    assert tid == loop_tid  # async ran on the event-loop thread


def test_sync_fn_is_offloaded_to_a_thread():
    async def go():
        main = threading.get_ident()

        def sf(x):
            return x, threading.get_ident()

        return main, await _invoke(sf, 7)

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

        await asyncio.gather(_invoke(slow), quick())
        return order

    order = asyncio.run(go())
    assert order == ["quick", "slow"]  # quick finished first despite slow being started
