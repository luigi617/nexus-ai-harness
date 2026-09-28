from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any


async def call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn(*args, **kwargs)`` whether it is written ``def`` or ``async def``.

    A coroutine function is awaited; a plain function is offloaded to a thread,
    so callers need not know which they were handed.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    # A callable object with async __call__ needs awaiting, not thread-offloading.
    dunder_call = getattr(fn, "__call__", None)  # noqa: B004
    if (
        not inspect.isroutine(fn)
        and not inspect.isclass(fn)
        and inspect.iscoroutinefunction(dunder_call)
    ):
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)
