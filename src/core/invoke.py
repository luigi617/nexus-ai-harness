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
    return await asyncio.to_thread(fn, *args, **kwargs)
