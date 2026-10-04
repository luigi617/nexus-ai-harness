from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

# Set by the awaiting task when it is cancelled while its call runs in a thread.
_offload_cancelled: ContextVar[threading.Event | None] = ContextVar(
    "_offload_cancelled", default=None
)


def offload_cancelled() -> bool | None:
    """Whether the task awaiting this thread-offloaded call has been cancelled.

    A worker thread cannot be stopped from outside, so long-running sync plugin
    code (such as a model backend's retry loop) polls this to stop early.

    Returns:
        ``None`` when not running inside a call offloaded by :func:`call`,
        otherwise whether the awaiting task was cancelled.
    """
    event = _offload_cancelled.get()
    return None if event is None else event.is_set()


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
    cancelled = threading.Event()
    token = _offload_cancelled.set(cancelled)  # to_thread copies it to the worker
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except asyncio.CancelledError:
        cancelled.set()
        raise
    finally:
        _offload_cancelled.reset(token)
