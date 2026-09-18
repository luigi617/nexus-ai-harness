from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from core.phase import Phase
from protocols.context import Context
from protocols.interceptor import Interceptor
from protocols.plugin import Plugin


async def _invoke(fn: Callable[..., Any], *args: Any) -> Any:
    """Call a plugin method that may be written as either ``def`` or ``async def``."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    return await asyncio.to_thread(fn, *args)


@dataclass(frozen=True)
class InterceptorBinding:
    """An interceptor plus the target type and phase it fires around.

    Registered in place of the bare interceptor so the phase and target type
    travel with it and survive a ``ctx.fork``.

    Attributes:
        target: The plugin type whose invocation the interceptor wraps; an
            invoked plugin matches when it is an instance of ``target``, so a
            concrete class binds only to that class, a protocol to any
            implementer.
        phase: ``Phase.BEFORE`` to run ahead of the invocation, ``Phase.AFTER``
            after it.
        interceptor: The interceptor to run.
    """

    target: type[Plugin]
    phase: Phase
    interceptor: Interceptor
    kind: ClassVar[str] = "interceptor-binding"
    requires: ClassVar[tuple[type, ...]] = ()


async def invoke(ctx: Context, fn: Callable[..., Any], *args: Any) -> Any:
    """Invoke a plugin method ``fn`` with any interceptors bound to it around it."""
    plugin = getattr(fn, "__self__", None)
    bound = [
        b
        for b in ctx.all(InterceptorBinding)
        if plugin is not None and isinstance(plugin, b.target)
    ]
    for binding in bound:
        if binding.phase is Phase.BEFORE:
            await _invoke(binding.interceptor.run, ctx)
    try:
        return await _invoke(fn, *args)
    finally:
        for binding in bound:
            if binding.phase is Phase.AFTER:
                await _invoke(binding.interceptor.run, ctx)
