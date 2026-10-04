from __future__ import annotations

import logging
import time

from nexus_ai_harness.core.events import ModelCallCompleted, ModelCallStarted
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.response import Response
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.tool import Tool

logger = logging.getLogger(__name__)


async def timed_complete(
    model: Model, history: list[Message], tools: list[Tool], ctx: Context
) -> Response:
    """Run one model completion, timing it and emitting its paired events.

    Emits :class:`ModelCallStarted` before the call and
    :class:`ModelCallCompleted` (with the response and duration) after it, both
    carrying the same ``call_id``. Loops and plugins that make their own model
    calls (summarizers, routers) use this so every call is timed and counted.

    Args:
        model: The model to call.
        history: The messages to send.
        tools: The tools to advertise to the model.
        ctx: The run context the events are emitted on.

    Returns:
        The model's response.
    """
    started = ModelCallStarted(list(history))
    ctx.emit(started)
    start = time.perf_counter()
    response = await ctx.invoke(model.complete, history, tools, ctx)
    duration = time.perf_counter() - start
    logger.debug(
        "model call completed in %.3fs (call_id=%s, usage=%s)",
        duration,
        started.call_id,
        response.usage,
    )
    ctx.emit(ModelCallCompleted(started.call_id, response, duration))
    return response
