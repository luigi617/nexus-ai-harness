from __future__ import annotations

import logging

from nexus_ai_harness.core.events import SessionEnded, SessionStarted
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.run import RunState
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.loop import Loop

logger = logging.getLogger(__name__)


async def run_session(ctx: Context, user_input: str) -> str:
    """Drive one session to completion and return its final text."""
    loop = ctx.get(Loop)
    if loop is None:
        raise LookupError("no loop plugin registered")

    logger.debug("session %s started", ctx.session_id)
    ctx.emit(SessionStarted(ctx.session_id))
    ctx.add_message(Message(role="user", content=str(user_input)))
    result = await ctx.invoke(loop.run, ctx)
    logger.debug(
        "session %s ended (stop_reason=%s)",
        ctx.session_id,
        ctx.state(RunState).stop_reason,
    )
    ctx.emit(SessionEnded(ctx.session_id, result))
    return result
