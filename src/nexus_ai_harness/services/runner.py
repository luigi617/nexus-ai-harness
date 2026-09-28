from __future__ import annotations

from nexus_ai_harness.core.events import SessionEnded, SessionStarted
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.loop import Loop


async def run_session(ctx: Context, user_input: str) -> str:
    """Drive one session to completion and return its final text."""
    loop = ctx.get(Loop)
    if loop is None:
        raise LookupError("no loop plugin registered")

    ctx.emit(SessionStarted(ctx.session_id))
    ctx.add_message(Message(role="user", content=str(user_input)))
    result = await ctx.invoke(loop.run, ctx)
    ctx.emit(SessionEnded(ctx.session_id, result))
    return result
