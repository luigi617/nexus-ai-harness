from __future__ import annotations

from core.events import SessionEnded, SessionStarted
from core.message import Message
from protocols.context import Context
from protocols.loop import Loop


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
