from __future__ import annotations

from core.invoke import invoke
from core.message import Message
from protocols.context import ContextManager
from protocols.loop import Loop
from protocols.mediator import Context
from protocols.provider import Provider


class ChatLoop(Loop):
    async def run(self, ctx: Context) -> str:
        provider = ctx.get(Provider)
        if provider is None:
            raise LookupError("no provider plugin registered")
        history = ctx.history
        for cm in ctx.all(ContextManager):
            history = await invoke(cm.process, history, ctx)
        response = await invoke(provider.complete, history, ctx)
        ctx.add_message(Message(role="assistant", content=response.text))
        return response.text
