from __future__ import annotations

from core.events import ModelCallStarted
from core.message import Message
from core.run import RunState
from protocols.context import Context
from protocols.context_manager import ContextManager
from protocols.loop import Loop
from protocols.model import Model
from protocols.router import Router
from services.invoke import invoke


class ChatLoop(Loop):
    requires = (Model,)

    async def run(self, ctx: Context) -> str:
        await ctx.apply_interventions()
        router = ctx.get(Router)
        history = ctx.history
        for cm in ctx.all(ContextManager):
            history = await invoke(ctx, cm.process, history, ctx)
        model = (
            await invoke(ctx, router.route, history, ctx)
            if router
            else ctx.get(Model)
        )
        if model is None:
            raise LookupError("no model registered")
        ctx.emit(ModelCallStarted(history))
        response = await invoke(ctx, model.complete, history, ctx)
        ctx.add_message(Message(role="assistant", content=response.text))
        ctx.state(RunState).stop_reason = "completed"
        return response.text
