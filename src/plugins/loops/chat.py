from __future__ import annotations

from core.invoke import invoke
from core.message import Message
from core.run import RunState
from protocols.context import ContextManager
from protocols.loop import Loop
from protocols.mediator import Context
from protocols.model import Model
from protocols.router import Router
from services.intervene import apply_interventions


class ChatLoop(Loop):
    async def run(self, ctx: Context) -> str:
        await apply_interventions(ctx)
        router = ctx.get(Router)
        history = ctx.history
        for cm in ctx.all(ContextManager):
            history = await invoke(cm.process, history, ctx)
        model = await invoke(router.route, history, ctx) if router else ctx.get(Model)
        if model is None:
            raise LookupError("no model registered")
        response = await invoke(model.complete, history, ctx)
        ctx.add_message(Message(role="assistant", content=response.text))
        ctx.state(RunState).stop_reason = "completed"
        return response.text
