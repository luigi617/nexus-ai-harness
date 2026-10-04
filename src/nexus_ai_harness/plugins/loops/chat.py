from __future__ import annotations

from nexus_ai_harness.core.events import ResponseReceived
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.run import RunState
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.context_manager import ContextManager
from nexus_ai_harness.protocols.loop import Loop
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.router import Router
from nexus_ai_harness.protocols.tool import Tool
from nexus_ai_harness.services.model_call import timed_complete


class ChatLoop(Loop):
    requires = (Model,)

    async def run(self, ctx: Context) -> str:
        await ctx.apply_interventions()
        router = ctx.get(Router)
        history = ctx.history
        for cm in ctx.all(ContextManager):
            history = await ctx.invoke(cm.process, history, ctx)
        model = (
            await ctx.invoke(router.route, history, ctx) if router else ctx.get(Model)
        )
        if model is None:
            raise LookupError("no model registered")
        response = await timed_complete(model, history, list(ctx.all(Tool)), ctx)
        ctx.emit(ResponseReceived(response))
        ctx.add_message(Message(role="assistant", content=response.text))
        ctx.state(RunState).stop_reason = "completed"
        return response.text
