from __future__ import annotations

from typing import ClassVar

from core.invoke import invoke
from core.message import Message
from core.run import RunState
from protocols.context import Context
from protocols.context_manager import ContextManager
from protocols.loop import Loop
from protocols.model import Model
from protocols.plugin import Plugin
from protocols.router import Router


class ChatLoop(Loop):
    # A single completion needs a Model (a Router may supply it); nothing else
    # is required.
    requires: ClassVar[tuple[type[Plugin], ...]] = (Model,)

    async def run(self, ctx: Context) -> str:
        await ctx.apply_interventions()
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
