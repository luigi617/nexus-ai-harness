from __future__ import annotations

from nexus_ai_harness.core.errors import ModelAPIError
from nexus_ai_harness.core.events import (
    ModelCallFailed,
    ModelCallStarted,
    ResponseReceived,
)
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.run import RunState
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.context_manager import ContextManager
from nexus_ai_harness.protocols.loop import Loop
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.router import Router
from nexus_ai_harness.protocols.tool import Tool


class ChatLoop(Loop):
    """Makes a single model call with no tool execution.

    Sets ``RunState.stop_reason`` to ``"completed"``, or to ``"model_error"``
    when the call fails; a failure emits ``ModelCallFailed`` and returns a
    ``"stopped: ..."`` text rather than raising.
    """

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
        ctx.emit(ModelCallStarted(list(history)))
        try:
            response = await ctx.invoke(
                model.complete, history, list(ctx.all(Tool)), ctx
            )
        except Exception as exc:  # end the run with a result, as AgenticLoop does
            attempts = exc.attempts if isinstance(exc, ModelAPIError) else None
            ctx.emit(ModelCallFailed(exc, attempts))
            ctx.state(RunState).stop_reason = "model_error"
            return f"stopped: model error: {exc}"
        ctx.emit(ResponseReceived(response))
        ctx.add_message(Message(role="assistant", content=response.text))
        ctx.state(RunState).stop_reason = "completed"
        return response.text
