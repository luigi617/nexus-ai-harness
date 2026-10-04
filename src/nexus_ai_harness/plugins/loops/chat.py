from __future__ import annotations

from nexus_ai_harness.core.errors import failure_attempts, is_model_failure
from nexus_ai_harness.core.events import ModelCallFailed, ResponseReceived
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
    """Makes a single model call with no tool execution.

    Sets ``RunState.stop_reason`` to ``"completed"``, or to ``"model_error"``
    when a model call fails (including one made by a ``Router`` or
    ``ContextManager``); a failure emits ``ModelCallFailed`` and returns a
    ``"stopped: ..."`` text rather than raising. Other errors propagate.
    """

    requires = (Model,)

    async def run(self, ctx: Context) -> str:
        await ctx.apply_interventions()
        router = ctx.get(Router)
        try:
            history = ctx.history
            for cm in ctx.all(ContextManager):
                history = await ctx.invoke(cm.process, history, ctx)
            model = (
                await ctx.invoke(router.route, history, ctx)
                if router
                else ctx.get(Model)
            )
            if model is None:
                raise LookupError("no model registered")
            response = await timed_complete(model, history, list(ctx.all(Tool)), ctx)
        except Exception as exc:  # end the run with a result, as AgenticLoop does
            if not is_model_failure(exc):
                raise
            ctx.emit(ModelCallFailed(exc, failure_attempts(exc)))
            ctx.state(RunState).stop_reason = "model_error"
            return f"stopped: model error: {exc}"
        ctx.emit(ResponseReceived(response))
        ctx.add_message(Message(role="assistant", content=response.text))
        ctx.state(RunState).stop_reason = "completed"
        return response.text
