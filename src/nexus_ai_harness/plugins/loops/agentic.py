from __future__ import annotations

import asyncio

from nexus_ai_harness.core.errors import ModelAPIError
from nexus_ai_harness.core.events import (
    IterationCompleted,
    IterationStarted,
    LoopStopped,
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
from nexus_ai_harness.protocols.tool_provider import ToolProvider
from nexus_ai_harness.services.guard_chain import GuardChain
from nexus_ai_harness.services.tool_runner import ToolRunner


class AgenticLoop(Loop):
    """Calls the model and runs its tool calls until it answers without any.

    Every exit sets ``RunState.stop_reason``: ``"completed"``, ``"interrupted"``,
    ``"guard: <reason>"``, or ``"model_error"``. A model call that still fails
    after the backend's own retries ends the run like a guard stop: it emits
    ``ModelCallFailed`` and ``LoopStopped`` and returns a ``"stopped: ..."`` text,
    so the caller gets a ``RunResult`` instead of an exception.
    """

    requires = (Model,)

    async def run(self, ctx: Context) -> str:
        router = ctx.get(Router)
        available = list(ctx.all(Tool))
        for provider in ctx.all(ToolProvider):  # tools discovered at runtime
            provided = await ctx.invoke(provider.provide_tools, ctx)
            available.extend(provided)
        tools = ToolRunner(available)
        guards = GuardChain()

        i = 0
        while True:
            if ctx.interrupted:
                ctx.state(RunState).stop_reason = "interrupted"
                ctx.emit(LoopStopped("interrupted"))
                return "stopped: interrupted"

            await ctx.apply_interventions()  # e.g. injected messages steer this turn

            ctx.emit(IterationStarted(i))
            decision = guards.check(ctx)
            if decision.stop:
                reason = f"guard: {decision.reason}"
                ctx.state(RunState).stop_reason = reason
                ctx.emit(IterationCompleted(i))
                ctx.emit(LoopStopped(reason))
                return f"stopped: {decision.reason}"

            history = ctx.history
            for cm in ctx.all(ContextManager):  # middleware chain
                history = await ctx.invoke(cm.process, history, ctx)
            model = (
                await ctx.invoke(router.route, history, ctx)
                if router
                else ctx.get(Model)
            )
            if model is None:
                raise LookupError("no model registered")
            ctx.emit(ModelCallStarted(list(history)))
            # Hand the model exactly the tools the runner can dispatch.
            try:
                response = await ctx.invoke(model.complete, history, available, ctx)
            except Exception as exc:  # like a tool error, a model error must not crash
                ctx.emit(ModelCallFailed(exc, _attempts(exc)))
                ctx.state(RunState).stop_reason = "model_error"
                ctx.emit(IterationCompleted(i))
                ctx.emit(LoopStopped("model_error"))
                return f"stopped: model error: {exc}"
            ctx.emit(ResponseReceived(response))
            ctx.add_message(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )

            if not response.tool_calls:  # natural exit — model is done
                # Re-check guards: a final response over budget isn't a clean exit.
                post = guards.check(ctx)
                if post.stop:
                    reason = f"guard: {post.reason}"
                    ctx.state(RunState).stop_reason = reason
                    ctx.emit(IterationCompleted(i))
                    ctx.emit(LoopStopped(reason))
                    return f"stopped: {post.reason}"
                ctx.emit(IterationCompleted(i))
                ctx.state(RunState).stop_reason = "completed"
                ctx.emit(LoopStopped("completed"))
                return response.text

            results = await asyncio.gather(
                *(tools.run(call, ctx) for call in response.tool_calls)
            )
            for message in results:
                ctx.add_message(message)
            ctx.emit(IterationCompleted(i))
            i += 1


def _attempts(exc: Exception) -> int | None:
    return exc.attempts if isinstance(exc, ModelAPIError) else None
