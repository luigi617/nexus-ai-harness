from __future__ import annotations

import asyncio
import logging
import time

from nexus_ai_harness.core.events import (
    IterationCompleted,
    IterationStarted,
    LoopStopped,
    ModelCallCompleted,
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

logger = logging.getLogger(__name__)


class AgenticLoop(Loop):
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
            started = ModelCallStarted(list(history))
            ctx.emit(started)
            start = time.perf_counter()
            # Hand the model exactly the tools the runner can dispatch.
            response = await ctx.invoke(model.complete, history, available, ctx)
            duration = time.perf_counter() - start
            logger.debug(
                "model call completed in %.3fs (call_id=%s, usage=%s)",
                duration,
                started.call_id,
                response.usage,
            )
            ctx.emit(ModelCallCompleted(started.call_id, response, duration))
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
