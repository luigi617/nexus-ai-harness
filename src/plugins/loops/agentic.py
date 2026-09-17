from __future__ import annotations

import asyncio

from core.events import (
    IterationCompleted,
    IterationStarted,
    LoopStopped,
    ModelCallStarted,
    ResponseReceived,
)
from core.invoke import invoke
from core.message import Message
from core.run import RunState
from protocols.context import Context
from protocols.context_manager import ContextManager
from protocols.loop import Loop
from protocols.model import Model
from protocols.router import Router
from protocols.tool import Tool
from services.guard_chain import GuardChain
from services.tool_runner import ToolRunner


class AgenticLoop(Loop):
    requires = (Model,)

    async def run(self, ctx: Context) -> str:
        router = ctx.get(Router)
        tools = ToolRunner(ctx.all(Tool))
        guards = GuardChain()

        i = 0
        while True:
            if ctx.interrupted:  # control
                ctx.state(RunState).stop_reason = "interrupted"
                ctx.emit(LoopStopped("interrupted"))
                return "stopped: interrupted"

            await ctx.apply_interventions()  # e.g. injected messages steer this turn

            ctx.emit(IterationStarted(i))
            decision = guards.check(ctx)
            if decision.stop:
                reason = f"guard: {decision.reason}"
                ctx.state(RunState).stop_reason = reason
                ctx.emit(LoopStopped(reason))
                return f"stopped: {decision.reason}"

            history = ctx.history
            for cm in ctx.all(ContextManager):  # middleware chain
                history = await invoke(cm.process, history, ctx)
            model = (
                await invoke(router.route, history, ctx) if router else ctx.get(Model)
            )
            if model is None:
                raise LookupError("no model registered")
            ctx.emit(ModelCallStarted(history))
            response = await invoke(model.complete, history, ctx)
            ctx.emit(ResponseReceived(response))
            ctx.add_message(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )

            if not response.tool_calls:  # natural exit — model is done
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
