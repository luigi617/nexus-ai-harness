from __future__ import annotations

import logging
import time
from collections.abc import Iterable

from nexus_ai_harness.core.events import (
    ToolCallCompleted,
    ToolCallDenied,
    ToolCallStarted,
)
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.permission import PermissionVerdict
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.tool import Tool
from nexus_ai_harness.services.permission_gate import PermissionGate

logger = logging.getLogger(__name__)


class ToolRunner:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._by_name: dict[str, Tool] = {t.name: t for t in tools}
        self._permission_gate = PermissionGate()

    def add(self, tool: Tool) -> None:
        self._by_name[tool.name] = tool

    async def run(self, call: dict, ctx: Context) -> Message:
        name = call.get("name", "")

        decision = await self._permission_gate.decide(call, ctx)
        if decision.verdict == PermissionVerdict.DENY:
            reason = decision.reason or "not permitted"
            logger.info("tool call %r denied: %s", name, reason)
            ctx.emit(ToolCallDenied(call, reason))
            return self._message(call, name, f"denied: {reason}")

        tool = self._by_name.get(name)
        if tool is None:
            reason = f"unknown tool {name!r}"
            logger.info("tool call denied: %s", reason)
            ctx.emit(ToolCallDenied(call, reason))
            return self._message(call, name, f"error: {reason}")

        started = ToolCallStarted(call)
        logger.debug("tool call %r started (call_id=%s)", name, started.call_id)
        ctx.emit(started)
        error: Exception | None = None
        start = time.perf_counter()
        try:
            content = await ctx.invoke(tool.run, call.get("arguments", {}), ctx)
        except Exception as exc:
            error = exc
            # The model only sees the message; the log keeps the traceback.
            logger.warning(
                "tool %r raised (call_id=%s)", name, started.call_id, exc_info=exc
            )
            content = f"error: {exc}"
        duration = time.perf_counter() - start
        logger.debug(
            "tool call %r completed in %.3fs (call_id=%s)",
            name,
            duration,
            started.call_id,
        )
        result = self._message(call, name, content)
        ctx.emit(
            ToolCallCompleted(
                call,
                result,
                call_id=started.call_id,
                duration=duration,
                error=error,
            )
        )
        return result

    @staticmethod
    def _message(call: dict, name: str, content: str) -> Message:
        return Message(
            role="tool",
            content=content,
            name=name,
            tool_use_id=call.get("id"),
        )
