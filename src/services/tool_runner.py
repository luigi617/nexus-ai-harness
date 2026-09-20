from __future__ import annotations

from collections.abc import Iterable

from core.events import ToolCallCompleted, ToolCallDenied, ToolCallStarted
from core.message import Message
from core.permission import PermissionVerdict
from protocols.context import Context
from protocols.tool import Tool
from services.permission_gate import PermissionGate


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
            ctx.emit(ToolCallDenied(call, reason))
            return self._message(call, name, f"denied: {reason}")

        tool = self._by_name.get(name)
        if tool is None:
            reason = f"unknown tool {name!r}"
            ctx.emit(ToolCallDenied(call, reason))
            return self._message(call, name, f"error: {reason}")

        ctx.emit(ToolCallStarted(call))
        content = await ctx.invoke(tool.run, call.get("arguments", {}), ctx)
        result = self._message(call, name, content)
        ctx.emit(ToolCallCompleted(call, result))
        return result

    @staticmethod
    def _message(call: dict, name: str, content: str) -> Message:
        return Message(
            role="tool",
            content=content,
            name=name,
            tool_use_id=call.get("id"),
        )
