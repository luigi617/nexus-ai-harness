from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from core.invoke import invoke
from protocols.context import Context
from protocols.spawner import Spawner
from protocols.tool import Tool


class Subagent(Tool):
    """A general delegation tool: runs a self-contained task in an isolated
    child agent and returns only the distilled result.
    The model decides when to delegate and what the task is.

    By default the child inherits all of the parent's plugins. Pass ``plugins``
    to restrict it to exactly that set (a subset of what the parent has).
    """

    name = "subagent"
    description = (
        "Delegate a self-contained sub-task to an isolated agent. It runs with "
        "its own fresh context and returns only a concise result, so the "
        "intermediate steps never clutter this conversation. Use for bounded, "
        "well-scoped work (searching, multi-step lookups, analysis) whose "
        "internal steps you don't need to see."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The self-contained task for the subagent to complete.",
            }
        },
        "required": ["task"],
    }

    def __init__(self, plugins: Iterable[object] | None = None) -> None:
        # None → inherit all parent plugins; a list → the child sees only these.
        self._plugins = None if plugins is None else list(plugins)

    async def run(self, arguments: dict, ctx: Context) -> str:
        spawner = ctx.get(Spawner)
        if spawner is None:
            return "error: no spawner configured"
        task = str(arguments.get("task", "")).strip()
        if not task:
            return "error: no task provided"

        child = ctx.fork(self._plugins)
        return await invoke(spawner.run, child, task)
