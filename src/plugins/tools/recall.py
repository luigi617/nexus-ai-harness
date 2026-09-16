from __future__ import annotations

from typing import ClassVar

from protocols.context import Context
from protocols.memory import MemoryStore
from protocols.tool import Tool


class Recall(Tool):
    """Search long-term memory for facts relevant to a query.

    Requires a MemoryStore plugin (kind="memory") to be registered.
    """

    name = "recall"
    description = (
        "Search long-term memory for previously saved facts. Call this when "
        "prior context (preferences, decisions, project details) would help. "
        "Each result shows its id, which can be passed to remember (to update) "
        "or forget (to delete)."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords describing what to look for.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return (default 5).",
            },
        },
        "required": ["query"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        store: MemoryStore | None = ctx.get(MemoryStore)
        if store is None:
            return "error: no memory store configured"
        query = str(arguments.get("query", "")).strip()
        limit = arguments.get("limit", 5)
        limit = limit if isinstance(limit, int) and limit > 0 else 5

        results = store.search(query, limit)
        if not results:
            return "no relevant memories found"
        return "\n".join(f"- ({item.id}) {item.text}" for item in results)
