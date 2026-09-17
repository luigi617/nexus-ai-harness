from __future__ import annotations

from typing import ClassVar

from protocols.context import Context
from protocols.memory import MemoryStore
from protocols.tool import Tool


class Forget(Tool):
    """Delete a memory by id.

    Requires a MemoryStore plugin (kind="memory") to be registered.
    """

    name = "forget"
    description = (
        "Permanently delete a memory by its id (obtain the id from recall). "
        "Use when a saved fact is wrong or no longer relevant."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Id of the memory to delete."},
        },
        "required": ["id"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        store: MemoryStore | None = ctx.get(MemoryStore)
        if store is None:
            return "error: no memory store configured"
        id = str(arguments.get("id", "")).strip()
        if not id:
            return "error: no id provided"
        return f"forgot ({id})" if store.delete(id) else f"no memory with id {id!r}"
