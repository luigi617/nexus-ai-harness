from __future__ import annotations

from typing import ClassVar

from protocols.context import Context
from protocols.memory import MemoryStore
from protocols.tool import Tool


class Remember(Tool):
    """Save a fact to long-term memory, or update an existing one.

    Requires a MemoryStore plugin (kind="memory") to be registered.
    """

    name = "remember"
    description = (
        "Save a fact to long-term memory so it survives across sessions. "
        "Omit 'id' to store a new fact; pass the 'id' of an existing memory "
        "(from recall) to overwrite it. Use for stable, reusable facts "
        "(preferences, decisions, project context) — not transient chatter."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The fact to remember."},
            "id": {
                "type": "string",
                "description": (
                    "Id of an existing memory to overwrite. Omit to create a new one."
                ),
            },
        },
        "required": ["text"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        store: MemoryStore | None = ctx.get(MemoryStore)
        if store is None:
            return "error: no memory store configured"
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: nothing to remember (empty text)"
        id = arguments.get("id") or None
        updating = id is not None and store.get(id) is not None
        item = store.save(text, id)
        verb = "updated" if updating else "remembered"
        return f"{verb} ({item.id})"
