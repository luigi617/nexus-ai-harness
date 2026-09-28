from __future__ import annotations

from dataclasses import dataclass, field

from nexus_ai_harness.core.ids import new_id


@dataclass
class Message:
    role: str
    content: str = ""
    id: str = field(default_factory=lambda: new_id("msg"))
    # assistant only: the tool calls the model requested this turn.
    tool_calls: list[dict] = field(default_factory=list)
    # tool only: which tool call this message answers, and the tool's name.
    tool_use_id: str | None = None
    name: str | None = None
