from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Response:
    """Response of the model for one completion."""

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    # token usage
    usage: dict = field(default_factory=dict)
    cost: float = 0.0
