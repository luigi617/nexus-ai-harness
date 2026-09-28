from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunState:
    """Run-level outcome metadata."""

    stop_reason: str = ""
