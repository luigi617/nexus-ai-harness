from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpawnState:
    """Per-session spawn metadata."""

    depth: int = 0
