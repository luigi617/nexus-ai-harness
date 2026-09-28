from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Edge:
    source: str
    target: str
    metadata: dict = field(default_factory=dict)
