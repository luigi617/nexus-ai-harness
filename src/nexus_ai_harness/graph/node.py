from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Node:
    id: str
    data: Any = None
