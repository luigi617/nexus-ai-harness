from __future__ import annotations

from enum import Enum


class Phase(Enum):
    """When an interceptor runs relative to the invocation it wraps."""

    BEFORE = "before"
    AFTER = "after"
