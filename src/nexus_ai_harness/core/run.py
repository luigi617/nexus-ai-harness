from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunState:
    """Run-level outcome metadata."""

    stop_reason: str = ""


@dataclass
class CostState:
    """Spend and token totals accumulated over a session by ``CostCounter``.

    Attributes:
        total: Running cost in USD.
        input_tokens: Running count of prompt tokens sent to the model.
        output_tokens: Running count of completion tokens the model produced.
    """

    total: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
