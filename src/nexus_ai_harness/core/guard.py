from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardDecision:
    stop: bool
    reason: str = ""

    @classmethod
    def proceed(cls) -> GuardDecision:
        return cls(stop=False)

    @classmethod
    def halt(cls, reason: str = "") -> GuardDecision:
        return cls(stop=True, reason=reason)
