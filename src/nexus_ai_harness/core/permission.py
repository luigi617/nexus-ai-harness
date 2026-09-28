from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PermissionVerdict(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"  # defer to a human/approver


@dataclass
class PermissionDecision:
    verdict: PermissionVerdict
    reason: str = ""

    @classmethod
    def allow(cls) -> PermissionDecision:
        return cls(PermissionVerdict.ALLOW)

    @classmethod
    def deny(cls, reason: str = "") -> PermissionDecision:
        return cls(PermissionVerdict.DENY, reason)

    @classmethod
    def ask(cls, reason: str = "") -> PermissionDecision:
        return cls(PermissionVerdict.ASK, reason)


@dataclass
class ApprovalRequest:
    """An asked-for tool call presented to the approver."""

    call: dict
    reason: str | None
    origin: str = ""
