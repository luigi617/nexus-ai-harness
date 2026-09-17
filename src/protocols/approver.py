from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable
from typing import ClassVar, Protocol, runtime_checkable

from core.permission import ApprovalRequest
from protocols.context import Context
from protocols.plugin import Plugin


@runtime_checkable
class Approver(Plugin, Protocol):
    kind: ClassVar[str] = "approver"

    @abstractmethod
    def approve(self, request: ApprovalRequest, ctx: Context) -> bool | Awaitable[bool]:
        """Approve or deny an asked-for tool call.
        May be sync or ``async def`` — the harness adapts."""
