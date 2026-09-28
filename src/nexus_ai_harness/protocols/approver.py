from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from nexus_ai_harness.core.permission import ApprovalRequest
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.plugin import Plugin


class Approver(Plugin):
    @abstractmethod
    def approve(self, request: ApprovalRequest, ctx: Context) -> bool | Awaitable[bool]:
        """Approve or deny an asked-for tool call.

        May be sync or ``async def`` — the harness adapts.
        """
