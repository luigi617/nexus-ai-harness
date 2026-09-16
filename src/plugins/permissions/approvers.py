from __future__ import annotations

import asyncio

from core.permission import ApprovalRequest
from protocols.approver import Approver
from protocols.context import Context


class AutoApprove(Approver):
    """Approve every ask — for non-interactive/automated runs."""

    def approve(self, request: ApprovalRequest, ctx: Context) -> bool:
        return True


class ConsoleApprover(Approver):
    """Prompt on the terminal: y (once) / n (deny) / a (always allow this tool)."""

    def __init__(self) -> None:
        self._always: set[str] = set()
        self._lock = asyncio.Lock()

    async def approve(self, request: ApprovalRequest, ctx: Context) -> bool:
        name = request.call.get("name", "")
        if name in self._always:
            return True

        async with self._lock:  # one prompt at a time
            prefix = f"[{request.origin}] " if request.origin else ""
            args = request.call.get("arguments", {})
            answer = (
                (
                    await asyncio.to_thread(
                        input, f"{prefix}{request.reason} {name}({args}) [y/n/a]: "
                    )
                )
                .strip()
                .lower()
            )

        if answer == "a":
            self._always.add(name)
            return True
        return answer == "y"
