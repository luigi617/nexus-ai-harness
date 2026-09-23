from __future__ import annotations

from core.events import ApprovalRequested
from core.permission import ApprovalRequest, PermissionDecision, PermissionVerdict
from core.spawn import SpawnState
from protocols.approver import Approver
from protocols.context import Context
from protocols.permission import Permission


class PermissionGate:
    """Combines all permission plugins and resolves an 'ask' via the approver."""

    async def decide(self, call: dict, ctx: Context) -> PermissionDecision:
        decision = self._combine(call, ctx)
        if decision.verdict != PermissionVerdict.ASK:
            return decision

        ctx.emit(ApprovalRequested(call, decision.reason))
        approver = ctx.get(Approver)
        request = ApprovalRequest(
            call=call,
            reason=decision.reason,
            origin=self._origin(ctx),
        )
        if approver is not None:
            approved = await ctx.invoke(approver.approve, request, ctx)
            if approved:
                return PermissionDecision.allow()
        # Don't echo the ASK prompt as the denial reason; report a rejection instead.
        return PermissionDecision.deny("not approved")

    @staticmethod
    def _origin(ctx: Context) -> str:
        """Harness-computed label of who is asking.

        Empty for the main agent; for a subagent, its depth and task. Kept here
        so approver authors never touch spawn internals.
        """
        depth = ctx.state(SpawnState).depth
        if depth == 0:
            return ""
        task = next((m.content for m in ctx.history if m.role == "user"), "")
        if len(task) > 40:
            task = task[:40] + "…"
        return f"subagent depth={depth} | task: {task!r}"

    @staticmethod
    def _combine(call: dict, ctx: Context) -> PermissionDecision:
        # Precedence: any deny wins; else any ask; else allow.
        result = PermissionDecision.allow()
        for permission in ctx.all(Permission):
            decision = permission.check(call, ctx)
            if decision.verdict == PermissionVerdict.DENY:
                return decision
            if (
                decision.verdict == PermissionVerdict.ASK
                and result.verdict == PermissionVerdict.ALLOW
            ):
                result = decision
        return result
