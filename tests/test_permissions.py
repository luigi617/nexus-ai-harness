from __future__ import annotations

import asyncio

from core.events import ApprovalRequested, Event
from core.message import Message
from core.permission import ApprovalRequest, PermissionVerdict
from core.spawn import SpawnState
from plugins.permissions import AllowList, AskUnless, AutoApprove, DenyList
from protocols.approver import Approver
from protocols.hook import Hook
from services.permission_gate import PermissionGate
from tests.conftest import make_ctx


class DenyingApprover(Approver):
    """An approver that explicitly rejects every ask; records what it saw."""

    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    def approve(self, request: ApprovalRequest, ctx) -> bool:
        self.requests.append(request)
        return False


class EventRecorder(Hook):
    """A hook that keeps every emitted event object for assertions."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def on(self, event: Event, ctx) -> None:
        self.events.append(event)


def decide(call, *plugins):
    return asyncio.run(PermissionGate().decide(call, make_ctx(*plugins)))


def test_allowlist_allows_listed_and_denies_rest():
    assert decide({"name": "ok"}, AllowList(["ok"])).verdict == PermissionVerdict.ALLOW
    assert decide({"name": "no"}, AllowList(["ok"])).verdict == PermissionVerdict.DENY


def test_denylist_blocks_listed():
    assert decide({"name": "rm"}, DenyList(["rm"])).verdict == PermissionVerdict.DENY
    assert decide({"name": "ls"}, DenyList(["rm"])).verdict == PermissionVerdict.ALLOW


def combine(call, *plugins):
    # _combine holds precedence logic before an approver resolves ASK; assert directly.
    return PermissionGate._combine(call, make_ctx(*plugins))


def test_deny_beats_ask_beats_allow():
    # deny + ask -> deny (a deny short-circuits everything else)
    assert (
        combine({"name": "x"}, DenyList(["x"]), AskUnless([])).verdict
        == PermissionVerdict.DENY
    )
    # ask + allow -> ask (an ask overrides an allow when no deny is present)
    assert (
        combine({"name": "x"}, AskUnless([]), AllowList(["x"])).verdict
        == PermissionVerdict.ASK
    )
    # allow only -> allow
    assert combine({"name": "x"}, AllowList(["x"])).verdict == PermissionVerdict.ALLOW


def test_ask_resolved_by_approver():
    allow = decide({"name": "x"}, AskUnless([]), AutoApprove())
    assert allow.verdict == PermissionVerdict.ALLOW


def test_ask_denied_when_no_approver():
    d = decide({"name": "x"}, AskUnless([]))
    assert d.verdict == PermissionVerdict.DENY


def test_provenance_origin_reflects_subagent_depth():
    ctx = make_ctx(AskUnless([]))
    ctx.state(SpawnState).depth = 2
    ctx.add_message(Message(role="user", content="do the thing"))
    origin = PermissionGate._origin(ctx)
    assert "depth=2" in origin and "do the thing" in origin


def test_provenance_empty_for_main_agent():
    assert PermissionGate._origin(make_ctx()) == ""


def test_ask_denied_by_approver_returns_deny():
    # A rejecting approver must turn ASK into DENY and must have been consulted.
    approver = DenyingApprover()
    ctx = make_ctx(AskUnless([]), approver)
    decision = asyncio.run(PermissionGate().decide({"name": "x"}, ctx))
    assert decision.verdict == PermissionVerdict.DENY
    assert len(approver.requests) == 1
    assert approver.requests[0].call == {"name": "x"}


def test_ask_denied_by_approver_reports_rejection_reason():
    # A denial reports a rejection message ("not approved"), not the echoed ASK prompt.
    ctx = make_ctx(AskUnless([]), DenyingApprover())
    decision = asyncio.run(PermissionGate().decide({"name": "x"}, ctx))
    assert decision.verdict == PermissionVerdict.DENY
    assert decision.reason == "not approved"
    assert decision.reason != "Allow tool 'x'?"


def test_ask_emits_approval_requested_event_with_call_and_reason():
    rec = EventRecorder()
    ctx = make_ctx(AskUnless([]), AutoApprove(), rec)
    asyncio.run(PermissionGate().decide({"name": "x"}, ctx))
    requested = [e for e in rec.events if isinstance(e, ApprovalRequested)]
    assert len(requested) == 1
    assert requested[0].call == {"name": "x"}
    assert requested[0].reason == "Allow tool 'x'?"


def test_ask_emits_approval_requested_even_without_approver():
    # ApprovalRequested fires before the approver check — even when none is registered.
    rec = EventRecorder()
    ctx = make_ctx(AskUnless([]), rec)
    decision = asyncio.run(PermissionGate().decide({"name": "x"}, ctx))
    assert decision.verdict == PermissionVerdict.DENY
    requested = [e for e in rec.events if isinstance(e, ApprovalRequested)]
    assert len(requested) == 1
    assert requested[0].call == {"name": "x"}


def test_no_permission_plugins_defaults_to_allow():
    # Default-open posture: with zero Permission plugins registered, _combine()
    # returns allow(), so ToolRunner would execute any tool by default. Locking
    # this in forces a conscious decision if it should ever become deny-by-default.
    assert decide({"name": "anything"}).verdict == PermissionVerdict.ALLOW


def test_origin_truncates_long_task_with_ellipsis():
    ctx = make_ctx()
    ctx.state(SpawnState).depth = 1
    long_task = "x" * 45  # exceeds the 40-char cap
    ctx.add_message(Message(role="user", content=long_task))
    origin = PermissionGate._origin(ctx)
    assert "depth=1" in origin
    assert "…" in origin
    assert ("x" * 40) in origin  # first 40 chars kept
    assert long_task not in origin  # full untruncated string not present


def test_origin_subagent_with_empty_history_has_empty_task():
    ctx = make_ctx()
    ctx.state(SpawnState).depth = 3  # subagent, but no user message in history
    origin = PermissionGate._origin(ctx)
    assert "depth=3" in origin
    assert "task: ''" in origin  # empty-string default rendered


# --- denial reason on approver rejection ---------------------------------


def test_denied_ask_reports_a_rejection_reason_not_the_prompt():
    # A denied ask must report a rejection message, not echo the ask prompt.
    ctx = make_ctx(AskUnless(trusted=[]), DenyingApprover())
    decision = asyncio.run(PermissionGate().decide({"name": "foo"}, ctx))
    assert decision.verdict == PermissionVerdict.DENY
    assert decision.reason != "Allow tool 'foo'?"
