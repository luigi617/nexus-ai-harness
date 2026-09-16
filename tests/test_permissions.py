from __future__ import annotations

import asyncio

from tests.conftest import make_ctx

from core.message import Message
from core.permission import PermissionVerdict
from core.spawn import SpawnState
from plugins.permissions import AllowList, AskUnless, AutoApprove, DenyList
from services.permission_gate import PermissionGate


def decide(call, *plugins):
    return asyncio.run(PermissionGate().decide(call, make_ctx(*plugins)))


def test_allowlist_allows_listed_and_denies_rest():
    assert decide({"name": "ok"}, AllowList(["ok"])).verdict == PermissionVerdict.ALLOW
    assert decide({"name": "no"}, AllowList(["ok"])).verdict == PermissionVerdict.DENY


def test_denylist_blocks_listed():
    assert decide({"name": "rm"}, DenyList(["rm"])).verdict == PermissionVerdict.DENY
    assert decide({"name": "ls"}, DenyList(["rm"])).verdict == PermissionVerdict.ALLOW


def test_deny_beats_ask_beats_allow():
    # AllowList denies "x"; AskUnless would ask. Deny must win.
    d = decide({"name": "x"}, AllowList([]), AskUnless([]))
    assert d.verdict == PermissionVerdict.DENY


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
