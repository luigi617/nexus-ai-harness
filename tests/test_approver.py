from __future__ import annotations

import asyncio
import builtins
import time

import pytest

from core.permission import ApprovalRequest
from plugins.permissions import AutoApprove, ConsoleApprover
from tests.conftest import make_ctx


def req(name="tool", origin=""):
    return ApprovalRequest(
        call={"name": name, "arguments": {}}, reason="Allow?", origin=origin
    )


def test_auto_approve_always_true():
    assert AutoApprove().approve(req(), make_ctx()) is True


def test_console_yes_no(monkeypatch):
    ctx = make_ctx()
    monkeypatch.setattr(builtins, "input", lambda _p: "y")
    assert asyncio.run(ConsoleApprover().approve(req(), ctx)) is True
    monkeypatch.setattr(builtins, "input", lambda _p: "n")
    assert asyncio.run(ConsoleApprover().approve(req(), ctx)) is False


def test_console_always_skips_second_prompt(monkeypatch):
    ctx = make_ctx()
    ap = ConsoleApprover()
    calls = []
    monkeypatch.setattr(builtins, "input", lambda p: (calls.append(p), "a")[1])
    assert asyncio.run(ap.approve(req("writefile"), ctx)) is True
    # second time: "always" fast-path, no prompt
    assert asyncio.run(ap.approve(req("writefile"), ctx)) is True
    assert len(calls) == 1


def test_console_shows_provenance_in_prompt(monkeypatch):
    ctx = make_ctx()
    seen = []
    monkeypatch.setattr(builtins, "input", lambda p: (seen.append(p), "y")[1])
    request = req(origin="subagent depth=2 | task: 'audit'")
    asyncio.run(ConsoleApprover().approve(request, ctx))
    assert "subagent depth=2" in seen[0]


def test_console_serializes_concurrent_prompts(monkeypatch):
    ctx = make_ctx()
    ap = ConsoleApprover()
    live = 0
    max_live = 0

    def blocking(_p):
        nonlocal live, max_live
        live += 1
        max_live = max(max_live, live)
        time.sleep(0.05)
        live -= 1
        return "y"

    monkeypatch.setattr(builtins, "input", blocking)

    async def go():
        return await asyncio.gather(*(ap.approve(req(f"t{i}"), ctx) for i in range(3)))

    assert asyncio.run(go()) == [True, True, True]
    assert max_live == 1  # lock serialized the prompts


def test_console_concurrent_same_tool_prompts_once(monkeypatch):
    # Concurrent asks for one tool must prompt once: 2nd hits the always fast-path.
    ctx = make_ctx()
    ap = ConsoleApprover()
    calls = []

    def blocking(p):
        calls.append(p)
        time.sleep(0.05)  # hold the lock long enough for the second to queue
        return "a"

    monkeypatch.setattr(builtins, "input", blocking)

    async def go():
        return await asyncio.gather(
            ap.approve(req("writefile"), ctx),
            ap.approve(req("writefile"), ctx),
        )

    assert asyncio.run(go()) == [True, True]
    assert len(calls) == 1  # only one prompt for the shared tool name


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("yes", False),
        ("", False),
        ("  ", False),
        ("Y", True),
        ("y", True),
        ("n", False),
    ],
)
def test_console_answer_parsing(monkeypatch, answer, expected):
    # Only "y"/"Y" (after strip().lower()) approve; everything else denies.
    ctx = make_ctx()
    monkeypatch.setattr(builtins, "input", lambda _p: answer)
    assert asyncio.run(ConsoleApprover().approve(req(), ctx)) is expected


def test_console_handles_none_reason(monkeypatch):
    # A None reason is interpolated raw as "None" and approve() still works.
    ctx = make_ctx()
    seen = []
    monkeypatch.setattr(builtins, "input", lambda p: (seen.append(p), "y")[1])
    request = ApprovalRequest(call={"name": "tool", "arguments": {}}, reason=None)
    assert asyncio.run(ConsoleApprover().approve(request, ctx)) is True
    assert "None" in seen[0]
