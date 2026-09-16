from __future__ import annotations

import asyncio

from core.spawn import SpawnState
from plugins.loops import AgenticLoop
from plugins.permissions import AutoApprove
from plugins.spawner import InProcessSpawner
from plugins.tools import Recall, Remember
from protocols.approver import Approver
from protocols.tool import Tool
from tests.conftest import make_ctx


def test_fork_inherits_all_and_increments_depth():
    parent = make_ctx(AgenticLoop(), Remember(), Recall())
    child = parent.fork(None)
    assert child.state(SpawnState).depth == 1
    assert {t.name for t in child.all(Tool)} == {"remember", "recall"}


def test_fork_restricts_to_given_plugins():
    parent = make_ctx(AgenticLoop(), Remember(), Recall())
    only_remember = Remember()
    child = parent.fork([only_remember])
    assert {t.name for t in child.all(Tool)} == {"remember"}


def test_fork_inherits_parent_approver():
    approver = AutoApprove()
    parent = make_ctx(approver)
    child = parent.fork([Remember()])  # no approver in the restricted set
    assert child.get(Approver) is approver  # escalation path preserved


def test_fork_is_isolated_from_parent():
    parent = make_ctx()
    child = parent.fork(None)
    child.state(SpawnState).depth = 99
    assert parent.state(SpawnState).depth == 0  # independent state


def test_spawner_refuses_beyond_max_depth():
    ctx = make_ctx()
    ctx.state(SpawnState).depth = 3
    out = asyncio.run(InProcessSpawner(max_depth=2).run(ctx, "task"))
    assert "max subagent depth" in out


def test_spawner_runs_child_within_depth():
    from core.response import Response
    from protocols.model import Model

    class P(Model):
        async def complete(self, history, ctx):
            return Response(text="child-result")

    parent = make_ctx(AgenticLoop(), P())
    child = parent.fork(None)  # depth 1
    out = asyncio.run(InProcessSpawner(max_depth=2).run(child, "do it"))
    assert out == "child-result"


# --- Subagent tool (end-to-end delegation) -------------------------------


def test_subagent_tool_delegates_and_returns_distilled_result():
    from core.response import Response
    from plugins.tools import Subagent
    from protocols.model import Model

    class P(Model):
        async def complete(self, history, ctx):
            return Response(text="child-answer")

    sub = Subagent()
    parent = make_ctx(AgenticLoop(), P(), InProcessSpawner(), sub)
    out = asyncio.run(sub.run({"task": "do research"}, parent))
    assert out == "child-answer"


def test_subagent_tool_errors_without_spawner():
    from plugins.tools import Subagent

    out = asyncio.run(Subagent().run({"task": "x"}, make_ctx()))
    assert "no spawner" in out


def test_subagent_tool_rejects_empty_task():
    from plugins.tools import Subagent

    out = asyncio.run(Subagent().run({"task": "  "}, make_ctx(InProcessSpawner())))
    assert "no task" in out
