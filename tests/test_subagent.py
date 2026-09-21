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


def test_fork_override_shadows_inherited_plugin_of_the_same_protocol():
    from protocols.memory import MemoryItem, MemoryStore

    class Store(MemoryStore):
        def __init__(self, tag):
            self.tag = tag

        def save(self, text, id=None):
            return MemoryItem(text=text, id=id or "x")

        def get(self, id):
            return None

        def search(self, query, limit=5):
            return []

        def all(self):
            return []

        def delete(self, id):
            return False

    base = Store("base")
    parent = make_ctx(AgenticLoop(), base)
    child = parent.fork(overrides={MemoryStore: Store("agent")})
    assert child.get(MemoryStore).tag == "agent"  # override wins
    assert child.get(AgenticLoop) is not None  # rest of the parent still inherited
    assert parent.get(MemoryStore).tag == "base"  # parent untouched


def test_fork_override_replaces_all_inherited_plugins_of_the_protocol():
    # A harness can hold many plugins of one protocol; an override must drop
    # every inherited one, not merely out-rank them for get().
    parent = make_ctx(Remember(), Recall())  # two Tools
    only_tool = Remember()
    child = parent.fork(overrides={Tool: only_tool})
    assert child.all(Tool) == [only_tool]  # both inherited tools gone


def test_fork_override_keyed_by_concrete_class_spares_siblings():
    # A concrete-class key targets only that implementation; a sibling
    # implementation of the same protocol is left inherited.
    remember, recall = Remember(), Recall()
    swap = Remember()
    child = make_ctx(remember, recall).fork(overrides={Remember: swap})
    names = sorted(t.name for t in child.all(Tool))
    assert names == ["recall", "remember"]  # Recall untouched
    assert child.all(Remember) == [swap]  # only the Remember instance replaced


def test_fork_overrides_apply_to_a_restricted_set():
    restricted = Remember()
    override = Recall()
    child = make_ctx(Remember(), Recall()).fork(
        [restricted], overrides={Tool: override}
    )
    # The override replaces within the restricted set, not the full parent.
    assert child.all(Tool) == [override]


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


def test_subagent_tool_overrides_a_capability_for_the_child():
    from core.response import Response
    from plugins.tools import Subagent
    from protocols.model import Model

    class Parent(Model):
        async def complete(self, history, ctx):
            return Response(text="parent-model")

    class Child(Model):
        async def complete(self, history, ctx):
            return Response(text="child-model")

    # Swap the model for the subagent only.
    sub = Subagent(overrides={Model: Child()})
    parent = make_ctx(AgenticLoop(), Parent(), InProcessSpawner(), sub)
    out = asyncio.run(sub.run({"task": "go"}, parent))
    assert out == "child-model"  # child ran on the override, not the parent's model


def test_subagent_tool_errors_without_spawner():
    from plugins.tools import Subagent

    out = asyncio.run(Subagent().run({"task": "x"}, make_ctx()))
    assert "no spawner" in out


def test_subagent_tool_rejects_empty_task():
    from plugins.tools import Subagent

    out = asyncio.run(Subagent().run({"task": "  "}, make_ctx(InProcessSpawner())))
    assert "no task" in out
