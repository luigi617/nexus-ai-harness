from __future__ import annotations

import asyncio

from nexus_ai_harness.core.response import Response
from nexus_ai_harness.harness import NexusAIHarness
from nexus_ai_harness.harness.session import Session
from nexus_ai_harness.plugins.context_manager.summarizing import SummaryState
from nexus_ai_harness.plugins.loops import AgenticLoop
from nexus_ai_harness.plugins.permissions import AllowList, AutoApprove
from nexus_ai_harness.plugins.persistence import (
    FileSessionStore,
    fork_session,
    resume,
)
from nexus_ai_harness.plugins.persistence.autosave import AutoSave
from tests.conftest import RecordingTool, ScriptedModel


def _harness(store: FileSessionStore, *responses: Response) -> NexusAIHarness:
    return (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(ScriptedModel(*responses))
        .use(store)
        .use(AutoSave())
    )


def test_run_is_autosaved_then_resumed(tmp_path):
    store = FileSessionStore(tmp_path)

    async def scenario():
        # Turn 1: a plain completion. AutoSave persists it as it runs.
        session = Session()
        harness = _harness(store, Response(text="Noted: the code is ORCA-1."))
        await harness.run("Remember the code ORCA-1.", session=session)
        await harness.stop()

        # The session is on disk and carries the turn-1 exchange.
        snapshot = store.load(session.id)
        assert snapshot is not None
        assert snapshot["id"] == session.id
        contents = [m["content"] for m in snapshot["history"]]
        assert "Remember the code ORCA-1." in contents
        assert "Noted: the code is ORCA-1." in contents

        # Resume into a fresh harness (as a new process would) and continue.
        resumed = resume(store, session.id)
        assert resumed is not None
        assert resumed.id == session.id
        history_before = len(resumed.history)
        assert history_before >= 2  # turn-1 user + assistant carried over

        harness2 = _harness(store, Response(text="The code is ORCA-1."))
        await harness2.run("What was the code again?", session=resumed)
        await harness2.stop()
        return resumed

    resumed = asyncio.run(scenario())

    # The continued session holds both turns, and the store reflects the latest.
    all_contents = [m.content for m in resumed.history]
    assert "Remember the code ORCA-1." in all_contents
    assert "What was the code again?" in all_contents

    final = store.load(resumed.id)
    assert final is not None
    final_contents = [m["content"] for m in final["history"]]
    assert "What was the code again?" in final_contents


class _CountingStore(FileSessionStore):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.saves = 0

    def save(self, session_id: str, data: dict) -> None:
        self.saves += 1
        super().save(session_id, data)


def test_autosave_writes_per_iteration_not_per_message(tmp_path):
    store = _CountingStore(tmp_path)
    calls = [{"name": "echo", "id": str(n), "arguments": {}} for n in range(3)]
    harness = (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(ScriptedModel(Response(text="", tool_calls=calls), Response(text="ok")))
        .use(RecordingTool("echo"))
        .use(AllowList(["echo"]))
        .use(AutoApprove())
        .use(store)
        .use(AutoSave())
    )
    result = harness.run_sync("go")
    messages = len(result.session.history)
    assert messages == 6  # user, assistant, three tool results, assistant
    # IterationStarted(0) for the prompt, one per completed iteration, the end.
    assert store.saves == 4
    saved = store.load(result.session.id)
    assert saved is not None
    assert len(saved["history"]) == messages
    assert saved["stop_reason"] == "completed"


def test_fork_branches_without_touching_the_original(tmp_path):
    store = FileSessionStore(tmp_path)

    async def scenario():
        root = Session()
        harness = _harness(store, Response(text="Noted: the code is ORCA-1."))
        await harness.run("Remember the code ORCA-1.", session=root)
        await harness.stop()
        original = store.load(root.id)

        # Branch the saved session and continue only the branch.
        branch = fork_session(store, root.id)
        assert branch is not None
        harness2 = _harness(store, Response(text="Branch reply."))
        await harness2.run("Take the branch.", session=branch)
        await harness2.stop()
        return root.id, original, branch

    root_id, original, branch = asyncio.run(scenario())

    assert store.load(root_id) == original  # autosave on the branch left it alone
    saved_branch = store.load(branch.id)
    assert saved_branch is not None
    assert saved_branch["parent_id"] == root_id
    contents = [m["content"] for m in saved_branch["history"]]
    assert "Remember the code ORCA-1." in contents
    assert "Take the branch." in contents
    assert sorted(store.list_ids()) == sorted([root_id, branch.id])


def test_summary_state_survives_resume(tmp_path):
    # A resumed run must reuse the saved summary instead of losing it.
    store = FileSessionStore(tmp_path)
    session = Session()
    session.state(SummaryState).upto = 1
    session.state(SummaryState).text = "earlier recap"
    harness = _harness(store, Response(text="ok"))
    harness.run_sync("hello", session=session)

    resumed = resume(store, session.id)
    assert resumed is not None
    assert resumed.state(SummaryState) == SummaryState(upto=1, text="earlier recap")
