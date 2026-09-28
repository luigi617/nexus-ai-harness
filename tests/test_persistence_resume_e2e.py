from __future__ import annotations

import asyncio

from nexus_ai_harness.core.response import Response
from nexus_ai_harness.harness import NexusAIHarness
from nexus_ai_harness.harness.session import Session
from nexus_ai_harness.plugins.loops import AgenticLoop
from nexus_ai_harness.plugins.persistence import FileSessionStore, resume
from nexus_ai_harness.plugins.persistence.autosave import AutoSave
from tests.conftest import ScriptedModel


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
