from __future__ import annotations

import pytest

from nexus_ai_harness.core.events import MessageAdded
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.run import RunState
from nexus_ai_harness.core.spawn import SpawnState
from nexus_ai_harness.plugins.persistence import (
    AutoSave,
    FileSessionStore,
    resume,
    session_from_dict,
    snapshot_from_ctx,
)
from nexus_ai_harness.protocols.session_store import SessionStore
from tests.conftest import make_ctx


def store(tmp_path):
    return FileSessionStore(tmp_path)


def test_store_satisfies_protocol(tmp_path):
    assert isinstance(store(tmp_path), SessionStore)


def test_save_load_roundtrip(tmp_path):
    s = store(tmp_path)
    data = {"id": "sess_1", "history": [], "stop_reason": "done"}
    s.save("sess_1", data)
    assert s.load("sess_1") == data


def test_load_missing_returns_none(tmp_path):
    assert store(tmp_path).load("nope") is None


def test_save_overwrites(tmp_path):
    s = store(tmp_path)
    s.save("sess_1", {"stop_reason": "a"})
    s.save("sess_1", {"stop_reason": "b"})
    assert s.load("sess_1") == {"stop_reason": "b"}


def test_list_ids(tmp_path):
    s = store(tmp_path)
    s.save("sess_a", {})
    s.save("sess_b", {})
    assert s.list_ids() == ["sess_a", "sess_b"]


def test_delete(tmp_path):
    s = store(tmp_path)
    s.save("sess_1", {})
    assert s.delete("sess_1") is True
    assert s.delete("sess_1") is False
    assert s.load("sess_1") is None


@pytest.mark.parametrize("bad", ["../escape", "a/b", "/etc/passwd"])
def test_path_escape_raises(tmp_path, bad):
    s = store(tmp_path)
    with pytest.raises(ValueError, match="invalid session id"):
        s.save(bad, {})


def test_snapshot_from_ctx_captures_state():
    ctx = make_ctx()
    ctx.add_message(Message(role="user", content="hi"))
    ctx.state(RunState).stop_reason = "completed"
    snap = snapshot_from_ctx(ctx)
    assert snap["id"] == ctx.session_id
    assert snap["stop_reason"] == "completed"
    assert snap["history"][0]["content"] == "hi"


def test_snapshot_session_roundtrip():
    ctx = make_ctx()
    ctx.add_message(Message(role="user", content="hi", id="msg_1"))
    ctx.add_message(Message(role="assistant", content="yo", id="msg_2"))
    ctx.state(RunState).stop_reason = "completed"
    session = session_from_dict(snapshot_from_ctx(ctx))
    assert session.id == ctx.session_id
    assert [(m.role, m.content, m.id) for m in session.history] == [
        ("user", "hi", "msg_1"),
        ("assistant", "yo", "msg_2"),
    ]
    assert session.state(RunState).stop_reason == "completed"


def test_autosave_requires_session_store():
    assert AutoSave.requires == (SessionStore,)


def test_autosave_writes_on_message_added(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave())
    ctx.add_message(Message(role="user", content="hi"))
    saved = s.load(ctx.session_id)
    assert saved is not None
    assert saved["history"][0]["content"] == "hi"


def test_autosave_writes_on_emitted_event(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave())
    ctx.emit(MessageAdded(Message(role="user", content="x")))
    assert (tmp_path / f"{ctx.session_id}.json").exists()


def test_resume_reconstructs_session(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave())
    ctx.add_message(Message(role="user", content="hi", id="msg_1"))
    ctx.state(RunState).stop_reason = "completed"
    s.save(ctx.session_id, snapshot_from_ctx(ctx))
    session = resume(s, ctx.session_id)
    assert session is not None
    assert session.id == ctx.session_id
    assert [(m.role, m.content) for m in session.history] == [("user", "hi")]
    assert session.state(RunState).stop_reason == "completed"


def test_resume_missing_returns_none(tmp_path):
    assert resume(store(tmp_path), "absent") is None


# --- robustness: AutoSave must never disrupt the run ------------------------


class _BoomStore(SessionStore):
    """A SessionStore whose save always raises, to prove AutoSave swallows it."""

    def save(self, session_id: str, data: dict) -> None:
        raise OSError("disk full")

    def load(self, session_id: str) -> dict | None:
        return None

    def list_ids(self) -> list[str]:
        return []

    def delete(self, session_id: str) -> bool:
        return False


def test_autosave_swallows_store_errors():
    # A persistence failure must not propagate out of emit()/add_message and
    # crash the agent loop.
    ctx = make_ctx(_BoomStore(), AutoSave())
    ctx.add_message(Message(role="user", content="hi"))  # must not raise


def test_autosave_ignores_forked_subagent_sessions(tmp_path):
    # Forked subagents run at depth > 0 on their own ephemeral sessions; saving
    # them would pollute the store, so AutoSave must skip them.
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave())
    ctx.state(SpawnState).depth = 1
    ctx.add_message(Message(role="user", content="child work"))
    assert s.list_ids() == []


# --- robustness: tolerant deserialization -----------------------------------


def test_session_from_dict_defaults_missing_id():
    session = session_from_dict({"history": [], "stop_reason": ""})
    assert session.id  # a fresh id is generated rather than raising KeyError


def test_session_from_dict_skips_malformed_messages():
    data = {
        "id": "sess_1",
        "history": [
            {"role": "user", "content": "keep"},
            {"content": "no role — dropped"},
            "not a dict — dropped",
        ],
    }
    session = session_from_dict(data)
    assert [(m.role, m.content) for m in session.history] == [("user", "keep")]


def test_session_from_dict_drops_unknown_message_fields():
    # A snapshot from a newer schema (extra Message field) must not raise.
    data = {"id": "s", "history": [{"role": "user", "content": "hi", "future": 1}]}
    session = session_from_dict(data)
    assert session.history[0].content == "hi"


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_session_id_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="invalid session id"):
        store(tmp_path).save(bad, {})


# --- robustness: corrupt/partial files must degrade, not crash resume --------


def test_load_corrupt_json_returns_none(tmp_path):
    # A truncated file (as an interrupted save would leave) must not raise out
    # of load()/resume() — it degrades to None like a missing session.
    s = store(tmp_path)
    (tmp_path / "sess_1.json").write_text('{"id": "sess_1", "hist', encoding="utf-8")
    assert s.load("sess_1") is None
    assert resume(s, "sess_1") is None


def test_load_non_object_json_returns_none(tmp_path):
    # Valid JSON that isn't an object (list/scalar) must degrade, not raise.
    s = store(tmp_path)
    (tmp_path / "sess_1.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert s.load("sess_1") is None
    assert resume(s, "sess_1") is None


def test_session_from_dict_tolerates_non_dict():
    # session_from_dict is public; a non-dict payload degrades to an empty session.
    session = session_from_dict("not a dict")  # type: ignore[arg-type]
    assert session.history == []


def test_save_is_atomic_and_leaves_no_temp(tmp_path):
    # os.replace-based save: content is intact and no *.tmp files linger.
    s = store(tmp_path)
    s.save("sess_1", {"id": "sess_1", "history": [], "stop_reason": "done"})
    assert s.load("sess_1") == {"id": "sess_1", "history": [], "stop_reason": "done"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_preserves_prior_snapshot_when_serialization_fails(tmp_path):
    # A non-serializable payload must raise before touching the good file, so
    # the previously-saved snapshot survives (json.dumps runs before any write).
    s = store(tmp_path)
    s.save("sess_1", {"stop_reason": "good"})
    with pytest.raises(TypeError):
        s.save("sess_1", {"bad": object()})
    assert s.load("sess_1") == {"stop_reason": "good"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_autosave_writes_on_session_ended(tmp_path):
    # AutoSave triggers on SessionEnded too — the end-of-run final snapshot.
    from nexus_ai_harness.core.events import SessionEnded

    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave())
    ctx.add_message(Message(role="user", content="hi"))
    (tmp_path / f"{ctx.session_id}.json").unlink()  # prove SessionEnded re-saves
    ctx.emit(SessionEnded(session_id=ctx.session_id, result="done"))
    assert (tmp_path / f"{ctx.session_id}.json").exists()
