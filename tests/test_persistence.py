from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from nexus_ai_harness.core.events import (
    IterationCompleted,
    IterationStarted,
    MessageAdded,
    SessionEnded,
    SessionSaveFailed,
)
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.persistable import PersistenceWarning, persistable
from nexus_ai_harness.core.run import RunState
from nexus_ai_harness.core.spawn import SpawnState
from nexus_ai_harness.harness.context import RunContext
from nexus_ai_harness.harness.registry import Registry
from nexus_ai_harness.harness.session import Session
from nexus_ai_harness.plugins.context_manager.summarizing import SummaryState
from nexus_ai_harness.plugins.interventions import InjectMessage
from nexus_ai_harness.plugins.persistence import (
    SCHEMA_VERSION,
    AutoSave,
    FileSessionStore,
    SnapshotVersionError,
    fork_session,
    migrate,
    register_migration,
    resume,
    schema,
    session_from_dict,
    snapshot_from_ctx,
    snapshot_from_session,
)
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.intervention import Intervention
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
    assert snap["version"] == SCHEMA_VERSION
    assert snap["parent_id"] is None
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


def test_autosave_every_message_writes_on_message_added(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave(every_message=True))
    ctx.add_message(Message(role="user", content="hi"))
    saved = s.load(ctx.session_id)
    assert saved is not None
    assert saved["history"][0]["content"] == "hi"


def test_autosave_writes_on_emitted_event(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave(every_message=True))
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


def test_autosave_does_not_crash_on_store_errors_but_reports_them():
    # A persistence failure must not crash the agent loop, yet must not vanish:
    # it warns and emits SessionSaveFailed.
    failures: list[SessionSaveFailed] = []
    ctx = make_ctx(_BoomStore(), AutoSave(every_message=True))
    ctx.on(SessionSaveFailed, lambda event, _ctx: failures.append(event))
    with pytest.warns(PersistenceWarning, match="disk full"):
        ctx.add_message(Message(role="user", content="hi"))  # must not raise
    assert len(failures) == 1
    assert failures[0].session_id == ctx.session_id
    assert isinstance(failures[0].error, OSError)


def test_autosave_strict_reraises_store_errors():
    ctx = make_ctx(_BoomStore(), AutoSave(every_message=True, strict=True))
    with pytest.raises(OSError, match="disk full"):
        ctx.add_message(Message(role="user", content="hi"))


def test_autosave_surfaces_serialization_failure(tmp_path):
    # A non-JSON message must be reported, not silently skipped on every save.
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave(every_message=True))
    with pytest.warns(PersistenceWarning, match="not JSON serializable"):
        ctx.add_message(Message(role="user", content=object()))  # type: ignore[arg-type]
    assert s.list_ids() == []


def test_autosave_ignores_forked_subagent_sessions(tmp_path):
    # Forked subagents run at depth > 0 on their own ephemeral sessions; saving
    # them would pollute the store, so AutoSave must skip them.
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave(every_message=True))
    ctx.state(SpawnState).depth = 1
    ctx.add_message(Message(role="user", content="child work"))
    ctx.emit(IterationCompleted(0))
    ctx.emit(SessionEnded(session_id=ctx.session_id, result="done"))
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
    # AutoSave always saves on SessionEnded — the end-of-run final snapshot.
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave())
    ctx.add_message(Message(role="user", content="hi"))
    assert s.list_ids() == []  # the default debounces per-message saves
    ctx.state(RunState).stop_reason = "completed"
    ctx.emit(SessionEnded(session_id=ctx.session_id, result="done"))
    saved = s.load(ctx.session_id)
    assert saved is not None
    assert saved["stop_reason"] == "completed"


# --- autosave debouncing ----------------------------------------------------


class _CountingStore(FileSessionStore):
    """A FileSessionStore that counts writes, to observe AutoSave's I/O."""

    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.saves = 0

    def save(self, session_id: str, data: dict) -> None:
        self.saves += 1
        super().save(session_id, data)


def test_autosave_default_saves_at_iteration_boundaries(tmp_path):
    s = _CountingStore(tmp_path)
    ctx = make_ctx(s, AutoSave())
    ctx.add_message(Message(role="user", content="hi"))
    assert s.saves == 0  # no write per message by default
    ctx.emit(IterationStarted(0))
    assert s.saves == 1  # the user prompt is on disk before the model call
    ctx.add_message(Message(role="assistant", content="a"))
    ctx.add_message(Message(role="tool", content="r"))
    ctx.emit(IterationCompleted(0))
    assert s.saves == 2
    saved = s.load(ctx.session_id)
    assert saved is not None
    assert [m["content"] for m in saved["history"]] == ["hi", "a", "r"]


def test_autosave_skips_boundaries_with_nothing_new(tmp_path):
    s = _CountingStore(tmp_path)
    ctx = make_ctx(s, AutoSave())
    ctx.add_message(Message(role="user", content="hi"))
    ctx.emit(IterationCompleted(0))
    ctx.emit(IterationStarted(1))  # nothing added since the last save
    assert s.saves == 1


def test_autosave_retries_after_a_failed_save(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave())
    boom = True

    def flaky_save(session_id: str, data: dict) -> None:
        if boom:
            raise OSError("transient")
        FileSessionStore.save(s, session_id, data)

    s.save = flaky_save  # type: ignore[method-assign]
    ctx.add_message(Message(role="user", content="hi"))
    with pytest.warns(PersistenceWarning):
        ctx.emit(IterationCompleted(0))
    boom = False
    ctx.emit(IterationStarted(1))  # still pending, so this boundary retries
    assert s.load(ctx.session_id) is not None


# --- schema versioning ------------------------------------------------------


def test_versionless_v1_snapshot_still_loads(tmp_path):
    # Files written before versioning existed are treated as v1 and migrated.
    s = store(tmp_path)
    legacy = {
        "id": "sess_old",
        "history": [{"role": "user", "content": "hi", "id": "msg_1"}],
        "stop_reason": "completed",
    }
    s.save("sess_old", legacy)
    session = resume(s, "sess_old")
    assert session is not None
    assert session.id == "sess_old"
    assert session.parent_id is None
    assert [m.content for m in session.history] == ["hi"]
    assert session.state(RunState).stop_reason == "completed"


def test_migrate_upgrades_v1_to_current():
    out = migrate({"id": "s", "history": []})
    assert out["version"] == SCHEMA_VERSION
    assert out["state"] == {}
    assert out["interventions"] == []
    assert out["parent_id"] is None


def test_newer_snapshot_version_fails_loudly(tmp_path):
    s = store(tmp_path)
    s.save("sess_1", {"version": SCHEMA_VERSION + 1, "id": "sess_1", "history": []})
    with pytest.raises(SnapshotVersionError, match="newer than the supported"):
        resume(s, "sess_1")


@pytest.mark.parametrize("bad", ["2", 0, -1, True, 1.5])
def test_invalid_snapshot_version_fails_loudly(bad):
    with pytest.raises(SnapshotVersionError, match="invalid snapshot version"):
        session_from_dict({"version": bad, "history": []})


def test_snapshot_version_error_is_a_value_error():
    assert issubclass(SnapshotVersionError, ValueError)


def test_migrations_chain_to_current(monkeypatch):
    # A future v3 only needs a bumped SCHEMA_VERSION plus one registered step.
    monkeypatch.setattr(schema, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    monkeypatch.setitem(
        schema._MIGRATIONS, SCHEMA_VERSION, lambda d: {**d, "added": True}
    )
    out = schema.migrate({"id": "s", "history": []})  # starts at v1
    assert out["version"] == SCHEMA_VERSION + 1
    assert out["added"] is True
    assert out["state"] == {}  # the v1 -> v2 step ran too


def test_missing_migration_step_fails_loudly(monkeypatch):
    monkeypatch.setattr(schema, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    with pytest.raises(SnapshotVersionError, match="no migration"):
        schema.migrate({"version": SCHEMA_VERSION, "history": []})


def test_register_migration_rejects_duplicates():
    with pytest.raises(ValueError, match="already registered"):
        register_migration(1)(lambda d: d)


# --- plugin state and pending interventions ---------------------------------


@persistable("tests.persistence.opaque")
@dataclass
class _OpaqueState:
    handle: object = None


@dataclass
class _EphemeralState:
    count: int = 0


class _Unregistered(Intervention):
    def apply(self, ctx) -> None:
        pass


def test_registered_state_roundtrips_and_unregistered_is_skipped():
    ctx = make_ctx()
    ctx.add_message(Message(role="user", content="hi"))
    summary = ctx.state(SummaryState)
    summary.upto, summary.text = 5, "recap"
    ctx.state(_EphemeralState).count = 9
    snap = snapshot_from_ctx(ctx)
    assert snap["state"] == {"summarizing.summary": {"upto": 5, "text": "recap"}}
    session = session_from_dict(snap)
    assert session.state(SummaryState) == SummaryState(upto=5, text="recap")
    assert session.state(_EphemeralState).count == 0  # not opted in


def test_non_serializable_registered_state_is_skipped_not_fatal(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx(s, AutoSave(every_message=True))
    ctx.state(_OpaqueState).handle = object()
    with pytest.warns(PersistenceWarning, match="tests.persistence.opaque"):
        ctx.add_message(Message(role="user", content="hi"))
    saved = s.load(ctx.session_id)
    assert saved is not None
    assert "tests.persistence.opaque" not in saved["state"]


def test_unclaimed_restored_state_survives_resave():
    # State for a class nobody has asked for yet (e.g. its plugin isn't loaded
    # this run) must be carried forward, not dropped by the next save.
    snap = {"version": SCHEMA_VERSION, "id": "s", "history": []}
    snap["state"] = {"some.future.plugin": {"x": 1}}
    session = session_from_dict(snap)
    assert snapshot_from_session(session)["state"] == {"some.future.plugin": {"x": 1}}


def test_pending_interventions_roundtrip(tmp_path):
    s = store(tmp_path)
    session = Session()
    session.submit(InjectMessage("be careful", role="user"))
    s.save(session.id, snapshot_from_session(session))
    restored = resume(s, session.id)
    assert restored is not None
    assert restored.take_interventions() == [InjectMessage("be careful")]
    assert session.pending_interventions() == [InjectMessage("be careful")]


def test_unregistered_intervention_is_skipped_with_warning():
    session = Session()
    session.submit(_Unregistered())
    with pytest.warns(PersistenceWarning, match="non-persistable"):
        snap = snapshot_from_session(session)
    assert snap["interventions"] == []


def test_malformed_interventions_are_skipped():
    snap = {"version": SCHEMA_VERSION, "history": []}
    snap["interventions"] = ["junk", {"data": {}}]
    assert session_from_dict(snap).pending_interventions() == []


def test_context_persistence_accessors_are_not_abstract():
    # Adding them must not break third-party Context implementations.
    for name in ("parent_session_id", "persisted_state", "pending_interventions"):
        assert name not in Context.__abstractmethods__


# --- fork -------------------------------------------------------------------


def test_session_fork_is_independent_with_lineage():
    original = Session(history=[Message(role="user", content="hi")])
    original.state(SummaryState).text = "recap"
    original.submit(InjectMessage("steer"))
    forked = original.fork()
    assert forked.id != original.id
    assert forked.parent_id == original.id
    assert forked.history == original.history
    assert forked.state(SummaryState).text == "recap"
    assert forked.pending_interventions() == [InjectMessage("steer")]
    forked.history.append(Message(role="assistant", content="branch"))
    forked.state(SummaryState).text = "changed"
    assert len(original.history) == 1
    assert original.state(SummaryState).text == "recap"
    original.interrupt()
    assert not forked.interrupted


def test_session_fork_accepts_explicit_id():
    assert Session().fork("sess_branch").id == "sess_branch"


def test_session_fork_drops_uncopyable_state_with_warning():
    original = Session()
    original.state(_OpaqueState).handle = threading.Lock()
    with pytest.warns(PersistenceWarning, match="uncopyable"):
        forked = original.fork()
    assert forked.state(_OpaqueState).handle is None


def test_fork_session_leaves_original_intact(tmp_path):
    s = store(tmp_path)
    ctx = make_ctx()
    ctx.add_message(Message(role="user", content="hi"))
    s.save(ctx.session_id, snapshot_from_ctx(ctx))
    before = s.load(ctx.session_id)

    forked = fork_session(s, ctx.session_id)
    assert forked is not None
    assert forked.parent_id == ctx.session_id
    assert s.load(ctx.session_id) == before  # the source is untouched
    saved = s.load(forked.id)
    assert saved is not None
    assert saved["parent_id"] == ctx.session_id
    assert [m["content"] for m in saved["history"]] == ["hi"]
    assert sorted(s.list_ids()) == sorted([ctx.session_id, forked.id])


def test_fork_session_refuses_to_overwrite(tmp_path):
    s = store(tmp_path)
    s.save("sess_a", {"id": "sess_a", "history": []})
    s.save("sess_b", {"id": "sess_b", "history": []})
    with pytest.raises(ValueError, match="already exists"):
        fork_session(s, "sess_a", new_id="sess_b")
    assert s.load("sess_b") == {"id": "sess_b", "history": []}


def test_fork_session_missing_source_returns_none(tmp_path):
    assert fork_session(store(tmp_path), "absent") is None


def test_forked_lineage_survives_resume(tmp_path):
    s = store(tmp_path)
    s.save("sess_root", {"id": "sess_root", "history": []})
    forked = fork_session(s, "sess_root", new_id="sess_child")
    assert forked is not None
    resumed = resume(s, "sess_child")
    assert resumed is not None
    assert resumed.parent_id == "sess_root"
    ctx = RunContext(resumed, Registry())
    assert snapshot_from_ctx(ctx)["parent_id"] == "sess_root"


# --- durable, collision-free writes -----------------------------------------


def test_concurrent_saves_of_one_id_do_not_collide(tmp_path):
    # Same-process saves used to share one temp name and could clobber each other.
    s = store(tmp_path)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(20):
                s.save("sess_1", {"writer": n, "i": i, "pad": "x" * 2000})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    loaded = s.load("sess_1")
    assert loaded is not None
    assert loaded["i"] == 19
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_fsyncs_file_and_directory(tmp_path, monkeypatch):
    synced: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    store(tmp_path).save("sess_1", {"id": "sess_1"})
    expected = 1 if os.name == "nt" else 2  # file, then its directory
    assert len(synced) == expected


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_snapshot_files_are_owner_only(tmp_path):
    store(tmp_path).save("sess_1", {"id": "sess_1"})
    assert (tmp_path / "sess_1.json").stat().st_mode & 0o077 == 0


def test_directory_fsync_failure_is_not_fatal(tmp_path, monkeypatch):
    s = store(tmp_path)
    real_open = os.open

    def no_dir_open(path, flags, *args, **kwargs):
        if Path(path).is_dir():
            raise OSError("directories unsupported")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", no_dir_open)
    s.save("sess_1", {"id": "sess_1"})
    assert s.load("sess_1") == {"id": "sess_1"}


def test_failed_replace_keeps_prior_snapshot_and_cleans_temp(tmp_path, monkeypatch):
    s = store(tmp_path)
    s.save("sess_1", {"stop_reason": "good"})

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="replace failed"):
        s.save("sess_1", {"stop_reason": "new"})
    assert s.load("sess_1") == {"stop_reason": "good"}
    assert list(tmp_path.glob("*.tmp")) == []
