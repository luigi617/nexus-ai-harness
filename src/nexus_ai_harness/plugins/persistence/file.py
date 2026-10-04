from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import asdict, fields
from pathlib import Path

from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.persistable import (
    PersistenceWarning,
    dump,
    persisted_name,
    restore,
)
from nexus_ai_harness.core.run import RunState
from nexus_ai_harness.harness.session import Session
from nexus_ai_harness.plugins.persistence.schema import SCHEMA_VERSION, migrate
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.intervention import Intervention
from nexus_ai_harness.protocols.session_store import SessionStore
from nexus_ai_harness.services.fs import replace_retrying

_MESSAGE_FIELDS = {f.name for f in fields(Message)}
"""Field names accepted by :class:`Message`, used to filter loaded snapshots."""


def snapshot_from_ctx(ctx: Context) -> dict:
    """Build a JSON-safe snapshot of the session behind ``ctx``.

    Args:
        ctx: The run context whose session should be captured.

    Returns:
        A dict with the schema ``version``, the session ``id`` and
        ``parent_id``, its ``history`` as message dicts, the run
        ``stop_reason``, the persistable plugin ``state``, and any pending
        ``interventions``.
    """
    return _snapshot(
        session_id=ctx.session_id,
        parent_id=ctx.parent_session_id,
        history=ctx.history,
        stop_reason=ctx.state(RunState).stop_reason,
        state=ctx.persisted_state(),
        interventions=ctx.pending_interventions(),
    )


def snapshot_from_session(session: Session) -> dict:
    """Build a JSON-safe snapshot of ``session``.

    Args:
        session: The session to capture.

    Returns:
        A snapshot in the same format as :func:`snapshot_from_ctx`.
    """
    return _snapshot(
        session_id=session.id,
        parent_id=session.parent_id,
        history=session.history,
        stop_reason=session.state(RunState).stop_reason,
        state=session.persisted_state(),
        interventions=session.pending_interventions(),
    )


def session_from_dict(data: dict) -> Session:
    """Rebuild a :class:`~harness.session.Session` from a snapshot dict.

    The snapshot is first upgraded to the current schema version (a snapshot
    without a ``version`` is treated as version 1). Within a supported version,
    loading is tolerant of imperfect input: a snapshot missing its ``id`` gets a
    fresh one, message entries that aren't dicts or lack a ``role`` are
    skipped, unknown message keys are dropped, and state or interventions whose
    class isn't registered are skipped with a
    :class:`~core.persistable.PersistenceWarning`.

    Args:
        data: A snapshot produced by :func:`snapshot_from_ctx`.

    Returns:
        A session carrying the saved id, lineage, history, run stop reason,
        persisted plugin state, and pending interventions.

    Raises:
        SnapshotVersionError: If the snapshot's version is newer than this
            release supports or is invalid, so data is never silently dropped.
    """
    if not isinstance(data, dict):  # a non-object snapshot degrades to empty
        data = {}
    data = migrate(data)
    history = []
    for message in data.get("history", []):
        if not isinstance(message, dict) or "role" not in message:
            continue  # skip malformed entries rather than raising
        kwargs = {k: v for k, v in message.items() if k in _MESSAGE_FIELDS}
        history.append(Message(**kwargs))
    session = Session(history=history)
    if data.get("id"):
        session.id = str(data["id"])
    if data.get("parent_id"):
        session.parent_id = str(data["parent_id"])
    session.state(RunState).stop_reason = str(data.get("stop_reason", "") or "")
    state = data.get("state")
    if isinstance(state, dict):
        session.restore_state(
            {str(k): v for k, v in state.items() if isinstance(v, dict)}
        )
    for intervention in _load_interventions(data.get("interventions")):
        session.submit(intervention)
    return session


class FileSessionStore(SessionStore):
    """One JSON file per session at ``<dir>/<session_id>.json``.

    The ``session_id`` is treated as untrusted and confined to the store
    directory so it can't escape via ``..`` or an absolute path. Snapshots are
    persisted as JSON only, never pickle. Each save writes a uniquely named
    temp file, flushes it to disk, and atomically renames it over the old
    snapshot, so a crash or a concurrent save never leaves a partial file.
    Snapshot files are created readable by the owner only.
    """

    def __init__(self, directory: str | Path = "~/.nexus-ai-harness/sessions") -> None:
        self._dir = Path(directory).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, data: dict) -> None:
        self._write(session_id, data, exclusive=False)

    def create(self, session_id: str, data: dict) -> None:
        # Hard-linking the temp file into place fails if the target exists, so
        # two racing creates (or a case-folded name clash) can't both win.
        self._write(session_id, data, exclusive=True)

    def load(self, session_id: str) -> dict | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        # Treat a corrupt, truncated, or non-object file as absent rather than
        # raising, so resume() degrades to None as documented (cf. FileMemoryStore).
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # ValueError covers json.JSONDecodeError
            return None
        return result if isinstance(result, dict) else None

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    # --- persistence helpers -------------------------------------------------

    def _write(self, session_id: str, data: dict, *, exclusive: bool) -> None:
        path = self._path(session_id)
        # Serialize first so a non-JSON payload raises before any file is touched.
        payload = json.dumps(data, indent=2)
        # A unique temp name stops concurrent saves of one id clobbering each other.
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            if exclusive:
                try:
                    os.link(tmp, path)
                except FileExistsError:
                    raise FileExistsError(
                        f"session {session_id!r} already exists"
                    ) from None
                except OSError:
                    # No hard links on this filesystem (e.g. FAT): fall back to a
                    # best-effort check, as the protocol's default does.
                    if path.exists():
                        raise FileExistsError(
                            f"session {session_id!r} already exists"
                        ) from None
                    replace_retrying(tmp, path)
            else:
                replace_retrying(tmp, path)
            _fsync_dir(path.parent)
        finally:
            tmp.unlink(missing_ok=True)  # drop the temp if it wasn't moved into place

    def _path(self, session_id: str) -> Path:
        # session_id may be untrusted; confine it to the store dir so it can't escape.
        if not session_id or not session_id.strip():
            raise ValueError(f"invalid session id: {session_id!r}")
        candidate = (self._dir / f"{session_id}.json").resolve()
        # Reject ids that normalize to another name (e.g. "./x" -> "x") too, so an
        # id always maps to exactly one file and list_ids() round-trips it. Casefold
        # the name check only: a case-insensitive filesystem resolves to the
        # on-disk spelling, which must still be accepted as that id.
        name = f"{session_id}.json"
        if (
            candidate.parent != self._dir.resolve()
            or candidate.name.casefold() != name.casefold()
        ):
            raise ValueError(f"invalid session id: {session_id!r}")
        return candidate


def resume(store: SessionStore, session_id: str) -> Session | None:
    """Reconstruct a saved session from ``store``.

    Resuming keeps the session's id, so saving it again (for example through
    :class:`~plugins.persistence.autosave.AutoSave`) overwrites the stored
    snapshot. Use :func:`fork_session` to continue without touching it.

    Args:
        store: The session store to load the snapshot from.
        session_id: The id of the session to resume.

    Returns:
        The rebuilt session, or ``None`` if no session with that id was stored.

    Raises:
        SnapshotVersionError: If the stored snapshot is from a newer, unsupported
            schema version.
    """
    data = store.load(session_id)
    return session_from_dict(data) if data is not None else None


def fork_session(
    store: SessionStore, source_id: str, new_id: str | None = None
) -> Session | None:
    """Branch a stored session into a new one, leaving the original intact.

    The fork is saved to ``store`` straight away under its new id, with
    ``parent_id`` recording ``source_id``, and returned ready to run. It is
    written with :meth:`~protocols.session_store.SessionStore.create`, so an
    existing session is never replaced; :class:`FileSessionStore` makes that
    check atomic, other stores may only make it best-effort.

    Args:
        store: The session store holding the source session.
        source_id: The id of the session to branch from.
        new_id: The fork's id; a fresh one is generated when omitted.

    Returns:
        The forked session, or ``None`` if no session ``source_id`` was stored.

    Raises:
        ValueError: If a session with ``new_id`` already exists in ``store``.
        SnapshotVersionError: If the source snapshot is from a newer,
            unsupported schema version.
    """
    source = resume(store, source_id)
    if source is None:
        return None
    forked = source.fork(new_id)
    try:
        store.create(forked.id, snapshot_from_session(forked))
    except FileExistsError as exc:  # never overwrite another session
        raise ValueError(f"session {forked.id!r} already exists") from exc
    return forked


def _snapshot(
    *,
    session_id: str,
    parent_id: str | None,
    history: list[Message],
    stop_reason: str,
    state: dict[str, dict],
    interventions: list[Intervention],
) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "id": session_id,
        "parent_id": parent_id,
        "history": [asdict(message) for message in history],
        "stop_reason": stop_reason,
        "state": state,
        "interventions": _dump_interventions(interventions),
    }


def _dump_interventions(interventions: list[Intervention]) -> list[dict]:
    out = []
    for intervention in interventions:
        name = persisted_name(type(intervention))
        data = dump(intervention)  # warns and yields None when not persistable
        if name is not None and data is not None:
            out.append({"type": name, "data": data})
    return out


def _load_interventions(raw: object) -> list[Intervention]:
    if not isinstance(raw, list):
        return []
    out: list[Intervention] = []
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
            continue  # malformed entries are skipped like malformed messages
        inst = restore(entry["type"], entry.get("data"))
        if isinstance(inst, Intervention):
            out.append(inst)
        elif inst is not None:
            warnings.warn(
                f"skipping {entry['type']!r}: it is not an Intervention",
                PersistenceWarning,
                stacklevel=2,
            )
    return out


def _fsync_dir(directory: Path) -> None:
    # Best-effort durability of the rename: Windows and some filesystems can't do it.
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
