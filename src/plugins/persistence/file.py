from __future__ import annotations

import json
import os
from dataclasses import asdict, fields
from pathlib import Path

from core.message import Message
from core.run import RunState
from harness.session import Session
from protocols.context import Context
from protocols.session_store import SessionStore

_MESSAGE_FIELDS = {f.name for f in fields(Message)}
"""Field names accepted by :class:`Message`, used to filter loaded snapshots."""


def snapshot_from_ctx(ctx: Context) -> dict:
    """Build a JSON-safe snapshot of the session behind ``ctx``.

    Args:
        ctx: The run context whose session should be captured.

    Returns:
        A dict with the session id, its history as message dicts, and the
        current run stop reason.
    """
    return {
        "id": ctx.session_id,
        "history": [asdict(message) for message in ctx.history],
        "stop_reason": ctx.state(RunState).stop_reason,
    }


def session_from_dict(data: dict) -> Session:
    """Rebuild a :class:`~harness.session.Session` from a snapshot dict.

    Tolerant of imperfect input: a snapshot missing its ``id`` gets a fresh
    one, message entries that aren't dicts or lack a ``role`` are skipped, and
    unknown message keys (e.g. from a newer schema) are dropped rather than
    raising — so a corrupt or forward-versioned file degrades instead of
    crashing the caller.

    Args:
        data: A snapshot produced by :func:`snapshot_from_ctx`.

    Returns:
        A session carrying the saved id, history, and run stop reason.
    """
    if not isinstance(data, dict):  # a non-object snapshot degrades to empty
        data = {}
    history = []
    for message in data.get("history", []):
        if not isinstance(message, dict) or "role" not in message:
            continue  # skip malformed entries rather than raising
        kwargs = {k: v for k, v in message.items() if k in _MESSAGE_FIELDS}
        history.append(Message(**kwargs))
    session = Session(history=history)
    if data.get("id"):
        session.id = str(data["id"])
    session.state(RunState).stop_reason = str(data.get("stop_reason", "") or "")
    return session


class FileSessionStore(SessionStore):
    """One JSON file per session at ``<dir>/<session_id>.json``.

    The ``session_id`` is treated as untrusted and confined to the store
    directory so it can't escape via ``..`` or an absolute path. Snapshots are
    persisted as JSON only, never pickle.
    """

    def __init__(self, directory: str | Path = "~/.nexus-ai-harness/sessions") -> None:
        self._dir = Path(directory).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, data: dict) -> None:
        path = self._path(session_id)
        # Write via temp + os.replace so an interrupted save can't leave a
        # truncated snapshot — resume must never read a half-written file.
        payload = json.dumps(data, indent=2)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)  # drop the temp if replace didn't consume it

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

    def _path(self, session_id: str) -> Path:
        # session_id may be untrusted; confine it to the store dir so it can't escape.
        if not session_id or not session_id.strip():
            raise ValueError(f"invalid session id: {session_id!r}")
        candidate = (self._dir / f"{session_id}.json").resolve()
        if candidate.parent != self._dir.resolve():
            raise ValueError(f"invalid session id: {session_id!r}")
        return candidate


def resume(store: SessionStore, session_id: str) -> Session | None:
    """Reconstruct a saved session from ``store``.

    Args:
        store: The session store to load the snapshot from.
        session_id: The id of the session to resume.

    Returns:
        The rebuilt session, or ``None`` if no session with that id was stored.
    """
    data = store.load(session_id)
    return session_from_dict(data) if data is not None else None
