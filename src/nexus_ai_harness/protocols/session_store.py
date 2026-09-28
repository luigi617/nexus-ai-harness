from __future__ import annotations

from abc import abstractmethod

from nexus_ai_harness.protocols.plugin import Plugin


class SessionStore(Plugin):
    """Durable persistence of session snapshots, keyed by session id.

    Lets a run be saved and later resumed. To respect the layering rule
    (``protocols`` must not depend on ``harness``), the store works in plain
    ``dict`` snapshots rather than ``Session`` objects: the harness serializes a
    session to a JSON-safe dict before :meth:`save` and rehydrates one from the
    dict :meth:`load` returns. Implementations persist as JSON only (never
    pickle) and confine any per-session file path by ``session_id``.

    A store keeps snapshots verbatim and is agnostic to their format: the
    snapshot carries its own schema ``version``, and upgrading an older one is
    done by the reader after :meth:`load`, not by the store. Branching a session
    is likewise built on :meth:`load` and :meth:`save`, so any store supports it.
    """

    @abstractmethod
    def save(self, session_id: str, data: dict) -> None:
        """Persist a session snapshot, overwriting any existing one.

        Args:
            session_id: The session's stable identifier and storage key.
            data: A JSON-serializable snapshot of the session.
        """

    @abstractmethod
    def load(self, session_id: str) -> dict | None:
        """Load a previously saved snapshot.

        Args:
            session_id: The session's identifier.

        Returns:
            The saved snapshot, or ``None`` if no session with that id exists.
        """

    @abstractmethod
    def list_ids(self) -> list[str]:
        """Return the ids of all stored sessions."""

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a stored session.

        Args:
            session_id: The session's identifier.

        Returns:
            ``True`` if a session was deleted, ``False`` if none existed.
        """
