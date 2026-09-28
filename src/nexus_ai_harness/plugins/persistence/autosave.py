from __future__ import annotations

import warnings
from typing import ClassVar

from nexus_ai_harness.core.events import (
    Event,
    IterationCompleted,
    IterationStarted,
    MessageAdded,
    ModelCallStarted,
    SessionEnded,
    SessionSaveFailed,
)
from nexus_ai_harness.core.persistable import PersistenceWarning
from nexus_ai_harness.core.spawn import SpawnState
from nexus_ai_harness.plugins.persistence.file import snapshot_from_ctx
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.hook import Hook
from nexus_ai_harness.protocols.plugin import Plugin
from nexus_ai_harness.protocols.session_store import SessionStore


class AutoSave(Hook):
    """Persist the running session at loop boundaries so it can be resumed.

    By default a snapshot is written to the registered :class:`SessionStore`
    when a loop iteration starts or completes and just before each model call,
    but only if a message was added since the last save, and always when the
    session ends. Saving at these boundaries rather than on every message keeps
    I/O proportional to the number of turns instead of the number of messages,
    and means a resumed session never starts partway through an iteration.
    The model-call boundary is what every loop passes through (``ChatLoop``
    emits no iteration events), so the user's input is on disk before the model
    is asked, even if that call then raises. Pass ``every_message=True`` to
    also save after each message, as earlier releases did by default.

    Two invariants keep this side-channel from interfering with the run:

    * It only persists the *root* conversation. Forked subagents run on their
      own ephemeral sessions (each with a fresh id), so saving them would
      pollute the store and burn I/O; the hook ignores any context whose
      :class:`~core.spawn.SpawnState` depth is non-zero, mirroring the e2e
      recorder's gate.
    * A failed save does not abort the run (unless ``strict``). It is not
      hidden either: each failure raises a
      :class:`~core.persistable.PersistenceWarning` and emits a
      :class:`~core.events.SessionSaveFailed` event, and the session stays
      pending so the next boundary retries it.

    Args:
        every_message: Also save after every added message.
        strict: Re-raise save failures, aborting the run, instead of warning.
    """

    requires: ClassVar[tuple[type[Plugin], ...]] = (SessionStore,)

    def __init__(self, *, every_message: bool = False, strict: bool = False) -> None:
        self._every_message = every_message
        self._strict = strict
        self._unsaved: set[str] = set()  # session ids with messages not yet saved

    def on(self, event: Event, ctx: Context) -> None:
        if not isinstance(
            event,
            MessageAdded
            | IterationStarted
            | IterationCompleted
            | ModelCallStarted
            | SessionEnded,
        ):
            return
        if ctx.state(SpawnState).depth != 0:  # ignore forked subagent sessions
            return
        session_id = ctx.session_id
        if isinstance(event, MessageAdded):
            self._unsaved.add(session_id)
            if self._every_message:
                self._save(ctx)
        elif isinstance(event, SessionEnded):
            self._save(ctx)  # the final snapshot carries the run's stop reason
        elif session_id in self._unsaved:
            self._save(ctx)

    def _save(self, ctx: Context) -> None:
        store = ctx.get(SessionStore)
        if store is None:
            return
        session_id = ctx.session_id
        try:
            store.save(session_id, snapshot_from_ctx(ctx))
        except Exception as exc:
            if self._strict:
                raise
            warnings.warn(
                f"AutoSave could not save session {session_id!r}: {exc!r}",
                PersistenceWarning,
                stacklevel=2,
            )
            ctx.emit(SessionSaveFailed(session_id, exc))
            return
        self._unsaved.discard(session_id)
