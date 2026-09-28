from __future__ import annotations

import contextlib
from typing import ClassVar

from nexus_ai_harness.core.events import Event, MessageAdded, SessionEnded
from nexus_ai_harness.core.spawn import SpawnState
from nexus_ai_harness.plugins.persistence.file import snapshot_from_ctx
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.hook import Hook
from nexus_ai_harness.protocols.plugin import Plugin
from nexus_ai_harness.protocols.session_store import SessionStore


class AutoSave(Hook):
    """Persist the running session whenever its state meaningfully changes.

    On every message added and when the session ends, the current session
    snapshot is written to the registered :class:`SessionStore`, so an
    interrupted run can be resumed from its last saved point.

    Two invariants keep this side-channel from interfering with the run:

    * It only persists the *root* conversation. Forked subagents run on their
      own ephemeral sessions (each with a fresh id), so saving them would
      pollute the store and burn I/O; the hook ignores any context whose
      :class:`~core.spawn.SpawnState` depth is non-zero, mirroring the e2e
      recorder's gate.
    * A save is best-effort. :meth:`~protocols.context.Context.emit` dispatches
      hooks without catching exceptions, so a raising ``save`` (disk full, an
      unwritable directory, a non-serializable snapshot) would abort the whole
      run. Persistence must never do that, so failures are swallowed.
    """

    requires: ClassVar[tuple[type[Plugin], ...]] = (SessionStore,)

    def on(self, event: Event, ctx: Context) -> None:
        if not isinstance(event, MessageAdded | SessionEnded):
            return
        if ctx.state(SpawnState).depth != 0:  # ignore forked subagent sessions
            return
        store = ctx.get(SessionStore)
        if store is None:
            return
        # Best-effort: a persistence failure must not crash the agent loop.
        with contextlib.suppress(Exception):
            store.save(ctx.session_id, snapshot_from_ctx(ctx))
