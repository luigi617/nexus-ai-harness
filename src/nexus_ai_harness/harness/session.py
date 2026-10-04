from __future__ import annotations

import copy
import warnings
from collections import deque
from dataclasses import dataclass, field
from threading import Event
from typing import TypeVar

from nexus_ai_harness.core.ids import new_id
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.persistable import (
    PersistenceWarning,
    dump,
    persisted_name,
    restore,
)
from nexus_ai_harness.protocols.intervention import Intervention

T = TypeVar("T")


@dataclass
class Session:
    """One conversation: its history, per-session plugin state, and controls.

    Attributes:
        id: The stable identifier, also the key a session store saves it under.
        history: The conversation so far, oldest message first.
        parent_id: The id of the session this one was forked from, if any.
    """

    id: str = field(default_factory=lambda: new_id("sess"))
    history: list[Message] = field(default_factory=list)
    # Shared signal so interrupting the root also stops in-flight subagents.
    _interrupt: Event = field(default_factory=Event)
    # Per-session intervention inbox (not shared with subagents).
    _inbox: deque[Intervention] = field(default_factory=deque)
    _state: dict[type, object] = field(default_factory=dict)
    parent_id: str | None = None
    # Restored state not yet claimed via state(); kept so it survives re-saving.
    _saved_state: dict[str, dict] = field(default_factory=dict)

    @property
    def interrupted(self) -> bool:
        return self._interrupt.is_set()

    def interrupt(self) -> None:
        """Request the running loop — and any subagents — to stop.

        Safe from another task/thread; honored before the next iteration.
        """
        self._interrupt.set()

    def clear_interrupt(self) -> None:
        self._interrupt.clear()

    def submit(self, intervention: Intervention) -> None:
        """Queue an intervention to apply before the loop's next iteration."""
        self._inbox.append(intervention)

    def take_interventions(self) -> list[Intervention]:
        out: list[Intervention] = []
        while self._inbox:
            out.append(self._inbox.popleft())
        return out

    def pending_interventions(self) -> list[Intervention]:
        """Return the queued interventions without consuming them."""
        return list(self._inbox)

    def state(self, cls: type[T]) -> T:
        """Return this session's instance of ``cls``.

        Creates it on first access: from saved data if the session was restored
        with state for ``cls`` (see :meth:`restore_state`), else a default.
        """
        inst = self._state.get(cls)
        if inst is None:
            inst = self._claim_saved(cls)
            if inst is None:
                inst = cls()
            self._state[cls] = inst
        return inst  # type: ignore[return-value]

    def persisted_state(self) -> dict[str, dict]:
        """Return the session's persistable state as JSON-safe data.

        Only state whose class is registered with
        :func:`~core.persistable.persistable` is included; other state is
        intentionally not persisted, and registered state that fails to
        serialize is skipped with a :class:`~core.persistable.PersistenceWarning`.
        Saved state that no code has asked for yet is carried through unchanged.

        Returns:
            A mapping of each state's stable name to its data.
        """
        out = dict(self._saved_state)
        for cls, inst in self._state.items():
            name = persisted_name(cls)
            if name is None:
                continue
            data = dump(inst)
            if data is not None:
                out[name] = data
        return out

    def restore_state(self, saved: dict[str, dict]) -> None:
        """Seed state from :meth:`persisted_state` output.

        Each entry is rebuilt lazily, the first time :meth:`state` asks for its
        class, so restoring does not require every plugin to be imported yet.

        Args:
            saved: A mapping of stable state names to their saved data.
        """
        self._saved_state.update(saved)

    def fork(self, new_id: str | None = None) -> Session:
        """Branch this session into an independent copy with a new id.

        The fork gets deep copies of the history and state plus the pending
        interventions, and records this session as its ``parent_id``, so it can
        be continued (and saved) without touching the original. Its interrupt
        signal is fresh. State that cannot be copied is dropped with a
        :class:`~core.persistable.PersistenceWarning` and starts from its
        default in the fork. This differs from ``Context.fork``, which starts a
        subagent on an empty session.

        Args:
            new_id: The fork's id; a fresh one is generated when omitted.

        Returns:
            The new session.
        """
        child = Session(history=copy.deepcopy(self.history), parent_id=self.id)
        if new_id is not None:
            child.id = new_id
        child._inbox = deque(self._inbox)
        child._saved_state = copy.deepcopy(self._saved_state)
        for cls, inst in self._state.items():
            try:
                child._state[cls] = copy.deepcopy(inst)
            except Exception as exc:  # e.g. state holding a lock or open handle
                warnings.warn(
                    f"fork dropped uncopyable state {cls.__qualname__}: {exc}",
                    PersistenceWarning,
                    stacklevel=2,
                )
        return child

    def _claim_saved(self, cls: type) -> object | None:
        name = persisted_name(cls)
        if name is None or name not in self._saved_state:
            return None
        inst = restore(name, self._saved_state.pop(name))
        return inst if isinstance(inst, cls) else None
