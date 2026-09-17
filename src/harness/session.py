from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Event
from typing import TypeVar

from core.ids import new_id
from core.message import Message
from protocols.intervention import Intervention

T = TypeVar("T")


@dataclass
class Session:
    id: str = field(default_factory=lambda: new_id("sess"))
    history: list[Message] = field(default_factory=list)
    # Shared signal so interrupting the root also stops in-flight subagents.
    _interrupt: Event = field(default_factory=Event)
    # Per-session intervention inbox (not shared with subagents).
    _inbox: deque[Intervention] = field(default_factory=deque)
    _state: dict[type, object] = field(default_factory=dict)

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

    def state(self, cls: type[T]) -> T:
        """Return this session's instance of ``cls``.

        Creates a default on first access.
        """
        inst = self._state.get(cls)
        if inst is None:
            inst = cls()
            self._state[cls] = inst
        return inst  # type: ignore[return-value]
