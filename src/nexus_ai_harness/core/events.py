from __future__ import annotations

from dataclasses import dataclass

from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.response import Response


class Event:
    """Base class for hook events."""


@dataclass
class IterationStarted(Event):
    index: int


@dataclass
class IterationCompleted(Event):
    """A loop iteration finished, whether it exited or will run again."""

    index: int


@dataclass
class ModelCallStarted(Event):
    """A model completion is about to be requested; carries the sent history."""

    history: list[Message]


@dataclass
class ResponseReceived(Event):
    response: Response


@dataclass
class MessageAdded(Event):
    message: Message


@dataclass
class ToolCallStarted(Event):
    """A tool call passed permission and is about to run."""

    call: dict


@dataclass
class ToolCallCompleted(Event):
    """A tool call finished; carries the result message."""

    call: dict
    result: Message


@dataclass
class ToolCallDenied(Event):
    """A tool call was blocked before running (denied or unknown tool)."""

    call: dict
    reason: str


@dataclass
class ApprovalRequested(Event):
    """A tool call resolved to ASK and is about to be sent to the approver."""

    call: dict
    reason: str | None


@dataclass
class SkillInvoked(Event):
    """A skill was invoked to load its instructions into the conversation."""

    name: str


@dataclass
class SessionStarted(Event):
    session_id: str


@dataclass
class SessionEnded(Event):
    session_id: str
    result: str


@dataclass
class LoopStopped(Event):
    """The loop terminated; reason distinguishes how (completed, interrupted, guard)."""

    reason: str


@dataclass
class ContextCompacted(Event):
    """A context manager shrank, or failed to shrink, the history sent to the model.

    ``strategy`` is ``"summarize"`` when older turns were folded into a model
    summary. Any other value is a deterministic fallback: ``"drop"`` omitted the
    oldest middle turns, ``"truncate"`` cut oversized message contents, and
    ``"overflow"`` means no step could bring the prompt under budget, so it is
    sent over budget. Token counts are estimates.
    """

    strategy: str
    reason: str
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int

    @property
    def fallback(self) -> bool:
        """Whether this compaction was a fallback rather than a summary."""
        return self.strategy != "summarize"
