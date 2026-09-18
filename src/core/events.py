from __future__ import annotations

from dataclasses import dataclass

from core.message import Message
from core.response import Response


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
