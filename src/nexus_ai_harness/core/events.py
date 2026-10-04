from __future__ import annotations

from dataclasses import dataclass, field

from nexus_ai_harness.core.ids import new_id
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
    """A model completion is about to be requested; carries the sent history.

    Attributes:
        history: The messages sent to the model.
        call_id: Correlates this call with its :class:`ModelCallCompleted`.
    """

    history: list[Message]
    # compare=False keeps value equality as it was before ids were added.
    call_id: str = field(default_factory=lambda: new_id("mcall"), compare=False)


@dataclass
class ResponseReceived(Event):
    response: Response


@dataclass
class MessageAdded(Event):
    message: Message


@dataclass
class ToolCallStarted(Event):
    """A tool call passed permission and is about to run.

    Attributes:
        call: The tool call requested by the model.
        call_id: Correlates this call with its :class:`ToolCallCompleted`.
    """

    call: dict
    # compare=False keeps value equality as it was before ids were added.
    call_id: str = field(default_factory=lambda: new_id("tcall"), compare=False)


@dataclass
class ToolCallCompleted(Event):
    """A tool call finished; carries the result message.

    Attributes:
        call: The tool call requested by the model.
        result: The tool message returned to the model.
        call_id: The ``call_id`` of the matching :class:`ToolCallStarted`.
        duration: Seconds the tool took to run.
        error: The exception the tool raised, or ``None`` if it returned normally.
    """

    call: dict
    result: Message
    call_id: str = ""
    duration: float = 0.0
    error: Exception | None = None


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
class SessionSaveFailed(Event):
    """Persisting a session snapshot raised; the run carries on regardless."""

    session_id: str
    error: BaseException


@dataclass
class LoopStopped(Event):
    """The loop terminated; reason distinguishes how (completed, interrupted, guard)."""

    reason: str


@dataclass
class ModelCallCompleted(Event):
    """A model completion returned; pairs with :class:`ModelCallStarted`.

    Attributes:
        call_id: The ``call_id`` of the matching :class:`ModelCallStarted`.
        response: The model's response, including its token usage and cost.
        duration: Seconds the completion took.
    """

    call_id: str
    response: Response
    duration: float
