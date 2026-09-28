from __future__ import annotations

import dataclasses
import logging
from typing import Any

from nexus_ai_harness.core.events import Event, SessionEnded, ToolCallCompleted
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.response import Response
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.hook import Hook

EVENT_LOGGER = "nexus_ai_harness.events"


class LoggingHook(Hook):
    """Log every harness event as one structured :mod:`logging` record.

    Each record's message is a compact ``EventName key=value ...`` line, and its
    ``extra`` carries machine-readable attributes a JSON formatter can emit:

    - ``event``: the event class name, e.g. ``"ToolCallCompleted"``.
    - ``session_id``: the session that emitted it.
    - ``event_fields``: a JSON-friendly dict of the event's fields.

    Bulky or sensitive payloads are summarized by default: the model history
    becomes a message count, messages and responses report sizes instead of
    text, and tool calls keep only their ``id`` and ``name``. Pass
    ``include_content=True`` to log message text, response text, and tool
    arguments too.

    Records go to the ``"nexus_ai_harness.events"`` logger unless another is
    given; like every harness logger it is silent until the application
    configures logging. A tool call that raised is logged at ``WARNING``.

    Args:
        logger: The logger to write to.
        level: The level for ordinary events.
        include_content: Whether to log conversation text and tool arguments.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        include_content: bool = False,
    ) -> None:
        self._logger = logger or logging.getLogger(EVENT_LOGGER)
        self._level = level
        self._include_content = include_content

    def on(self, event: Event, ctx: Context) -> None:
        level = self._level
        if isinstance(event, ToolCallCompleted) and event.error is not None:
            level = max(level, logging.WARNING)
        if not self._logger.isEnabledFor(level):
            return  # skip summarizing events nobody will see
        name = type(event).__name__
        fields = self._fields(event)
        summary = " ".join(
            f"{key}={_format(value)}"
            for key, value in fields.items()
            if isinstance(value, str | int | float | bool)
        )
        self._logger.log(
            level,
            "%s",
            f"{name} {summary}" if summary else name,
            extra={
                "event": name,
                "session_id": ctx.session_id,
                "event_fields": fields,
            },
        )

    def _fields(self, event: Event) -> dict[str, Any]:
        """Return a JSON-friendly summary of the event's fields."""
        raw = (
            {f.name: getattr(event, f.name) for f in dataclasses.fields(event)}
            if dataclasses.is_dataclass(event)
            else dict(vars(event))
        )
        out = {key: self._summarize(value) for key, value in raw.items()}
        if isinstance(event, SessionEnded) and not self._include_content:
            out["result"] = {"chars": len(event.result)}  # the final answer text
        return out

    def _summarize(self, value: Any) -> Any:
        """Reduce one field value to plain, loggable data."""
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Message):
            message: dict[str, Any] = {"role": value.role, "chars": len(value.content)}
            if value.name is not None:
                message["name"] = value.name
            if value.tool_use_id is not None:
                message["tool_use_id"] = value.tool_use_id
            if value.tool_calls:
                message["tool_calls"] = len(value.tool_calls)
            if self._include_content:
                message["content"] = value.content
            return message
        if isinstance(value, Response):
            response: dict[str, Any] = {
                "chars": len(value.text),
                "tool_calls": len(value.tool_calls),
                "usage": dict(value.usage),
                "cost": value.cost,
            }
            if self._include_content:
                response["text"] = value.text
            return response
        if isinstance(value, BaseException):
            return f"{type(value).__name__}: {value}"
        if isinstance(value, dict):  # a tool call
            call = {key: value[key] for key in ("id", "name") if key in value}
            if self._include_content:
                call["arguments"] = value.get("arguments", {})
            return call
        if isinstance(value, list | tuple):  # e.g. the history sent to the model
            return len(value)
        return repr(value)


def _format(value: str | float) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)
