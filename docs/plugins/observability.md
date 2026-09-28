# Observability

The harness reports what it does in three ways: standard-library logging,
timed events you can hook, and usage totals on the run result. None of them
needs an extra dependency.

## Logging

Every module logs to a `logging.getLogger(__name__)` logger under the
`nexus_ai_harness` namespace. The package installs only a `NullHandler`, so it
stays silent until your application configures logging:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("nexus_ai_harness").setLevel(logging.DEBUG)  # more detail
```

What gets logged:

- **`DEBUG`**: session start and end, each model call with its duration and
  token usage, and each tool call's start and duration.
- **`INFO`**: denied tool calls, and each MCP server connection with its tool count.
- **`WARNING`**: a tool (or MCP tool) that raised, with the full traceback, plus
  plugin `stop()` errors the harness would otherwise swallow.

The model still sees the same `"error: ..."` tool result; the traceback only goes
to the log. Tool arguments, message text, and MCP launch commands, URLs, and
environment are never logged, since they may hold user data or secrets.

## Structured event logs

Register `LoggingHook` to log every harness event as one structured record:

```python
from nexus_ai_harness.plugins.hooks import LoggingHook

harness.use(LoggingHook())                        # INFO, on "nexus_ai_harness.events"
harness.use(LoggingHook(level=logging.DEBUG))     # pick the level
harness.use(LoggingHook(include_content=True))    # also log text and tool arguments
```

Each record's message reads like `ToolCallCompleted call_id=tcall_... duration=0.012`,
and its `extra` carries `event` (the event name), `session_id`, and
`event_fields` (a JSON-friendly dict of the event's fields), ready for a JSON
formatter. By default, message text, response text, the final answer, and tool
arguments are reduced to sizes; pass `include_content=True` to keep them. A tool
call that raised is logged at `WARNING`.

## Timing and correlation

Model and tool calls each emit a started and a completed event that share a
`call_id`, so a hook can pair them even when tool calls run concurrently:

| Event | Fields |
|---|---|
| `ModelCallStarted` | `history`, `call_id` |
| `ModelCallCompleted` | `call_id`, `response` (with `usage` and `cost`), `duration` |
| `ToolCallStarted` | `call`, `call_id` |
| `ToolCallCompleted` | `call`, `result`, `call_id`, `duration`, `error` |

Durations are in seconds, measured with `time.perf_counter()`. `error` is the
exception a tool raised, or `None`. `ModelCallCompleted` is emitted just before
`ResponseReceived`.

```python
from nexus_ai_harness.core.events import ModelCallCompleted
from nexus_ai_harness.protocols.hook import Hook

class Latency(Hook):
    def on(self, event, ctx):
        if isinstance(event, ModelCallCompleted):
            print(f"{event.call_id}: {event.duration:.2f}s {event.response.usage}")
```

## Cost and token usage

`CostCounter` adds up each response's cost and its `input_tokens` and
`output_tokens`. The run result exposes the totals:

```python
result = await harness.run("hello")
result.cost    # 0.0123 (USD)
result.usage   # {"input_tokens": 1520, "output_tokens": 310}
```

`default_harness` registers `CostCounter` for you. Without it, both report zero.
Totals are per session, and a subagent's usage is counted in its own session.
