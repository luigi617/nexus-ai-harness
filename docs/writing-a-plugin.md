# Writing a plugin

A plugin is any class that implements a protocol from `protocols/` and is
registered on a harness with `.use(...)`.

## Steps

1. Pick (or define) a protocol in `protocols/`.
2. Implement it in `plugins/`. The method may be `def` or `async def`.
3. Register it: `harness.use(MyPlugin())`.
4. Consumers resolve it by type: `ctx.get(MyProtocol)` / `ctx.all(MyProtocol)`.

## Declaring dependencies

A plugin can declare the protocols or concrete plugins it depends on with a `requires` class var so
the harness can validate a composition *before* it runs, instead of surfacing a
missing dependency only when the loop calls `resolve()`:

```python
class AgenticLoop(Loop):
    requires = (Model,)
```

`harness.validate()` walks every registered plugin and checks each `requires`
entry against the registry. It returns the harness (so it chains after `use`)
when everything resolves, and raises `MissingDependencyError` with a rendered
tree when something is missing:

```text
AgenticLoop
├── Model ✓
└── ContextManager ✗

Missing dependency: AgenticLoop requires ContextManager
```

`harness.describe_dependencies()` renders the same tree for every plugin without
raising, for introspection. `requires` defaults to `()` — declaring it is
optional and validation is an explicit step, so plugins that omit it keep
resolving lazily at run time exactly as before.

## Observing events

Protocols *replace* a capability; events *extend* one. As it runs, the harness
emits typed events (in `core/events.py`) — `SessionStarted`, `IterationStarted`,
`ModelCallStarted`, `ResponseReceived`, `ToolCallStarted`, `ToolCallCompleted`,
`IterationCompleted`, `SessionEnded`, and more. Cross-cutting concerns like
tracing, logging, or metrics observe these without owning the model, loop, or
tool registry.

To observe events, implement the `Hook` protocol and register it with
`.use(...)`. Its `on(event, ctx)` sees every emitted event — switch on the
event type for the ones you care about — and can persist across the run via
`ctx.state(...)` (see `plugins/hooks/` for examples):

```python
class Reporter(Hook):
    def on(self, event: Event, ctx: Context) -> None:
        if isinstance(event, ModelCallStarted):
            print("calling model with", len(event.history))
        elif isinstance(event, ToolCallCompleted):
            log(ctx.session_id, event.result.content)
```

## Running before or after a plugin

Events are one way to react to the harness; the other is to bind an
`Interceptor` to a plugin *type*. Its `run(ctx)` fires automatically before or
after the harness invokes that type — no event, and neither plugin knows about
the other:

```python
class TimeModel(Interceptor):
    def run(self, ctx: Context) -> None:
        ctx.state(Timing).mark()

harness.use_before(Model, TimeModel())   # runs before each model call
harness.use_after(Tool, AuditLog())      # runs after each tool call
```

<!-- TODO:
- Minimal worked example (e.g. a Tool).
- The `kind` ClassVar and why it keys the registry.
- Reading/writing per-session state via `ctx.state(cls)`.
- Emitting events with `ctx.emit(...)`.
- Testing a plugin (see tests/conftest.py: make_ctx, ScriptedModel, RecordingTool).
-->
