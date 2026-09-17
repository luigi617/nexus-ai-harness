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

Register a handler for a single event type with `harness.on(...)`; the handler
receives the typed event (and optionally the run context), and fires for the
type and any subclass — so listening on `Event` observes everything:

```python
harness.on(ModelCallStarted, lambda e: print("calling model with", len(e.history)))
harness.on(ToolCallCompleted, lambda e, ctx: log(ctx.session_id, e.result.content))
```

For a stateful observer, implement the `Hook` protocol instead and register it
with `.use(...)`; its `on(event, ctx)` sees every emitted event and can persist
across the run via `ctx.state(...)` (see `plugins/hooks/` for examples).

<!-- TODO:
- Minimal worked example (e.g. a Tool).
- The `kind` ClassVar and why it keys the registry.
- Reading/writing per-session state via `ctx.state(cls)`.
- Emitting events with `ctx.emit(...)`.
- Testing a plugin (see tests/conftest.py: make_ctx, ScriptedModel, RecordingTool).
-->
