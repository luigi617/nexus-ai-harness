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

A plugin that already owns a capability can also subscribe to a single event
type without being a standalone `Hook`, by calling `ctx.on(EventType, handler)`
— typically from its lifecycle `start`. The subscription is *owned* by the
plugin whose bound method `handler` is, so it is torn down automatically when
that plugin is removed (see below); `ctx.on` returns a handle whose `remove()`
cancels it sooner by hand:

```python
class Tracer(Tool, Lifecycle):
    async def start(self, ctx: Context) -> None:
        ctx.on(ModelCallStarted, self.before_model)

    def before_model(self, event: Event, ctx: Context) -> None:
        ...
```

## Running before or after a plugin

Events are one way to react to the harness; the other is an `Interceptor`. It
declares the plugin *type* it wraps as its `target` and overrides `before`,
`after`, or both; they fire automatically around every invocation of that type
— no event, and neither plugin knows about the other. Register it like any
plugin with `.use`:

```python
class TimeModel(Interceptor):
    target = Model                         # every model call
    def before(self, ctx: Context) -> None:
        ctx.state(Timing).mark()

class AuditLog(Interceptor):
    target = Tool                          # every tool call
    def after(self, ctx: Context) -> None:
        ...

harness.use(TimeModel()).use(AuditLog())
```

Set `target` to a protocol (`Model`) to wrap every implementer, or to a concrete
class to wrap only that class. `before`/`after` are observation only — they
can't see the target's arguments or return value (that's what `ContextManager`,
`Router`, and `Guard` own). Both may be `def` or `async def`. `before` and
`after` interceptors each fire in registration order, and `after` always runs,
even if the invocation raised.

Interception is faithful: it fires on *every* invocation of the target.

## Owning resources with a lifecycle

A plugin that owns resources — an HTTP client, a connection, a background task
— needs to acquire them once and release them deterministically. Subclass
`Lifecycle` alongside your plugin base: `start(ctx)` runs before the first
`run`, and `stop()` runs on shutdown. Membership is by inheritance — the harness
manages exactly the registered plugins that subclass `Lifecycle` — and either
method may be left as its inherited no-op.

```python
class DbTool(Tool, Lifecycle):
    async def start(self, ctx: Context) -> None:
        self._pool = await connect()

    async def stop(self) -> None:
        await self._pool.close()
```

The harness starts lifecycle plugins in registration order and stops them in
reverse, so register a dependency before the plugin that needs it. `run` calls
`start()` automatically (it is idempotent) but never auto-stops, so call
`stop()` to release resources, or use the harness as an async context manager to
pair the two:

```python
async with harness:        # start() on enter, stop() on exit
    await harness.run("hello")
```

Every plugin's `stop()` runs even if an earlier one raises, so one failing
teardown can't leak another's resources; the first error is re-raised afterward.

## Removing a plugin and its registrations

`harness.unuse(plugin)` reverses `use`: it removes the plugin and every
registration it owns — its `ctx.on` subscriptions and any interceptor bindings
it provides — so a plugin's side effects never outlive it. When the harness is
started and the plugin is a `Lifecycle`, its `stop()` runs first to release
resources. Removing a plugin that was never registered is a no-op.

<!-- TODO:
- Minimal worked example (e.g. a Tool).
- Type-keyed resolution: why a plugin must subclass the base it implements.
- Reading/writing per-session state via `ctx.state(cls)`.
- Emitting events with `ctx.emit(...)`.
- Testing a plugin (see tests/conftest.py: make_ctx, ScriptedModel, RecordingTool).
-->
