# Writing a plugin

A plugin is a class that implements a protocol from `protocols/` and is
registered with `.use(...)`.

## Steps

1. Pick (or define) a protocol in `protocols/`.
2. Implement it. Its methods may be `def` or `async def`.
3. Register it: `harness.use(MyPlugin())`.
4. Other code finds it by type: `ctx.get(MyProtocol)` or `ctx.all(MyProtocol)`.

## Declaring dependencies

If your plugin needs another plugin to be present, declare it with `requires`.
The harness can then check the setup before a run instead of failing mid-run:

```python
class AgenticLoop(Loop):
    requires = (Model,)
```

Call `harness.validate()` to check every plugin's requirements up front. It
raises a clear error naming what's missing:

```text
Missing dependency: AgenticLoop requires ContextManager
```

`requires` is optional; omit it and dependencies resolve at run time as usual.

## Reacting to what the agent does

There are two ways to observe or wrap the agent without owning its parts.

### Observe events with a `Hook`

As it runs, the harness emits events such as `ModelCallStarted`,
`ResponseReceived`, `ToolCallCompleted`, and `SessionEnded`. Implement `Hook` to
watch them. This is useful for logging, tracing, or metrics:

```python
class Reporter(Hook):
    def on(self, event: Event, ctx: Context) -> None:
        if isinstance(event, ToolCallCompleted):
            log(ctx.session_id, event.result.content)
```

Register it with `.use(Reporter())`. See `core/events.py` for the full list of
events. Model and tool calls emit paired started/completed events that share a
`call_id` and carry a `duration`; see [Observability](plugins/observability.md).
A plugin that calls a model itself should go through
`services.model_call.timed_complete` so the call is timed, emitted, and counted.

To log from a plugin, use a module logger (`logger = logging.getLogger(__name__)`)
and never configure handlers or levels; that is the application's job.

### Run code before or after another plugin with an `Interceptor`

An interceptor wraps every call to a plugin type. Set `target` to a protocol to
wrap all of its implementers, or to a concrete class to wrap just that one:

```python
class TimeModel(Interceptor):
    target = Model                     # wrap every model call
    def before(self, invocation: Invocation, ctx: Context) -> None: ...
    def after(
        self, invocation: Invocation, outcome: InvocationOutcome, ctx: Context
    ) -> None: ...                     # always runs, even on error
```

`invocation` carries the plugin instance, the method name, and its arguments.
`outcome` carries the returned value, or the raised exception when the call (or
an earlier interceptor's `before`) failed. Both may be `def` or `async def`.
Overriding the older `before(self, ctx)` / `after(self, ctx)` signature still
works; the harness detects which signature an override uses by inspecting it.
Interceptors must not mutate `invocation.args`, `invocation.kwargs`, or
`outcome.result` — they're for observation, not for rewriting the call.

## Owning resources with a lifecycle

If your plugin holds a resource, such as a connection, client, or background
task, subclass `Lifecycle` to open it once and close it reliably:

```python
class DbTool(Tool, Lifecycle):
    async def start(self, ctx: Context) -> None:
        self._pool = await connect()

    async def stop(self) -> None:
        await self._pool.close()
```

`start` runs before the first run; `stop` runs on shutdown. The easiest way to
guarantee cleanup is to use the harness as a context manager:

```python
async with harness:        # start on enter, stop on exit
    await harness.run("hello")
```

## Removing a plugin

`harness.unuse(plugin)` removes a plugin and everything it registered (its event
subscriptions and interceptors), and stops it if it has a lifecycle. Removing a
plugin that was never registered does nothing.
