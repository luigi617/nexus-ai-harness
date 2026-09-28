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
watch them — useful for logging, tracing, or metrics:

```python
class Reporter(Hook):
    def on(self, event: Event, ctx: Context) -> None:
        if isinstance(event, ToolCallCompleted):
            log(ctx.session_id, event.result.content)
```

Register it with `.use(Reporter())`. See `core/events.py` for the full list of
events.

### Run code before or after another plugin with an `Interceptor`

An interceptor wraps every call to a plugin type. Set `target` to a protocol to
wrap all of its implementers, or to a concrete class to wrap just that one:

```python
class TimeModel(Interceptor):
    target = Model                     # wrap every model call
    def before(self, ctx: Context) -> None: ...
    def after(self, ctx: Context) -> None: ...   # always runs, even on error
```

`before` and `after` are for observation — they don't see the call's arguments
or return value. Both may be `def` or `async def`.

## Owning resources with a lifecycle

If your plugin holds a resource — a connection, client, or background task —
subclass `Lifecycle` to open it once and close it reliably:

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

`harness.unuse(plugin)` removes a plugin and everything it registered — its event
subscriptions and interceptors — and stops it if it has a lifecycle. Removing a
plugin that was never registered does nothing.
