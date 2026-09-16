# Writing a plugin

A plugin is any class that implements a protocol from `protocols/` and is
registered on a harness with `.use(...)`.

## Steps

1. Pick (or define) a protocol in `protocols/`.
2. Implement it in `plugins/`. The method may be `def` or `async def`.
3. Register it: `harness.use(MyPlugin())`.
4. Consumers resolve it by type: `ctx.get(MyProtocol)` / `ctx.all(MyProtocol)`.

<!-- TODO:
- Minimal worked example (e.g. a Tool).
- The `kind` ClassVar and why it keys the registry.
- Reading/writing per-session state via `ctx.state(cls)`.
- Emitting events with `ctx.emit(...)`.
- Testing a plugin (see tests/conftest.py: make_ctx, ScriptedModel, RecordingTool).
-->
