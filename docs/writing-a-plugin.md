# Writing a plugin

A plugin is any class that implements a protocol from `protocols/` and is
registered on a harness with `.use(...)`.

## Steps

1. Pick (or define) a protocol in `protocols/`.
2. Implement it in `plugins/`. The method may be `def` or `async def`.
3. Register it: `harness.use(MyPlugin())`.
4. Consumers resolve it by type: `ctx.get(MyProtocol)` / `ctx.all(MyProtocol)`.

## Declaring dependencies

A plugin can declare the protocols it depends on with a `requires` class var so
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

<!-- TODO:
- Minimal worked example (e.g. a Tool).
- The `kind` ClassVar and why it keys the registry.
- Reading/writing per-session state via `ctx.state(cls)`.
- Emitting events with `ctx.emit(...)`.
- Testing a plugin (see tests/conftest.py: make_ctx, ScriptedModel, RecordingTool).
-->
