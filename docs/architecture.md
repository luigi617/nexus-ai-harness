# Architecture

How the harness is put together.

## Everything is a plugin

An agent is composed by registering plugins on a harness with `.use(...)`. Each
plugin subclasses an **abstract base** in `protocols/` and is resolved by that
type, not by name.

<!-- TODO:
- type-based registry: how `.use()` stores plugins and `ctx.get(P)` / `ctx.all(P)`
  resolve them via `isinstance(plugin, P)`.
- type-keyed resolution and the bounded TypeVar `P = TypeVar("P", bound=Plugin)`.
-->

## Forking for subagents

A subagent runs on a **forked** context: `ctx.fork()` builds a child on a fresh
session that inherits the parent's plugins. Two arguments shape what it sees,
which is how a fleet of agents shares infrastructure while differing where it
matters (the issue that motivated this: shared model/permissions/telemetry,
per-agent memory):

```python
child = ctx.fork()                                    # inherit everything
child = ctx.fork(plugins=[search_tool])               # restrict to exactly these
child = ctx.fork(overrides={MemoryStore: AgentMemory()})  # swap every memory store
child = ctx.fork(overrides={FileMemoryStore: AgentMemory()})  # swap just that one
```

`overrides` is a `target -> replacement` mapping. Each target is matched by
`isinstance`, so a protocol base (`MemoryStore`) swaps out every implementer
while a concrete class (`FileMemoryStore`) swaps only that one; fork drops the
matches and registers the replacement in their place — a swap, not an append.
Matching by type (rather than trusting `get()` to pick the last-registered one)
is what makes it correct for protocols a harness holds several of, like tools or
hooks. The child's session is isolated; only the interrupt signal is shared, so
interrupting the root stops its subagents.

## Layering

Dependencies flow one way: `core ← protocols ← {services, plugins, harness}`.

<!-- TODO:
- `core/`      — plain data types (Message, Response, RunState, ...).
- `protocols/` — abstract base classes (the interfaces consumers import), plus
  `MemoryItem`, the base `@dataclass` a store's item type extends.
- `plugins/`   — concrete implementations of the protocols.
- `services/`  — shared logic reused across plugins/harness (runner, guard chain).
- `harness/`   — Session, RunContext (the Context), Registry, the harness itself.
- Why harness does NOT depend on plugins.
-->

## Sync or async, one method

Plugin methods may be written `def` or `async def`; the harness adapts. Call
one through `ctx.invoke(fn, *args)`, which runs any interceptors bound to the
plugin `fn` belongs to (inferred from `fn`, so binding to a concrete class wraps
only that class) around the call. It builds on `call(fn, *args)` in
`core/invoke.py`, the raw sync/async adapter (await if a coroutine, else
`asyncio.to_thread`).

<!-- TODO: explain interceptor dispatch. -->


## Per-session state

`ctx.state(cls)` returns a lazily-created, type-keyed dataclass scoped to the
session — how plugins keep counters/settings without an untyped bag.

<!-- TODO: example. -->
