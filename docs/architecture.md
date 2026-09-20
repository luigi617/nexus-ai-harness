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
