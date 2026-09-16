# Architecture

How the harness is put together.

## Everything is a plugin

An agent is composed by registering plugins on a harness with `.use(...)`. Each
plugin implements a **protocol** and is resolved by its protocol type, not by name.

<!-- TODO:
- kind-based registry: how `.use()` stores plugins and `ctx.get(P)` / `ctx.all(P)`
  resolve them by `P.kind`.
- type-keyed resolution and the bounded TypeVar `P = TypeVar("P", bound=Plugin)`.
-->

## Layering

Dependencies flow one way: `core ← protocols ← {services, plugins, harness}`.

<!-- TODO:
- `core/`      — plain data types (Message, Response, RunState, ...).
- `protocols/` — runtime_checkable Protocols; the only thing consumers import.
- `plugins/`   — concrete implementations of the protocols.
- `services/`  — shared logic reused across plugins/harness (runner, guard chain).
- `harness/`   — Session, RunContext (the Context), Registry, the harness itself.
- Why harness does NOT depend on plugins.
-->

## Sync or async, one method

Plugin methods may be written `def` or `async def`; the harness adapts via
`core/invoke.py`.

<!-- TODO: explain invoke(fn, *args): await if coroutine, else asyncio.to_thread. -->

## Per-session state

`ctx.state(cls)` returns a lazily-created, type-keyed dataclass scoped to the
session — how plugins keep counters/settings without an untyped bag.

<!-- TODO: example. -->
