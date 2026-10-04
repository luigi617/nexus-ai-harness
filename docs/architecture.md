# Architecture

The harness is a small core that composes plugins. This page covers the ideas
you need to use it and to write your own plugin.

## Everything is a plugin

You build an agent by registering plugins on a harness with `.use(...)`:

```python
harness = NexusAIHarness().use(AgenticLoop()).use(AnthropicModel(model="..."))
```

Each plugin implements a **protocol** (an interface such as `Model`, `Tool`, or
`Memory`). Plugins are resolved by that protocol, not by name, so you can swap
one implementation for another without changing the code that uses it.

## Subagents run in their own context

A subagent runs on a **forked** context: a fresh, isolated session that inherits
the parent's plugins. You choose what it inherits:

```python
child = ctx.fork()                                     # inherit everything
child = ctx.fork(plugins=[search_tool])                # only these plugins
child = ctx.fork(overrides={Memory: AgentMemory()})    # swap one plugin
```

This lets a fleet of agents share the same model, permissions, and telemetry
while keeping their own memory or tools. Subagent work stays isolated from the
parent, but stopping the parent also stops its subagents.

## Sync or async, your choice

Any plugin method may be written as `def` or `async def`; the harness runs it
correctly either way. Use `harness.run(...)` from async code or
`harness.run_sync(...)` from sync code.

## Per-session state

A plugin can keep state that lives for one session with `ctx.state(cls)`, for
example a token counter or a per-run setting. State is scoped to the session, so
it never leaks between runs. State isn't saved with the session unless its class
opts in with `@persistable`; see [Persistence](plugins/persistence.md).
