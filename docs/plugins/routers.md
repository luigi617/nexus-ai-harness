# Routers

A `Router` (protocol: `protocols/router.py`) picks which `Model` handles a turn,
choosing among the candidates from `ctx.all(Model)`. When a router is registered
the loop calls it each turn instead of `ctx.get(Model)`.

<!-- TODO:
- LLMRouter: an LLM decides which model, matched on `provider$name` or `name`.
- StickyRouter: decide once on the first task, reuse the choice thereafter.
-->
