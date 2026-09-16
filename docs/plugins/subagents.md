# Subagents

The `Subagent` tool delegates a task to a fresh agent run with an isolated
context. It runs on a `Spawner` (protocol: `protocols/spawner.py`); the default
is `InProcessSpawner`.

<!-- TODO:
- fork(): fresh Session, shared registry subset, depth + 1, inherited approver.
- Depth limit (default 2) lives on the spawner.
- plugins=None inherits all parent plugins; a list restricts the subset.
- Concurrency / thread-safety notes.
-->
