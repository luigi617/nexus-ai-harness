# Interventions

External control over a running agent, applied at a safe checkpoint (before the
loop's next iteration) — never mid-flight.

## Two mechanisms, different scopes

| | Scope | How |
|---|---|---|
| **Interrupt** | Whole tree (root → subagents) | Shared signal; `fork` passes it to children; every loop checks it before its next iteration. |
| **Interventions** (`InjectMessage`, …) | Root session only | Per-session inbox; `fork` gives children a fresh empty one; the loop drains and applies each. |

```python
session = Session()
task = asyncio.create_task(harness.run("do the big task", session=session))
session.submit(InjectMessage("prioritize correctness over speed"))  # steer next turn
session.interrupt()                                                  # or stop the whole tree
```

<!-- TODO:
- The Intervention protocol: apply(ctx), sync or async; not a registered plugin.
- Session.interrupt() / clear_interrupt() / interrupted; submit() / take_interventions().
- Adding a new intervention type (Pause, SwapModel, ...): just implement apply(ctx).
-->
