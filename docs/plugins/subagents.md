# Subagents

The `Subagent` tool lets an agent hand off a task to a fresh, isolated agent and
get back just the result. This keeps the main conversation focused while the
subagent does the detailed work.

```python
from nexus_ai_harness.plugins.tools import Subagent
from nexus_ai_harness.plugins.spawner import InProcessSpawner

harness.use(Subagent()).use(InProcessSpawner())
```

The subagent inherits the parent's plugins (model, tools, permissions) but runs
in its own session. To limit how deep subagents can nest (a subagent spawning
another subagent), set `InProcessSpawner(max_depth=2)`. Use `max_concurrent` to
cap how many run at once.

The default harness already includes subagents.
