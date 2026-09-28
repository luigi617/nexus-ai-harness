# Interventions

Interventions let you steer or stop an agent while it's running. They take effect
at a safe point — between iterations of the loop — never in the middle of a step.

```python
import asyncio
from nexus_ai_harness.harness import Session
from nexus_ai_harness.plugins.interventions import InjectMessage

session = Session()
task = asyncio.create_task(harness.run("do the big task", session=session))

session.submit(InjectMessage("prioritize correctness over speed"))  # steer the next turn
session.interrupt()                                                 # stop the run
```

## Two ways to intervene

- **Inject a message** — add guidance the agent reads on its next turn, without
  stopping it. Applies to the session you submit it to.
- **Interrupt** — stop the run. Interrupting the main agent also stops any
  subagents it spawned.
