# Memory

Memory lets an agent save notes and recall them in later sessions. Register a
memory store plus the memory tools, and the agent can use them on its own.

```python
from nexus_ai_harness.plugins.memory import FileMemoryStore
from nexus_ai_harness.plugins.tools import Remember, Recall, Forget

harness.use(FileMemoryStore())          # defaults to ~/.nexus-ai-harness/memory
harness.use(Remember()).use(Recall()).use(Forget())
```

Pass a path to store memories elsewhere: `FileMemoryStore("./my-memory")`.

## The tools

- **`remember`**: save a note.
- **`recall`**: search saved notes.
- **`forget`**: delete a note.

The default harness includes memory already, so you only wire it up when
building your own.
