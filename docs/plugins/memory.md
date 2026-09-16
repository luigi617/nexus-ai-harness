# Memory

Long-term memory is a `MemoryStore` (protocol: `protocols/memory.py`) plus the
tools that use it. Tools reach the store through the context, never via `__init__`.

<!-- TODO:
- MemoryStore protocol and FileMemoryStore / FileMemoryItem.
- The tools: remember (upsert), recall (search), forget.
- Where the store is registered (default_harness) and the memory_dir.
-->
