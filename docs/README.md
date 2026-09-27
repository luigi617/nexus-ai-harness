# Documentation

- [Architecture](architecture.md) — how the harness is put together.
- [Writing a plugin](writing-a-plugin.md) — implement a protocol and register it.
- [Code style](code-style.md) — coding conventions for the repo.

## Plugins

- [Models](plugins/models.md)
- [Routers](plugins/routers.md)
- [Memory](plugins/memory.md)
- [Subagents](plugins/subagents.md)
- [Skills](plugins/skills.md)
- [Interventions](plugins/interventions.md)
- Sandbox — confine tool file access and subprocesses to a workspace root.
- Filesystem/shell tools — `read_file`, `write_file`, `list_dir`, and `shell`.
- MCP — connect to MCP servers and expose their tools to the loop.
- Persistence — save session snapshots and resume a run later.
