# Documentation

- [Architecture](architecture.md): how the harness fits together.
- [Writing a plugin](writing-a-plugin.md): build and register your own.
- [Code style](code-style.md): conventions for contributors.

## Plugins

- [Models](plugins/models.md): the LLM backends you can call.
- [Routers](plugins/routers.md): pick which model handles each turn.
- [Memory](plugins/memory.md): remember and recall across sessions.
- [Subagents](plugins/subagents.md): delegate a task to a fresh agent.
- [Skills](plugins/skills.md): load task-specific instructions on demand.
- [Evaluators](plugins/evaluators.md): structured scoring of agent runs.
- [Interventions](plugins/interventions.md): steer or stop a running agent.
- [Coding tools](plugins/coding-tools.md): read, search, edit, and run code in a
  workspace.

Other built-in plugins:

- **Sandbox**: confine tool file access and commands to a workspace folder.
- **MCP**: connect to MCP servers and use their tools.
- **Persistence**: save a session and resume it later.
