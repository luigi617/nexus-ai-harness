# Nexus AI Harness

A plugin-based harness for building LLM agents.

Everything an agent needs is a **plugin**, including agent loop, model, context management, permission, memory, subagent, and tools. Plugins are registered on a harness and resolved by their protocol
type, so you compose an agent by picking the pieces you want.

## Example Usage

Using the default harness
```python
from plugins import default_harness
from plugins.models import BedrockModel

harness = default_harness(BedrockModel(model="..."))
answer = harness.run_sync("What is 128 * 47?")  # or: await harness.run(...)
```

Or build your own:

```python
from harness import NexusAIHarness
from plugins.loops import AgenticLoop

harness = (
    NexusAIHarness()
    .use(AgenticLoop())
    .use(BedrockModel(model="..."))
    .use(...)
)
```

## Documentation

See [docs/](docs/README.md) for architecture, plugin guides, and how to write your own.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License
MIT