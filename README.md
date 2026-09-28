# Nexus AI Harness

Build LLM agents by composing the pieces you want.

Every part of an agent — the loop, model, memory, tools, permissions, subagents,
and more — is a **plugin**. You pick the plugins you need and register them on a
harness. Swap any piece without touching the rest.

## Install

```bash
pip install nexus-ai-harness
```

Set the API key for the model you'll use (copy `.env.example` to `.env` and fill
it in). For Anthropic that's `ANTHROPIC_API_KEY`; see [Models](docs/plugins/models.md)
for other providers.

## Usage

Start with the batteries-included harness:

```python
from nexus_ai_harness.plugins import default_harness
from nexus_ai_harness.plugins.models import AnthropicModel

harness = default_harness(AnthropicModel(model="..."))
answer = harness.run_sync("What is 128 * 47?")   # or: await harness.run(...)
```

Or compose your own:

```python
from nexus_ai_harness.harness import NexusAIHarness
from nexus_ai_harness.plugins.loops import AgenticLoop
from nexus_ai_harness.plugins.models import AnthropicModel

harness = (
    NexusAIHarness()
    .use(AgenticLoop())
    .use(AnthropicModel(model="..."))
    .use(...)                       # add the plugins you want
)
```

## Documentation

See [docs/](docs/README.md) for the architecture, the plugin guides, and how to
write your own plugin.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
