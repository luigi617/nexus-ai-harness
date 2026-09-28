# Nexus AI Harness

Build LLM agents by composing the pieces you want.

Every part of an agent is a **plugin**: the loop, model, memory, tools,
permissions, subagents, and more. You pick the plugins you need and register them
on a harness, and you can swap any piece without touching the rest.

## Why Nexus AI Harness?

- **Start in one line.** A batteries-included harness gives you a working agent
  right away.
- **Compose, don't configure.** Build an agent by adding the plugins you want.
- **Swap any part.** Change the model, memory, or tools without rewriting the
  rest.
- **Many models, one interface.** Anthropic, OpenAI, Gemini, Bedrock, and more,
  all called the same way.
- **Extend without forking.** Add your own plugin by implementing an interface.
- **Safe by default.** Built-in permissions, sandboxing, and spend and time
  limits.
- **Ready for real work.** Memory, subagents, skills, MCP tools, and session
  resume come included.

## Install

```bash
pip install nexus-ai-harness
```

Set the API key for the model you'll use (copy `.env.example` to `.env` and fill
it in). For Anthropic that's `ANTHROPIC_API_KEY`. See [Models](docs/plugins/models.md)
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
