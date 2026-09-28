# Models

A model is the LLM backend the agent calls. Pick one, give it a model name, and
register it:

```python
from nexus_ai_harness.plugins.models import AnthropicModel

harness.use(AnthropicModel(model="..."))
```

Register more than one model and add a [router](routers.md) to choose between
them per turn.

## Available backends

Each backend reads its API key from an environment variable (set it in `.env`):

| Class | Provider | API key env var |
|---|---|---|
| `BedrockModel` | Amazon Bedrock | `AWS_BEARER_TOKEN_BEDROCK` |
| `AnthropicModel` | Anthropic | `ANTHROPIC_API_KEY` |
| `OpenAIModel` | OpenAI | `OPENAI_API_KEY` |
| `GeminiModel` | Google Gemini | `GEMINI_API_KEY` |
| `GroqModel` | Groq | `GROQ_API_KEY` |
| `XAIModel` | xAI | `XAI_API_KEY` |
| `DeepSeekModel` | DeepSeek | `DEEPSEEK_API_KEY` |
| `MiniMaxModel` | MiniMax | `MINIMAX_API_KEY` |
| `QwenModel` | Qwen | `DASHSCOPE_API_KEY` |
| `GLMModel` | GLM | `ZHIPUAI_API_KEY` |

To reach any OpenAI-compatible endpoint not listed above, use
`OpenAICompatibleModel`.

## Cost tracking

Each backend ships indicative per-token pricing, so hooks like the cost counter
can report the estimated USD cost of a run.
