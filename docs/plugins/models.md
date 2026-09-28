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

## Retries and errors

Every backend retries transient failures itself: rate limits (429), 408, 409,
most 5xx responses, dropped connections, and timeouts. Waits grow exponentially
with jitter, and a provider's `Retry-After` header is honored. The default is 3
retries; pass `max_retries` to change it, or `retry` for full control:

```python
from nexus_ai_harness.plugins.models import AnthropicModel, RetryPolicy

AnthropicModel(model="...", max_retries=0)  # fail fast
AnthropicModel(model="...", retry=RetryPolicy(backoff_base=0.5, backoff_cap=10))
```

`BedrockModel` hands retrying to botocore's `standard` mode and uses only
`max_retries` from the policy.

A request that still fails raises a typed error from `nexus_ai_harness.core.errors`.
Each one is a `RuntimeError`, so existing `except RuntimeError` code keeps
working:

| Error | Meaning |
|---|---|
| `ModelAPIError` | The provider rejected the request; has `status`, `body`, `retryable`, `retry_after`, `attempts`. |
| `RateLimitError` | The provider throttled the request. |
| `ContextLengthExceeded` | The prompt did not fit in the model's context window. Never retried. |

Connection failures and timeouts that outlast the retries are re-raised
unchanged (`urllib.error.URLError`, `TimeoutError`).

The built-in loops don't let a failed model call crash the run. They emit a
`ModelCallFailed` event and end the run with `stop_reason == "model_error"`, the
same way a guard stop ends it. The output is `"stopped: model error: ..."`.

## Cost tracking

Each backend ships indicative per-token pricing, so hooks like the cost counter
can report the estimated USD cost of a run.
