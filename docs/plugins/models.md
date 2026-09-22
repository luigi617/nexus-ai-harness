# Models

A `Model` (protocol: `protocols/model.py`) turns a message history into a
`Response`. It is identified for routing by `provider` and `name`, and
`complete(history, ctx)` may be sync or `async`.

## Available backends

| Class | Provider | API key env var |
|---|---|---|
| `BedrockModel` | `bedrock` | `AWS_BEARER_TOKEN_BEDROCK` |
| `AnthropicModel` | `anthropic` | `ANTHROPIC_API_KEY` |
| `OpenAIModel` | `openai` | `OPENAI_API_KEY` |
| `GeminiModel` | `gemini` | `GEMINI_API_KEY` |
| `GroqModel` | `groq` | `GROQ_API_KEY` |
| `XAIModel` | `xai` | `XAI_API_KEY` |
| `DeepSeekModel` | `deepseek` | `DEEPSEEK_API_KEY` |
| `MiniMaxModel` | `minimax` | `MINIMAX_API_KEY` |
| `QwenModel` | `qwen` | `DASHSCOPE_API_KEY` |
| `GLMModel` | `glm` | `ZHIPUAI_API_KEY` |



## Cost

Each backend has an indicative `pricing` table (`{model: (input, output)}`,
USD per 1M tokens);