from __future__ import annotations

from plugins.models import (
    AnthropicModel,
    BedrockModel,
    DeepSeekModel,
    GeminiModel,
    GLMModel,
    GroqModel,
    MiniMaxModel,
    OpenAICompatibleModel,
    OpenAIModel,
    QwenModel,
    XAIModel,
)
from protocols.model import Model

_PROVIDERS: dict[str, type[Model]] = {
    "bedrock": BedrockModel,
    "anthropic": AnthropicModel,
    "openai": OpenAIModel,
    "openai_compatible": OpenAICompatibleModel,
    "deepseek": DeepSeekModel,
    "gemini": GeminiModel,
    "glm": GLMModel,
    "groq": GroqModel,
    "minimax": MiniMaxModel,
    "qwen": QwenModel,
    "xai": XAIModel,
}


def build_model(spec: str, *, max_tokens: int = 1024) -> Model:
    """Construct a Model from ``"provider:model-id"`` (defaults to bedrock).

    Args:
        spec: A ``provider:model-id`` string; a bare id assumes the bedrock
            provider.
        max_tokens: Max output tokens passed to the model backend.

    Returns:
        An instance of the backend for the named provider.

    Raises:
        ValueError: If the provider prefix is not a known backend.
    """
    provider, _, model_id = spec.partition(":")
    if not model_id:  # a bare id has no provider prefix
        provider, model_id = "bedrock", spec
    if provider not in _PROVIDERS:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"unknown provider {provider!r}; known: {known}")
    return _PROVIDERS[provider](model=model_id, max_tokens=max_tokens)
