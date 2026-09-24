from __future__ import annotations

from typing import ClassVar

from plugins.models.openai_compatible import OpenAICompatibleModel


class DeepSeekModel(OpenAICompatibleModel):
    provider = "deepseek"
    base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"
    descriptions: ClassVar[dict[str, str]] = {
        "deepseek-chat": "general-purpose chat model",
        "deepseek-flash": "DeepSeek V4.1 Flash model for reasoning and agentic coding",
        "deepseek-reasoner": "reasoning-optimized model",
        "deepseek-v4-flash": "DeepSeek V4.1 Flash model for reasoning and agentic codi",
        "deepseek-v4-flash-vision-exp": "DeepSeek V4.1 Flash model for reasoning and a",
        "deepseek-v4-pro": "DeepSeek V4 Pro snapshot with million-token context and su",
    }
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "deepseek-flash": (0.15, 0.6),
        "deepseek-v4-flash": (0.15, 0.6),
        "deepseek-v4-flash-vision-exp": (0.15, 0.6),
        "deepseek-v4-pro": (0.435, 0.87),
    }
