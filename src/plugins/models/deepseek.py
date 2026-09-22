from __future__ import annotations

from typing import ClassVar

from plugins.models.openai_compatible import OpenAICompatibleModel


class DeepSeekModel(OpenAICompatibleModel):
    provider = "deepseek"
    base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"
    descriptions: ClassVar[dict[str, str]] = {
        "deepseek-chat": "general-purpose chat model",
        "deepseek-reasoner": "reasoning-optimized model",
    }
