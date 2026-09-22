from __future__ import annotations

from typing import ClassVar

from plugins.models.openai_compatible import OpenAICompatibleModel


class OpenAIModel(OpenAICompatibleModel):
    provider = "openai"
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "gpt-4o": (2.5, 10.0),
        "gpt-4o-mini": (0.15, 0.6),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "gpt-4o": "capable general-purpose model",
        "gpt-4o-mini": "fast and cheap; simple tasks",
    }
