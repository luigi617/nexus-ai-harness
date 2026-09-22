from __future__ import annotations

from typing import ClassVar

from plugins.models.openai_compatible import OpenAICompatibleModel


class XAIModel(OpenAICompatibleModel):
    provider = "xai"
    base_url = "https://api.x.ai/v1"
    api_key_env = "XAI_API_KEY"
    descriptions: ClassVar[dict[str, str]] = {
        "grok-4": "xAI flagship reasoning model",
    }
