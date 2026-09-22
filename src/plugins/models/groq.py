from __future__ import annotations

from plugins.models.openai_compatible import OpenAICompatibleModel


class GroqModel(OpenAICompatibleModel):
    provider = "groq"
    base_url = "https://api.groq.com/openai/v1"
    api_key_env = "GROQ_API_KEY"
