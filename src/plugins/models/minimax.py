from __future__ import annotations

from plugins.models.openai_compatible import OpenAICompatibleModel


class MiniMaxModel(OpenAICompatibleModel):
    provider = "minimax"
    base_url = "https://api.minimax.chat/v1"
    api_key_env = "MINIMAX_API_KEY"
