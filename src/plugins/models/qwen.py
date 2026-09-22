from __future__ import annotations

from plugins.models.openai_compatible import OpenAICompatibleModel


class QwenModel(OpenAICompatibleModel):
    provider = "qwen"
    # DashScope OpenAI-compatible mode (international endpoint).
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env = "DASHSCOPE_API_KEY"
