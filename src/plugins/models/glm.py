from __future__ import annotations

from plugins.models.openai_compatible import OpenAICompatibleModel


class GLMModel(OpenAICompatibleModel):
    provider = "glm"
    # Zhipu AI (GLM) OpenAI-compatible endpoint.
    base_url = "https://open.bigmodel.cn/api/paas/v4"
    api_key_env = "ZHIPUAI_API_KEY"
