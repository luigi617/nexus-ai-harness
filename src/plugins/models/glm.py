from __future__ import annotations

from typing import ClassVar

from plugins.models.openai_compatible import OpenAICompatibleModel


class GLMModel(OpenAICompatibleModel):
    provider = "glm"
    # Zhipu AI (GLM) OpenAI-compatible endpoint.
    base_url = "https://open.bigmodel.cn/api/paas/v4"
    api_key_env = "ZHIPUAI_API_KEY"
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "glm-4.5": (0.6, 2.2),
        "glm-4.5-air": (0.2, 1.1),
        "glm-4.5-flash": (0.0, 0.0),
        "glm-4.5v": (0.6, 1.8),
        "glm-4.6": (0.6, 2.2),
        "glm-4.6v": (0.3, 0.9),
        "glm-4.6v-flash": (0.0, 0.0),
        "glm-4.7": (0.6, 2.2),
        "glm-4.7-flash": (0.0, 0.0),
        "glm-4.7-flashx": (0.07, 0.4),
        "glm-5": (1.0, 3.2),
        "glm-5.1": (1.4, 4.4),
        "glm-5.2": (1.4, 4.4),
        "glm-5.3": (1.4, 4.4),
        "glm-5.3-flash": (0.15, 0.5),
        "glm-5.3-flashx": (0.37, 1.25),
        "glm-5v-turbo": (5.0, 22.0),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "glm-4.5": "Hybrid-reasoning GLM release that made the 4.5 line broadly useful",
        "glm-4.5-air": "Lighter GLM-4.5 variant for fast coding assistance and cheaper",
        "glm-4.5-flash": "Efficient GLM model for fast reasoning, coding, and agent wo",
        "glm-4.5v": "GLM vision model for visual reasoning, documents, and multimodal",
        "glm-4.6": "Late GLM-4 workhorse for coding agents, reasoning, and structured",
        "glm-4.6v": "GLM vision model for visual reasoning, documents, and multimodal",
        "glm-4.6v-flash": "Lightweight GLM vision model for visual reasoning, document",
        "glm-4.7": "Mature GLM model for dependable coding, reasoning, and structured",
        "glm-4.7-flash": "Budget GLM lane for fast coding help, routing, and everyday",
        "glm-4.7-flashx": "Efficient GLM model for fast reasoning, coding, and agent w",
        "glm-5": "General GLM flagship for coding, analysis, and tool-heavy engineerin",
        "glm-5.1": "Strong GLM coding model for agentic engineering, terminals, and re",
        "glm-5.2": "Open flagship GLM for long-horizon coding agents and million-token",
        "glm-5.3": "Flagship GLM model for long-horizon coding, agents, and complex pr",
        "glm-5.3-flash": "Native multimodal GLM model for efficient coding and long-ho",
        "glm-5.3-flashx": "High-speed GLM-5.3-Flash serving option for coding and agen",
        "glm-5v-turbo": "Fast GLM vision model for screenshots, documents, and multimo",
    }
