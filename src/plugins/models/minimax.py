from __future__ import annotations

from typing import ClassVar

from plugins.models.openai_compatible import OpenAICompatibleModel


class MiniMaxModel(OpenAICompatibleModel):
    provider = "minimax"
    base_url = "https://api.minimax.chat/v1"
    api_key_env = "MINIMAX_API_KEY"
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "MiniMax-M2": (0.3, 1.2),
        "MiniMax-M2.1": (0.3, 1.2),
        "MiniMax-M2.5": (0.3, 1.2),
        "MiniMax-M2.5-highspeed": (0.6, 2.4),
        "MiniMax-M2.7": (0.3, 1.2),
        "MiniMax-M2.7-highspeed": (0.6, 2.4),
        "MiniMax-M3": (0.3, 1.2),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "MiniMax-M2": "Efficient open MiniMax model built for coding agents and tool-h",
        "MiniMax-M2.1": "Earlier MiniMax agent model for practical coding and producti",
        "MiniMax-M2.5": "Prior MiniMax coding model for agent workflows, office edits,",
        "MiniMax-M2.5-highspeed": "High-speed MiniMax model for low-latency coding and",
        "MiniMax-M2.7": "Open MiniMax flagship for coding agents, office automation, a",
        "MiniMax-M2.7-highspeed": "Low-latency M2.7 variant for interactive coding pla",
        "MiniMax-M3": "MiniMax multimodal model for long-context coding, perception, a",
    }
