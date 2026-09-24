from __future__ import annotations

from typing import ClassVar

from plugins.models.openai_compatible import OpenAICompatibleModel


class QwenModel(OpenAICompatibleModel):
    provider = "qwen"
    # DashScope OpenAI-compatible mode (international endpoint).
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key_env = "DASHSCOPE_API_KEY"
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "deepseek-v4-flash-0731": (0.2, 0.4),
        "glm-5.2": (1.4, 4.4),
        "kimi-k3": (3.0, 15.0),
        "qvq-max": (1.2, 4.8),
        "qwen-flash": (0.05, 0.4),
        "qwen-max": (1.6, 6.4),
        "qwen-plus": (0.4, 1.2),
        "qwen-plus-character-ja": (0.5, 1.4),
        "qwen-turbo": (0.05, 0.2),
        "qwen-vl-max": (0.8, 3.2),
        "qwen-vl-plus": (0.21, 0.63),
        "qwen2-5-14b-instruct": (0.35, 1.4),
        "qwen2-5-32b-instruct": (0.7, 2.8),
        "qwen2-5-72b-instruct": (1.4, 5.6),
        "qwen2-5-7b-instruct": (0.175, 0.7),
        "qwen2-5-vl-72b-instruct": (2.8, 8.4),
        "qwen2-5-vl-7b-instruct": (0.35, 1.05),
        "qwen3-14b": (0.35, 1.4),
        "qwen3-235b-a22b": (0.7, 2.8),
        "qwen3-32b": (0.7, 2.8),
        "qwen3-8b": (0.18, 0.7),
        "qwen3-coder-30b-a3b-instruct": (0.45, 2.25),
        "qwen3-coder-480b-a35b-instruct": (1.5, 7.5),
        "qwen3-coder-flash": (0.3, 1.5),
        "qwen3-coder-plus": (1.0, 5.0),
        "qwen3-max": (1.2, 6.0),
        "qwen3-next-80b-a3b-instruct": (0.5, 2.0),
        "qwen3-next-80b-a3b-thinking": (0.5, 6.0),
        "qwen3-vl-235b-a22b": (0.7, 2.8),
        "qwen3-vl-30b-a3b": (0.2, 0.8),
        "qwen3-vl-plus": (0.2, 1.6),
        "qwen3.5-122b-a10b": (0.4, 3.2),
        "qwen3.5-27b": (0.3, 2.4),
        "qwen3.5-35b-a3b": (0.25, 2.0),
        "qwen3.5-397b-a17b": (0.6, 3.6),
        "qwen3.5-plus": (0.4, 2.4),
        "qwen3.6-27b": (0.6, 3.6),
        "qwen3.6-35b-a3b": (0.248, 1.485),
        "qwen3.6-flash": (0.1875, 1.125),
        "qwen3.6-max-preview": (1.3, 7.8),
        "qwen3.6-plus": (0.5, 3.0),
        "qwen3.7-max": (2.5, 7.5),
        "qwen3.7-plus": (0.5, 3.0),
        "qwen3.8-flash": (0.15, 0.47),
        "qwen3.8-max": (2.0, 6.0),
        "qwq-plus": (0.8, 2.4),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "deepseek-v4-flash-0731": "Official DeepSeek V4 Flash release with enhanced agentic capabilities and integrated DSpark speculative decoding",  # noqa: E501
        "glm-5.2": "Open flagship GLM for long-horizon coding agents and million-token context work",  # noqa: E501
        "kimi-k3": "Multimodal Kimi model with 1M context and toggleable max-effort thinking for long-horizon agent work",  # noqa: E501
        "qvq-max": "Qwen vision-language model for visual reasoning, documents, and agent tasks",  # noqa: E501
        "qwen-flash": "Efficient Qwen model for fast chat, extraction, and high-volume",
        "qwen-max": "Flagship Qwen model for complex reasoning, coding, and agentic wo",
        "qwen-plus": "Qwen instruction model for multilingual chat, reasoning, and too",
        "qwen-plus-character-ja": "Qwen instruction model for multilingual chat, reaso",
        "qwen-turbo": "Efficient Qwen model for fast chat, extraction, and high-volume",
        "qwen-vl-max": "Qwen vision-language model for visual reasoning, documents, an",
        "qwen-vl-plus": "Qwen vision-language model for visual reasoning, documents, a",
        "qwen2-5-14b-instruct": "Qwen instruction model for multilingual chat, reasoni",
        "qwen2-5-32b-instruct": "Qwen instruction model for multilingual chat, reasoni",
        "qwen2-5-72b-instruct": "Qwen instruction model for multilingual chat, reasoni",
        "qwen2-5-7b-instruct": "Qwen instruction model for multilingual chat, reasonin",
        "qwen2-5-vl-72b-instruct": "Qwen vision-language model for visual reasoning, d",
        "qwen2-5-vl-7b-instruct": "Qwen vision-language model for visual reasoning, do",
        "qwen3-14b": "Qwen instruction model for multilingual chat, reasoning, and too",
        "qwen3-235b-a22b": "Large open Qwen MoE for multilingual reasoning, coding, an",
        "qwen3-32b": "Dense open Qwen model for self-hosted chat, reasoning, and codin",
        "qwen3-8b": "Qwen instruction model for multilingual chat, reasoning, and tool",
        "qwen3-coder-30b-a3b-instruct": "Smaller Qwen coder for efficient local agents",
        "qwen3-coder-480b-a35b-instruct": "Open Qwen coding heavyweight for repository",
        "qwen3-coder-flash": "Qwen coding model for software agents, repository edits,",
        "qwen3-coder-plus": "Hosted Qwen coder for software agents, repo edits, and lo",
        "qwen3-max": "Flagship Qwen3 model for coding agents, complex reasoning, and t",
        "qwen3-next-80b-a3b-instruct": "Qwen instruction model for multilingual chat,",
        "qwen3-next-80b-a3b-thinking": "Efficient Qwen thinking model for local reason",
        "qwen3-vl-235b-a22b": "Qwen vision-language model for visual reasoning, docume",
        "qwen3-vl-30b-a3b": "Qwen vision-language model for visual reasoning, document",
        "qwen3-vl-plus": "Qwen vision-language model for visual reasoning, documents,",
        "qwen3.5-122b-a10b": "Qwen vision-language model for visual reasoning, documen",
        "qwen3.5-27b": "Qwen vision-language model for visual reasoning, documents, an",
        "qwen3.5-35b-a3b": "Qwen vision-language model for visual reasoning, documents",
        "qwen3.5-397b-a17b": "Large open Qwen multimodal MoE for visual agents and lon",
        "qwen3.5-plus": "Qwen vision-language model for visual reasoning, documents, a",
        "qwen3.6-27b": "Qwen vision-language model for visual reasoning, documents, an",
        "qwen3.6-35b-a3b": "Open multimodal Qwen MoE for local agents that need vision",
        "qwen3.6-flash": "Qwen vision-language model for visual reasoning, documents,",
        "qwen3.6-max-preview": "Flagship Qwen model for complex reasoning, coding, and",
        "qwen3.6-plus": "Earlier Qwen multimodal workhorse for million-token agent and",
        "qwen3.7-max": "Qwen frontier model tuned for agent frameworks, coding assista",
        "qwen3.7-plus": "Multimodal Qwen workhorse for long-context agents, visual inp",
        "qwen3.8-flash": "Qwen vision-language model for visual reasoning, documents,",
        "qwen3.8-max": "2.4-trillion-parameter MoE flagship for coding, professional w",
        "qwq-plus": "Qwen reasoning model for deliberate problem solving, math, and coding",  # noqa: E501
    }
