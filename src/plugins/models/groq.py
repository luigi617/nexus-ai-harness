from __future__ import annotations

from typing import ClassVar

from plugins.models.openai_compatible import OpenAICompatibleModel


class GroqModel(OpenAICompatibleModel):
    provider = "groq"
    base_url = "https://api.groq.com/openai/v1"
    api_key_env = "GROQ_API_KEY"
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "llama-3.1-8b-instant": (0.05, 0.08),
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "openai/gpt-oss-120b": (0.15, 0.6),
        "openai/gpt-oss-20b": (0.075, 0.3),
        "openai/gpt-oss-safeguard-20b": (0.075, 0.3),
        "qwen/qwen3.6-27b": (0.6, 3.0),
        "qwen/qwen3.8-27b": (0.8, 4.0),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "llama-3.1-8b-instant": "Compact Llama instruction model for fast chat and loc",
        "llama-3.3-70b-versatile": "Open Llama instruction model for multilingual chat",
        "openai/gpt-oss-120b": "Open GPT reasoning model for self-hosted agents and co",
        "openai/gpt-oss-20b": "Open-weight GPT model for self-hosted reasoning and ins",
        "openai/gpt-oss-safeguard-20b": "Safety model for policy screening, moderation",
        "qwen/qwen3.6-27b": "Qwen vision-language model for visual reasoning, document",
        "qwen/qwen3.8-27b": "Dense 27B vision-language model for coding, agent tasks,",
    }
