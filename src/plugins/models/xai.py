from __future__ import annotations

from typing import ClassVar

from plugins.models.openai_compatible import OpenAICompatibleModel


class XAIModel(OpenAICompatibleModel):
    provider = "xai"
    base_url = "https://api.x.ai/v1"
    api_key_env = "XAI_API_KEY"
    descriptions: ClassVar[dict[str, str]] = {
        "grok-4": "xAI flagship reasoning model",
        "grok-4.20-0309-non-reasoning": "Grok model for agentic tool use, reasoning, c",
        "grok-4.20-0309-reasoning": "Reasoning Grok for document-heavy analysis and lo",
        "grok-4.20-multi-agent-0309": "Grok model for agentic tool use, reasoning, cod",
        "grok-4.3": "xAI's Grok for chat, coding, agentic tools, and lower hallucinati",
        "grok-4.5": "xAI's Grok model for chat, coding, agentic tools, and lower hallu",
        "grok-4.6": "xAI's frontier model for long-running agents, coding, knowledge w",
        "grok-4.7": "xAI's frontier model for long-running agents, coding, knowledge w",
        "grok-build-0.1": "Fast Grok coding model tuned for agentic engineering and it",
    }
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "grok-4.20-0309-non-reasoning": (1.25, 2.5),
        "grok-4.20-0309-reasoning": (1.25, 2.5),
        "grok-4.20-multi-agent-0309": (1.25, 2.5),
        "grok-4.3": (1.25, 2.5),
        "grok-4.5": (2.0, 6.0),
        "grok-4.6": (2.0, 6.0),
        "grok-4.7": (2.0, 6.0),
        "grok-build-0.1": (1.0, 2.0),
    }
