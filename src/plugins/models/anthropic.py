from __future__ import annotations

from typing import Any, ClassVar

from core.message import Message
from core.response import Response
from plugins.models._http import post_json
from plugins.models.base import BaseModel
from protocols.tool import Tool

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicModel(BaseModel):
    """The Anthropic Messages API (direct, not via Bedrock)."""

    provider = "anthropic"
    base_url = "https://api.anthropic.com/v1"
    api_key_env = "ANTHROPIC_API_KEY"
    # USD per 1M tokens: (input, output). Indicative — verify current pricing.
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "claude-fable-5": (10.0, 50.0),
        "claude-fable-5-1": (10.0, 50.0),
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-haiku-4-5-20251001": (1.0, 5.0),
        "claude-opus-4-5": (5.0, 25.0),
        "claude-opus-4-5-20251101": (5.0, 25.0),
        "claude-opus-4-6": (5.0, 25.0),
        "claude-opus-4-7": (5.0, 25.0),
        "claude-opus-4-8": (5.0, 25.0),
        "claude-opus-5": (5.0, 25.0),
        "claude-opus-5-5": (4.0, 20.0),
        "claude-sonnet-4-5": (3.0, 15.0),
        "claude-sonnet-4-5-20250929": (3.0, 15.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-sonnet-5": (2.0, 10.0),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "claude-fable-5": "Claude model for creative writing, analysis, and controlled",
        "claude-fable-5-1": "Claude model for demanding reasoning and long-horizon age",
        "claude-haiku-4-5": "Fast Claude lane for lightweight agents, office tasks, an",
        "claude-haiku-4-5-20251001": "Fast Claude model for responsive assistance, cla",
        "claude-opus-4-5": "Flagship Claude model for deep reasoning, coding, and long",
        "claude-opus-4-5-20251101": "Flagship Claude model for deep reasoning, coding,",
        "claude-opus-4-6": "High-end Claude for difficult coding, planning, and slower",
        "claude-opus-4-7": "Stronger Opus tier for advanced software work and high-sta",
        "claude-opus-4-8": "Top Claude Opus tier for the hardest reasoning, coding, an",
        "claude-opus-5": "Strongest Claude Opus model for coding, agents, and professi",
        "claude-opus-5-5": "Claude model for long-running agentic coding and knowledge",
        "claude-sonnet-4-5": "Balanced Claude model for coding, analysis, agent workfl",
        "claude-sonnet-4-5-20250929": "Balanced Claude model for coding, analysis, age",
        "claude-sonnet-4-6": "Claude workhorse for coding agents, careful analysis, an",
        "claude-sonnet-5": "Everyday Claude agent model for coding, planning, browsing",
    }

    def _generate(self, history: list[Message], tools: list[Tool]) -> Response:
        system, messages = self._to_messages(history)
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **self.params,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [self._tool_spec(t) for t in tools]
        headers: dict[str, str] = {"anthropic-version": ANTHROPIC_VERSION}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        response = post_json(
            f"{self.base_url}/messages", payload, headers, self.timeout
        )
        parsed = self._parse(response)
        parsed.cost = self._cost(parsed.usage)
        return parsed

    @staticmethod
    def _to_messages(history: list[Message]) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        messages: list[dict] = []
        pending_results: list[dict] = []

        def add_turn(role: str, blocks: list[dict]) -> None:
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": list(blocks)})

        def flush_results() -> None:
            if pending_results:
                add_turn("user", pending_results)
                pending_results.clear()

        for m in history:
            if m.role == "tool":
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_use_id,
                        "content": m.content,
                    }
                )
                continue

            flush_results()
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                add_turn("user", [{"type": "text", "text": m.content}])
            elif m.role == "assistant":
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for call in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id"),
                            "name": call.get("name"),
                            "input": call.get("arguments", {}),
                        }
                    )
                if blocks:  # an empty assistant turn is rejected by the API
                    add_turn("assistant", blocks)

        flush_results()
        return "\n".join(system_parts), messages

    @staticmethod
    def _tool_spec(tool: Tool) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters or {"type": "object", "properties": {}},
        }

    @staticmethod
    def _parse(response: dict) -> Response:
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "arguments": block.get("input", {}),
                    }
                )
        usage = response.get("usage", {})
        return Response(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        )
