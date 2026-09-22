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
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        "claude-3-5-haiku-20241022": (0.8, 4.0),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "claude-3-5-sonnet-20241022": "balanced capability and cost",
        "claude-3-5-haiku-20241022": "fastest and cheapest; simple tasks",
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
