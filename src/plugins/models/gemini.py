from __future__ import annotations

from typing import Any, ClassVar

from core.message import Message
from core.response import Response
from plugins.models._http import post_json
from plugins.models.base import BaseModel
from protocols.tool import Tool


class GeminiModel(BaseModel):
    """Google Gemini via the generateContent API."""

    provider = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env = "GEMINI_API_KEY"
    # USD per 1M tokens: (input, output). Indicative — verify current pricing.
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "gemini-1.5-pro": (1.25, 5.0),
        "gemini-1.5-flash": (0.075, 0.3),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "gemini-1.5-pro": "capable general-purpose model",
        "gemini-1.5-flash": "fast and cheap; simple tasks",
    }

    def _generate(self, history: list[Message], tools: list[Tool]) -> Response:
        system, contents = self._to_contents(history)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self.max_tokens, **self.params},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [
                {"functionDeclarations": [self._tool_spec(t) for t in tools]}
            ]
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        url = f"{self.base_url}/models/{self.name}:generateContent"
        response = post_json(url, payload, headers, self.timeout)
        parsed = self._parse(response)
        parsed.cost = self._cost(parsed.usage)
        return parsed

    @staticmethod
    def _to_contents(history: list[Message]) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        contents: list[dict] = []

        def add_turn(role: str, parts: list[dict]) -> None:
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": list(parts)})

        for m in history:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "user":
                add_turn("user", [{"text": m.content}])
            elif m.role == "tool":
                # Gemini carries tool results as a user-role functionResponse.
                add_turn(
                    "user",
                    [
                        {
                            "functionResponse": {
                                "name": m.name,
                                "response": {"result": m.content},
                            }
                        }
                    ],
                )
            elif m.role == "assistant":
                parts: list[dict] = []
                if m.content:
                    parts.append({"text": m.content})
                for call in m.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "name": call.get("name"),
                                "args": call.get("arguments", {}),
                            }
                        }
                    )
                if parts:  # an empty assistant turn is rejected by the API
                    add_turn("model", parts)

        return "\n".join(system_parts), contents

    @staticmethod
    def _tool_spec(tool: Tool) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters or {"type": "object", "properties": {}},
        }

    @staticmethod
    def _parse(response: dict) -> Response:
        candidates = response.get("candidates") or [{}]
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for i, part in enumerate(parts):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                name = fc.get("name")
                tool_calls.append(
                    {
                        # Gemini omits call ids; synthesize a stable one.
                        "id": f"call_{name}_{i}",
                        "name": name,
                        "arguments": fc.get("args", {}),
                    }
                )
        usage = response.get("usageMetadata", {})
        return Response(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
            },
        )
