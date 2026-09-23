from __future__ import annotations

import json
from typing import Any

from core.message import Message
from core.response import Response
from plugins.models._http import post_json
from plugins.models.base import BaseModel
from protocols.tool import Tool


class OpenAICompatibleModel(BaseModel):
    """Base for backends that speak the OpenAI Chat Completions wire format.

    Most public LLM APIs (OpenAI, DeepSeek, MiniMax, Qwen, GLM, Groq, xAI) are
    compatible; each is a subclass setting ``provider``, ``base_url`` and
    ``api_key_env``. ``base_url``/``api_key`` are also overridable per instance,
    so this class is usable directly against any compatible endpoint.
    """

    provider = "openai"
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"

    def _generate(self, history: list[Message], tools: list[Tool]) -> Response:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": self._to_messages(history),
            "max_tokens": self.max_tokens,
            **self.params,
        }
        if tools:
            payload["tools"] = [self._tool_spec(t) for t in tools]
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = post_json(
            f"{self.base_url}/chat/completions", payload, headers, self.timeout
        )
        parsed = self._parse(response)
        parsed.cost = self._cost(parsed.usage)
        return parsed

    @staticmethod
    def _to_messages(history: list[Message]) -> list[dict]:
        messages: list[dict] = []
        for m in history:
            if m.role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_use_id,
                        "content": m.content,
                    }
                )
            elif m.role == "assistant":
                if not m.content and not m.tool_calls:
                    continue  # an empty assistant turn is rejected by the API
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": m.content or None,
                }
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": call.get("id"),
                            "type": "function",
                            "function": {
                                "name": call.get("name"),
                                "arguments": json.dumps(call.get("arguments", {})),
                            },
                        }
                        for call in m.tool_calls
                    ]
                messages.append(msg)
            else:  # system, user
                messages.append({"role": m.role, "content": m.content})
        return messages

    @staticmethod
    def _tool_spec(tool: Tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}},
            },
        }

    @staticmethod
    def _parse(response: dict) -> Response:
        choices = response.get("choices") or [{}]
        message = choices[0].get("message", {})
        tool_calls: list[dict] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(
                {
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": args or {},
                }
            )
        usage = response.get("usage", {})
        return Response(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        )
