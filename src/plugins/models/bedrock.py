from __future__ import annotations

import os
from typing import Any, ClassVar

import boto3
from dotenv import load_dotenv

from core.message import Message
from core.response import Response
from plugins.models.base import BaseModel
from protocols.tool import Tool

DEFAULT_REGION = "us-east-1"


class BedrockModel(BaseModel):
    provider = "bedrock"
    # USD per 1M tokens: (input, output). Models not listed here cost 0.
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "us.anthropic.claude-opus-4-8": (15.0, 75.0),
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": (3.0, 15.0),
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": (0.8, 4.0),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "us.anthropic.claude-opus-4-8": "most capable; hard reasoning, complex tasks",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": "balanced capability and cost",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": "fastest and cheapest; simple",
    }

    def __init__(
        self,
        model: str,
        region: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        **params,
    ) -> None:
        self.name = model
        self.description = self.descriptions.get(model, "")
        self.params = params
        load_dotenv()
        self.region = region or os.getenv("AWS_REGION") or DEFAULT_REGION
        self.api_key = api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self.api_key:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.api_key
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _generate(self, history: list[Message], tools: list[Tool]) -> Response:
        system, messages = self._to_converse(history)
        kwargs: dict[str, Any] = {
            "modelId": self.name,
            "messages": messages,
            "inferenceConfig": {"maxTokens": self.max_tokens, **self.params},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["toolConfig"] = {"tools": [self._tool_spec(t) for t in tools]}
        response = self._get_client().converse(**kwargs)
        parsed = self._parse(response)
        parsed.cost = self._cost(parsed.usage)
        return parsed

    @staticmethod
    def _to_converse(history: list[Message]) -> tuple[list[dict], list[dict]]:
        system: list[dict] = []
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
                # Batch consecutive tool results into one user turn.
                pending_results.append(
                    {
                        "toolResult": {
                            "toolUseId": m.tool_use_id,
                            "content": [{"text": m.content}],
                        }
                    }
                )
                continue

            flush_results()
            if m.role == "system":
                system.append({"text": m.content})
            elif m.role == "user":
                add_turn("user", [{"text": m.content}])
            elif m.role == "assistant":
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"text": m.content})
                for call in m.tool_calls:
                    blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": call.get("id"),
                                "name": call.get("name"),
                                "input": call.get("arguments", {}),
                            }
                        }
                    )
                add_turn("assistant", blocks)

        flush_results()
        return system, messages

    @staticmethod
    def _tool_spec(tool: Tool) -> dict:
        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "json": tool.parameters or {"type": "object", "properties": {}}
                },
            }
        }

    @staticmethod
    def _parse(response: dict) -> Response:
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                use = block["toolUse"]
                tool_calls.append(
                    {
                        "id": use.get("toolUseId"),
                        "name": use.get("name"),
                        "arguments": use.get("input", {}),
                    }
                )
        usage = response.get("usage", {})
        return Response(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
            },
        )
