from __future__ import annotations

import os
from typing import Any, ClassVar

import boto3
from botocore.config import Config
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
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": (0.8, 4.0),
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": (3.0, 15.0),
        "us.anthropic.claude-fable-5": (11.0, 55.0),
        "us.anthropic.claude-fable-5-1": (11.0, 55.0),
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.1, 5.5),
        "us.anthropic.claude-opus-4-1-20250805-v1:0": (15.0, 75.0),
        "us.anthropic.claude-opus-4-5-20251101-v1:0": (5.5, 27.5),
        "us.anthropic.claude-opus-4-6-v1": (5.5, 27.5),
        "us.anthropic.claude-opus-4-7": (5.5, 27.5),
        "us.anthropic.claude-opus-4-8": (5.5, 27.5),
        "us.anthropic.claude-opus-5": (5.5, 27.5),
        "us.anthropic.claude-opus-5-5": (4.4, 22.0),
        "us.anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.3, 16.5),
        "us.anthropic.claude-sonnet-4-6": (3.3, 16.5),
        "us.anthropic.claude-sonnet-5": (2.2, 11.0),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": "fastest and cheapest; simple",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": "balanced capability and cost",
        "us.anthropic.claude-fable-5": "Claude model for creative writing, analysis, a",
        "us.anthropic.claude-fable-5-1": "Claude model for demanding reasoning and lon",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Fast Claude model for responsi",
        "us.anthropic.claude-opus-4-1-20250805-v1:0": "Flagship Claude model for deep",
        "us.anthropic.claude-opus-4-5-20251101-v1:0": "Flagship Claude model for deep",
        "us.anthropic.claude-opus-4-6-v1": "High-end Claude for difficult coding, plan",
        "us.anthropic.claude-opus-4-7": "Stronger Opus tier for advanced software work",
        "us.anthropic.claude-opus-4-8": "most capable; hard reasoning, complex tasks",
        "us.anthropic.claude-opus-5": "Strongest Claude Opus model for coding, agents,",
        "us.anthropic.claude-opus-5-5": "Claude model for long-running agentic coding",
        "us.anthropic.claude-sonnet-4-20250514-v1:0": "Balanced Claude model for codin",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Balanced Claude model for cod",
        "us.anthropic.claude-sonnet-4-6": "Claude workhorse for coding agents, careful",
        "us.anthropic.claude-sonnet-5": "Everyday Claude agent model for coding, plann",
    }

    def __init__(
        self,
        model: str,
        region: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        **params,
    ) -> None:
        self.name = model
        self.description = self.descriptions.get(model, "")
        self.params = params
        load_dotenv()
        self.region = region or os.getenv("AWS_REGION") or DEFAULT_REGION
        self.api_key = api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        self.max_tokens = max_tokens
        # Accepted explicitly so it is not swept into self.params, where it would
        # pollute inferenceConfig (Converse rejects unknown members).
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self.api_key:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.api_key
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(connect_timeout=self.timeout, read_timeout=self.timeout),
            )
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
                if blocks:  # an empty assistant turn is rejected by the API
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
