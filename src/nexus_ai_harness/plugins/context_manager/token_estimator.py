from __future__ import annotations

import json

from nexus_ai_harness.core.message import Message
from nexus_ai_harness.protocols.token_estimator import TokenEstimator


class CharTokenEstimator(TokenEstimator):
    """A tokenizer-free estimate: about one token per ``chars_per_token`` chars.

    Counts message content, tool-call names and arguments, and a fixed
    per-message overhead for role and framing tokens. Cheap and
    provider-agnostic, so it suits budgeting rather than billing.

    Args:
        chars_per_token: Characters assumed per token; 4 is typical for
            English prose, lower is more conservative for code or JSON.
        per_message: Tokens added per message for role and framing.
    """

    def __init__(self, chars_per_token: float = 4.0, per_message: int = 4) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        self._chars_per_token = chars_per_token
        self._per_message = per_message

    def estimate(self, messages: list[Message]) -> int:
        chars = 0
        for m in messages:
            chars += len(m.content)
            for call in m.tool_calls:
                chars += len(str(call.get("name", "")))
                chars += len(json.dumps(call.get("arguments", {}), default=str))
        return int(chars / self._chars_per_token) + self._per_message * len(messages)
