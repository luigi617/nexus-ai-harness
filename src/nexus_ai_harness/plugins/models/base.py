from __future__ import annotations

import os
from abc import abstractmethod
from typing import ClassVar

from dotenv import load_dotenv

from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.response import Response
from nexus_ai_harness.plugins.models.retry import RetryPolicy
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.tool import Tool


class BaseModel(Model):
    """Shared base for HTTP model backends.

    Subclasses set ``provider`` and, for keyed HTTP APIs, ``base_url`` and
    ``api_key_env``; they implement :meth:`_generate`. Common construction
    (key/base-url resolution, ``.env`` loading, params) and cost accounting
    live here. Keys are read only from configuration — never hardcoded.

    Transient request failures are retried with exponential backoff: pass
    ``max_retries`` to change only the retry count, or ``retry`` for full control
    (backoff, jitter, and injectable sleep/clock). ``max_retries=0`` disables it.
    """

    provider: str = ""
    base_url: str = ""
    api_key_env: str = ""
    # USD per 1M tokens: (input, output). Models not listed cost 0. Prices are
    # indicative and change often — verify against the provider's pricing page.
    pricing: ClassVar[dict[str, tuple[float, float]]] = {}
    descriptions: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int | None = None,
        retry: RetryPolicy | None = None,
        **params,
    ) -> None:
        self.name = model
        self.description = self.descriptions.get(model, "")
        self.params = params
        load_dotenv()
        self.base_url = (base_url or self.base_url).rstrip("/")
        self.api_key = api_key or (
            os.getenv(self.api_key_env) if self.api_key_env else None
        )
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry = RetryPolicy.resolve(retry, max_retries)

    def complete(
        self, history: list[Message], tools: list[Tool], ctx: Context
    ) -> Response:
        return self._generate(history, tools)

    def _cost(self, usage: dict) -> float:
        input_price, output_price = self.pricing.get(self.name, (0.0, 0.0))
        return (
            usage.get("input_tokens", 0) * input_price
            + usage.get("output_tokens", 0) * output_price
        ) / 1_000_000

    @abstractmethod
    def _generate(self, history: list[Message], tools: list[Tool]) -> Response:
        """Perform one completion request and return a parsed ``Response``."""
