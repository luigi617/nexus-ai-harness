from __future__ import annotations

import os
from abc import abstractmethod
from typing import ClassVar

from dotenv import load_dotenv

from core.message import Message
from core.response import Response
from protocols.context import Context
from protocols.model import Model
from protocols.tool import Tool


class BaseModel(Model):
    """Shared base for HTTP model backends.

    Subclasses set ``provider`` and, for keyed HTTP APIs, ``base_url`` and
    ``api_key_env``; they implement :meth:`_generate`. Common construction
    (key/base-url resolution, ``.env`` loading, params) and cost accounting
    live here. Keys are read only from configuration — never hardcoded.
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

    def complete(self, history: list[Message], ctx: Context) -> Response:
        return self._generate(history, ctx.all(Tool))

    def _cost(self, usage: dict) -> float:
        input_price, output_price = self.pricing.get(self.name, (0.0, 0.0))
        return (
            usage.get("input_tokens", 0) * input_price
            + usage.get("output_tokens", 0) * output_price
        ) / 1_000_000

    @abstractmethod
    def _generate(self, history: list[Message], tools: list[Tool]) -> Response:
        """Perform one completion request and return a parsed ``Response``."""
