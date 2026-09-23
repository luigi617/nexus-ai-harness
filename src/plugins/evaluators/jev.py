from __future__ import annotations

import os
from typing import Any, ClassVar

from dotenv import load_dotenv

from core.evaluation import EvaluationResult
from plugins.models._http import post_json
from protocols.context import Context
from protocols.evaluator import Evaluator


def noul(instructions: str | dict | list, criteria: dict | None = None) -> dict:
    """Build a yes/no question (returns a 0-1 ``noul`` score)."""
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        q["criteria"] = criteria
    return q


def choice(instructions: str | dict | list, criteria: dict) -> dict:
    """Build a pick-one question. ``criteria`` maps option -> description."""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str | dict | list, criteria: list) -> dict:
    """Build a rubric question. ``criteria`` is an ordered list of levels (2-10)."""
    return {"type": "score", "instructions": instructions, "criteria": criteria}


class JevEvaluator(Evaluator):
    """TypeSafe's Jev "System One" model (``api.typesafe.ai``).

    Returns typed decisions (noul/choice/score) with calibrated probabilities
    rather than text, so it is an :class:`Evaluator`, not a ``Model``.
    """

    provider = "typesafe"
    api_key_env = "TYPESAFE_API_KEY"
    base_url = "https://api.typesafe.ai/v1"
    # USD per 1M tokens: (input, output). Indicative — verify current pricing.
    pricing: ClassVar[dict[str, tuple[float, float]]] = {}

    def __init__(
        self,
        model: str = "jev-latest",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **params,
    ) -> None:
        self.name = model
        self.params = params
        load_dotenv()
        self.base_url = (base_url or self.base_url).rstrip("/")
        self.api_key = api_key or os.getenv(self.api_key_env)
        self.timeout = timeout

    def evaluate(
        self,
        state: str | dict | list,
        questions: dict[str, dict],
        ctx: Context,
    ) -> EvaluationResult:
        payload: dict[str, Any] = {
            "state": state,
            "model": self.name,
            "questions": questions,
            **self.params,
        }
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = post_json(
            f"{self.base_url}/systemone", payload, headers, self.timeout
        )
        result = self._parse(response)
        result.cost = self._cost(result.usage)
        return result

    def _cost(self, usage: dict) -> float:
        input_price, output_price = self.pricing.get(self.name, (0.0, 0.0))
        return (
            usage.get("input_tokens", 0) * input_price
            + usage.get("output_tokens", 0) * output_price
        ) / 1_000_000

    @staticmethod
    def _parse(response: dict) -> EvaluationResult:
        usage = response.get("usage", {})
        return EvaluationResult(
            model=response.get("model", ""),
            answers=response.get("answers", {}),
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
        )
