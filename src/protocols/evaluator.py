from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable

from core.evaluation import EvaluationResult
from protocols.context import Context
from protocols.plugin import Plugin


class Evaluator(Plugin):
    """A structured-evaluation backend: state + typed questions -> typed answers.

    Distinct from :class:`~protocols.model.Model`: an evaluator carries no
    conversation and emits neither free text nor tool calls, so the agentic and
    chat loops (which fetch ``Model`` and read ``response.text`` /
    ``response.tool_calls``) cannot drive it. Register and fetch it by the
    ``Evaluator`` type, use it directly, or wrap it in a Tool.
    """

    provider: str = ""
    name: str = ""

    @abstractmethod
    def evaluate(
        self,
        state: str | dict | list,
        questions: dict[str, dict],
        ctx: Context,
    ) -> EvaluationResult | Awaitable[EvaluationResult]:
        """Evaluate ``questions`` against ``state`` and return typed answers.

        May be implemented as sync or ``async def`` — the harness adapts.
        """
