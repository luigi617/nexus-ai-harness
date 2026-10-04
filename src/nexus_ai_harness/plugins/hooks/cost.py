from __future__ import annotations

from dataclasses import dataclass

from nexus_ai_harness.core.events import Event, ModelCallCompleted, ResponseReceived
from nexus_ai_harness.core.response import Response
from nexus_ai_harness.core.run import CostState
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.hook import Hook

__all__ = ["CostCounter", "CostState"]


@dataclass
class _LastCounted:
    """The response most recently counted from a ModelCallCompleted."""

    response: Response | None = None


class CostCounter(Hook):
    """Accumulate each model response's cost (USD) and token usage into CostState.

    Every :class:`ModelCallCompleted` is counted, which covers auxiliary calls
    such as summarization and LLM routing as well as the loop's own. A
    :class:`ResponseReceived` is counted only when no ``ModelCallCompleted``
    already carried the same response, so loops that emit just
    ``ResponseReceived`` are still counted, and nothing is counted twice.
    """

    def on(self, event: Event, ctx: Context) -> None:
        if isinstance(event, ModelCallCompleted):
            ctx.state(_LastCounted).response = event.response
            self._add(event.response, ctx)
        elif isinstance(event, ResponseReceived):
            last = ctx.state(_LastCounted)
            if last.response is event.response:
                last.response = None  # already counted from ModelCallCompleted
                return
            self._add(event.response, ctx)

    @staticmethod
    def _add(response: Response, ctx: Context) -> None:
        state = ctx.state(CostState)
        state.total += response.cost
        state.input_tokens += response.usage.get("input_tokens", 0)
        state.output_tokens += response.usage.get("output_tokens", 0)
