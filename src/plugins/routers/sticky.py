from __future__ import annotations

from dataclasses import dataclass

from core.message import Message
from protocols.context import Context
from protocols.model import Model
from protocols.router import Router
from services.invoke import invoke


@dataclass
class StickyState:
    model: Model | None = None


class StickyRouter(Router):
    """Decide once, then reuse that model for the rest of the run.

    Delegates to an inner router on the first turn (when the history holds the
    first task) and reuses that model afterwards. Wrap any router, e.g.
    ``StickyRouter(LLMRouter(...))`` to pick a model from the first task and
    keep it.
    """

    def __init__(self, inner: Router) -> None:
        self._inner = inner

    async def route(self, history: list[Message], ctx: Context) -> Model:
        state = ctx.state(StickyState)
        if state.model is None:
            state.model = await invoke(ctx, self._inner.route, history, ctx)
        return state.model
