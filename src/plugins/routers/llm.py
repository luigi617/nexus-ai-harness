from __future__ import annotations

from core.invoke import invoke
from core.message import Message
from protocols.context import Context
from protocols.model import Model
from protocols.router import Router

_ROUTER_PROMPT = (
    "You are a model router. Given the user's request, choose the single most "
    "suitable model from the options below. Reply with ONLY the model id, "
    "nothing else.\n\nOptions:\n{options}"
)


def _id(model: Model) -> str:
    """Routing identifier composed from the model's protocol fields."""
    return f"{model.provider}${model.name}"


class LLMRouter(Router):
    """Asks a decider model which registered model should handle the request,
    each turn. Candidates are the models registered on the harness
    (``ctx.all(Model)``); it reasons over each model's ``id`` and ``description``.

    ``decider`` makes the routing decision; if omitted, the first registered
    model is used.
    """

    def __init__(self, decider: Model | None = None) -> None:
        self._decider = decider

    async def route(self, history: list[Message], ctx: Context) -> Model:
        candidates = ctx.all(Model)
        if not candidates:
            raise LookupError("no models registered")

        task = next((m.content for m in reversed(history) if m.role == "user"), "")
        options = "\n".join(
            f"- {_id(m)}: {getattr(m, 'description', '')}" for m in candidates
        )
        request = [
            Message(role="system", content=_ROUTER_PROMPT.format(options=options)),
            Message(role="user", content=task),
        ]
        decider = self._decider or candidates[0]
        response = await invoke(decider.complete, request, ctx)
        return self._match(response.text, candidates)

    @staticmethod
    def _match(text: str, candidates: list[Model]) -> Model:
        lowered = (text or "").strip().lower()
        for model in candidates:
            ident, name = _id(model).lower(), model.name.lower()
            if (ident and ident in lowered) or (name and name in lowered):
                return model
        return candidates[0]  # fall back to the first if nothing matched
