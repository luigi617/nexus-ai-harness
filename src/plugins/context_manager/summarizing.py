from __future__ import annotations

from dataclasses import dataclass

from core.invoke import invoke
from core.message import Message
from protocols.context import ContextManager
from protocols.mediator import Context
from protocols.provider import Provider


@dataclass
class SummaryState:
    upto: int = 0
    text: str = ""


_SUMMARY_PROMPT = (
    "You are compacting a long agent conversation to save context space. "
    "Rewrite the material below into a dense, factual recap that a fresh "
    "assistant could resume from without losing anything load-bearing. "
    "Preserve, under these headings: Goal, Decisions, Files/artifacts touched, "
    "Current state, Open threads. Keep concrete names, ids, and values verbatim. "
    "Omit pleasantries and narration. Output only the recap."
)


class SummarizingContextManager(ContextManager):
    """
    Collapses the middle of a long history into a cached LLM summary.
    """

    def __init__(self, max_messages: int = 40, keep_recent: int = 12) -> None:
        if keep_recent >= max_messages:
            raise ValueError("keep_recent must be < max_messages")
        self._max_messages = max_messages
        self._keep_recent = keep_recent

    async def process(self, history: list[Message], ctx: Context) -> list[Message]:
        head_end = self._prefix_end(history)
        state = ctx.state(SummaryState)
        cutoff = max(state.upto, head_end)

        # Fold newly-overflowing messages into the summary only when the live
        # tail has grown past the budget. Otherwise reuse the cached summary.
        if len(history) - cutoff > self._max_messages:
            new_cutoff = len(history) - self._keep_recent
            folded = await self._summarize(state.text, history[cutoff:new_cutoff], ctx)
            if folded is None:
                return history  # summarization unavailable — stay a no-op
            state.text = folded
            state.upto = new_cutoff
            cutoff = new_cutoff

        if not state.text:
            return history

        summary = Message(
            role="user",
            content=f"[Conversation summary so far]\n{state.text}",
        )
        return [*history[:head_end], summary, *history[cutoff:]]

    @staticmethod
    def _prefix_end(history: list[Message]) -> int:
        """Index past the stable instruction prefix: leading system messages
        plus the first user message."""
        i = 0
        while i < len(history) and history[i].role == "system":
            i += 1
        if i < len(history) and history[i].role == "user":
            i += 1
        return i

    async def _summarize(
        self, prior: str, messages: list[Message], ctx: Context
    ) -> str | None:
        provider = ctx.get(Provider)
        if provider is None or not messages:
            return None

        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages if m.content)
        body = (
            transcript
            if not prior
            else f"Existing recap:\n{prior}\n\nNew messages:\n{transcript}"
        )
        request = [
            Message(role="system", content=_SUMMARY_PROMPT),
            Message(role="user", content=body),
        ]
        try:
            response = await invoke(provider.complete, request, ctx)
        except Exception:
            return None
        text = (response.text or "").strip()
        return text or None
