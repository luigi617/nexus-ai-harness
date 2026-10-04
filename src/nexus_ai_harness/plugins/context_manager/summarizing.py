from __future__ import annotations

from dataclasses import dataclass

from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.persistable import persistable
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.context_manager import ContextManager
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.services.model_call import timed_complete


@persistable("summarizing.summary")
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
    """Collapses the middle of a long history into a cached Model summary."""

    def __init__(self, max_messages: int = 40, keep_recent: int = 12) -> None:
        if keep_recent >= max_messages:
            raise ValueError("keep_recent must be < max_messages")
        self._max_messages = max_messages
        self._keep_recent = keep_recent

    async def process(self, history: list[Message], ctx: Context) -> list[Message]:
        head_end = self._prefix_end(history)
        state = ctx.state(SummaryState)
        cutoff = max(state.upto, head_end)

        # Re-summarize only once the live tail outgrows the budget; else reuse.
        if len(history) - cutoff > self._max_messages:
            new_cutoff = len(history) - self._keep_recent
            # Never begin the live tail with an orphaned tool result: a toolResult
            # whose toolUse was folded into the summary is rejected by the backend.
            while new_cutoff < len(history) and history[new_cutoff].role == "tool":
                new_cutoff += 1
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
        """Index past the stable instruction prefix.

        The prefix is the leading system messages plus the first user message.
        """
        i = 0
        while i < len(history) and history[i].role == "system":
            i += 1
        if i < len(history) and history[i].role == "user":
            i += 1
        return i

    @staticmethod
    def _render(m: Message) -> str | None:
        """Render a message for the summary transcript, or None if it is empty.

        Includes tool-call intent so tool-only assistant turns (``content=""``
        with populated ``tool_calls``) are not silently dropped from the recap.
        """
        parts: list[str] = []
        if m.content:
            parts.append(m.content)
        for tc in m.tool_calls:
            parts.append(
                f"[tool_call {tc.get('name', '')} args={tc.get('arguments', {})}]"
            )
        if not parts:
            return None
        return f"{m.role}: {' '.join(parts)}"

    async def _summarize(
        self, prior: str, messages: list[Message], ctx: Context
    ) -> str | None:
        model = ctx.get(Model)
        if model is None or not messages:
            return None

        transcript = "\n".join(
            line for line in (self._render(m) for m in messages) if line is not None
        )
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
            # Summarization is a plain completion with no tools to offer.
            response = await timed_complete(model, request, [], ctx)
        except Exception:
            return None
        text = (response.text or "").strip()
        return text or None
