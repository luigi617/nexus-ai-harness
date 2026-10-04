from __future__ import annotations

from dataclasses import dataclass, field, replace

from nexus_ai_harness.core.events import (
    ContextCompacted,
    Event,
    ModelCallStarted,
    ResponseReceived,
)
from nexus_ai_harness.core.message import Message
from nexus_ai_harness.core.persistable import persistable
from nexus_ai_harness.plugins.context_manager.token_estimator import (
    CharTokenEstimator,
)
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.context_manager import ContextManager
from nexus_ai_harness.protocols.lifecycle import Lifecycle
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.token_estimator import TokenEstimator
from nexus_ai_harness.services.model_call import timed_complete


@persistable("summarizing.summary")
@dataclass
class SummaryState:
    """Compaction bookkeeping of one :class:`SummarizingContextManager` in a session.

    Attributes:
        upto: History index up to which messages are folded into ``text``.
        text: The cached rolling summary, or ``""`` before the first fold.
        dropped_upto: History index up to which messages are omitted by the
            drop fallback without being summarized.
        head_chars: Character limit applied to the head user message once the
            head alone outgrew half the token budget, or ``0`` if never.
        summaries: How many folds into the summary succeeded.
        fallbacks: How many times a fallback (drop, truncate, or overflow) ran.
        last_fallback: ``"<strategy>: <reason>"`` of the latest fallback.
    """

    upto: int = 0
    text: str = ""
    dropped_upto: int = 0
    head_chars: int = 0
    summaries: int = 0
    fallbacks: int = 0
    last_fallback: str = ""
    # Set while the prompt stays over budget, so the overflow is reported once.
    overflowed: bool = field(default=False, repr=False)


@dataclass
class _UsageState:
    pending_estimate: int = 0
    sent_estimate: int = 0
    input_tokens: int = 0


@dataclass
class _Slot:
    summary: SummaryState = field(default_factory=SummaryState)
    usage: _UsageState = field(default_factory=_UsageState)


@dataclass
class _Slots:
    """Session state keyed by manager, so chained instances don't share cuts."""

    by_manager: dict[object, _Slot] = field(default_factory=dict)
    # Only the first manager to ask gets the persistable SummaryState singleton.
    claimed: bool = False


_SUMMARY_HEADER = "[Conversation summary so far]"
_OMITTED_NOTE = "[{n} earlier messages omitted to fit the context window]"
_SUMMARY_TRUNCATED = "[recap truncated to its length limit]"
_TRUNCATED_NOTE = "\n[... {n} characters truncated to fit the context window ...]\n"
# Providers report cached prompt tokens separately from uncached input tokens.
_INPUT_USAGE_KEYS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
_RENDER_CHARS = 4000  # per message, so one huge tool result can't flood the recap
_DEFAULT_CHUNK_CHARS = 128_000
_HEAD_SHARE = 0.5  # of max_tokens the head may keep once the prompt overflows
_SUMMARY_SHARE = 0.25  # of max_tokens the summary may take at most
_MIN_TRUNCATED_CHARS = 200
_MESSAGE_BUDGET = "message budget exceeded"
_TOKEN_BUDGET = "token budget exceeded"


def _summary_prompt(max_tokens: int) -> str:
    return (
        "You are compacting a long agent conversation to save context space. "
        "Rewrite the material below into a dense, factual recap that a fresh "
        "assistant could resume from without losing anything load-bearing. "
        "Preserve, under these headings: Goal, Decisions, Files/artifacts touched, "
        "Current state, Open threads. Keep concrete names, ids, and values "
        "verbatim. Omit pleasantries and narration. "
        f"Keep the recap under {max_tokens} tokens. When an existing recap is "
        "given, merge the new messages into it: keep facts that still matter, "
        "update ones that changed, and drop ones that are resolved or superseded, "
        "so the recap never grows past that limit. Output only the recap."
    )


def _truncate_text(text: str, limit: int) -> str:
    """Keep the start and end of ``text`` in about ``limit`` chars, marking the cut."""
    if len(text) <= limit:
        return text
    keep = max(limit - len(_TRUNCATED_NOTE.format(n=len(text))), 0)
    head = keep * 2 // 3
    tail = keep - head
    note = _TRUNCATED_NOTE.format(n=len(text) - keep)
    return text[:head] + note + (text[-tail:] if tail > 0 else "")


@dataclass(frozen=True)
class _Layout:
    """The pieces of the model's view: kept head, merged notes, verbatim tail."""

    head: list[Message]
    notes: list[str]
    tail: list[Message]

    @classmethod
    def of(
        cls,
        head: list[Message],
        history: list[Message],
        cutoff: int,
        start: int,
        summary: str,
    ) -> _Layout:
        notes: list[str] = []
        if summary:
            notes.append(f"{_SUMMARY_HEADER}\n{summary}")
        if start > cutoff:
            notes.append(_OMITTED_NOTE.format(n=start - cutoff))
        return cls(head, notes, history[start:])

    def view(self) -> list[Message]:
        """Join the pieces so that no two user turns are ever adjacent.

        The notes are merged into the head user message, and so is a tail that
        opens on a user turn, since strict-alternation backends reject both.
        """
        if not self.notes:
            return [*self.head, *self.tail]
        head, tail = self.head, self.tail
        parts: list[str] = []
        if head and head[-1].role == "user":
            parts.append(head[-1].content)
            head = head[:-1]
        parts.extend(self.notes)
        if tail and tail[0].role == "user":
            parts.append(tail[0].content)
            tail = tail[1:]
        seam = Message(role="user", content="\n\n".join(parts))
        return [*head, seam, *tail]


class SummarizingContextManager(ContextManager, Lifecycle):
    """Keeps the history sent to the model within a message and token budget.

    The stable head (leading system messages plus the first user message) is
    kept. Once the live history outgrows a budget, the oldest turns after the
    head are folded into a cached, length-capped model summary, and only the
    most recent turns stay verbatim. The summary is merged into the head user
    message, so user and assistant turns keep strictly alternating.

    When no model is available or summarization fails, those turns are dropped
    instead and a note marks the gap. If the result still exceeds the token
    budget, oversized tool results and then the largest remaining messages
    (content and tool-call arguments) are truncated. If the head alone takes
    more than half the token budget once the prompt overflows, the first user
    message is truncated to fit that half, keeping its start and end. Every
    compaction emits a :class:`ContextCompacted` event and is counted in
    :meth:`state`; a prompt that stays over budget after all of this is
    reported once as an ``"overflow"``. No cut ever separates a tool result
    from the assistant tool call it answers.

    Token counts come from ``estimator``, else a registered
    :class:`TokenEstimator`, else :class:`CharTokenEstimator`. Once the harness
    has started this plugin, estimates are anchored to the ``input_tokens`` the
    provider reported for the previous model call, which also accounts for tool
    schemas the history alone cannot show.

    Each instance keeps its own per-session state, so several can be chained.

    Args:
        max_messages: Compact once more than this many messages follow the
            head and summary; ``None`` disables the message budget.
        keep_recent: How many of the newest messages a compaction keeps
            verbatim; the token budget may keep fewer, never below one.
        max_tokens: Compact once the estimated prompt exceeds this many tokens;
            ``None`` disables the token budget. Set it below the model's
            context window to leave room for tool schemas and the output.
        max_summary_tokens: Upper bound on the summary's length, further capped
            at a quarter of ``max_tokens``. The summarizer is asked to stay
            under it, and longer summaries are truncated.
        estimator: The token estimator to use instead of resolving one.

    Raises:
        ValueError: If neither budget is set or a limit is out of range.
    """

    def __init__(
        self,
        max_messages: int | None = 40,
        keep_recent: int = 12,
        *,
        max_tokens: int | None = None,
        max_summary_tokens: int = 2000,
        estimator: TokenEstimator | None = None,
    ) -> None:
        if max_messages is None and max_tokens is None:
            raise ValueError("set max_messages, max_tokens, or both")
        if keep_recent < 0:
            raise ValueError("keep_recent must be >= 0")
        if max_messages is not None and keep_recent >= max_messages:
            raise ValueError("keep_recent must be < max_messages")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if max_summary_tokens <= 0:
            raise ValueError("max_summary_tokens must be > 0")
        self._max_messages = max_messages
        self._keep_recent = keep_recent
        self._max_tokens = max_tokens
        self._max_summary_tokens = max_summary_tokens
        self._estimator_override = estimator
        self._default_estimator = CharTokenEstimator()
        self._anchor_eligible = True

    async def start(self, ctx: Context) -> None:
        """Subscribe to model calls so estimates can use reported token usage."""
        ctx.on(ModelCallStarted, self._on_model_call)
        ctx.on(ResponseReceived, self._on_response)
        # A non-last chained manager's view isn't what the model actually saw.
        peers = ctx.all(SummarizingContextManager)
        self._anchor_eligible = not peers or peers[-1] is self

    def state(self, ctx: Context) -> SummaryState:
        """Return this manager's compaction bookkeeping for ``ctx``'s session.

        Args:
            ctx: The run context whose session the state belongs to.

        Returns:
            The live, mutable state; created on first access.
        """
        return self._slot(ctx).summary

    def _slot(self, ctx: Context) -> _Slot:
        slots = ctx.state(_Slots)
        slot = slots.by_manager.get(self)
        if slot is None:
            # First manager to ask reuses the persistable singleton, so resume works.
            summary = ctx.state(SummaryState) if not slots.claimed else SummaryState()
            slots.claimed = True
            slot = slots.by_manager[self] = _Slot(summary=summary)
        return slot

    async def process(self, history: list[Message], ctx: Context) -> list[Message]:
        state = self.state(ctx)
        if state.upto > len(history) or state.dropped_upto > len(history):
            # The history was rewound or replaced, so the cached cuts are stale.
            state.upto, state.text, state.dropped_upto = 0, "", 0
            state.head_chars, state.overflowed = 0, False
        head_end = self._prefix_end(history)
        cutoff = max(state.upto, head_end)
        start = max(cutoff, state.dropped_upto)
        head = self._cut_head(history[:head_end], state.head_chars)
        layout = _Layout.of(head, history, cutoff, start, state.text)
        view = layout.view() if layout.notes or state.head_chars else history

        reason = self._over_budget(history, start, view, ctx)
        if reason is None:
            state.overflowed = False
            return view

        if not state.head_chars and self._over_tokens(view, ctx):
            # Folding the tail can't make room for a head that fills the budget.
            limit = self._head_limit(layout, ctx)
            if limit:
                state.head_chars = limit
                layout = replace(layout, head=self._cut_head(layout.head, limit))
                compacted = layout.view()
                self._record(
                    ctx,
                    state,
                    "truncate",
                    "head exceeds half the token budget",
                    view,
                    compacted,
                )
                view = compacted
                reason = self._over_budget(history, start, view, ctx)
                if reason is None:
                    state.overflowed = False
                    return view

        fixed = self._estimate(replace(layout, tail=[]).view(), ctx)
        # Under a token overrun alone, folding only helps if the head fits.
        foldable = reason == _MESSAGE_BUDGET or (
            self._max_tokens is not None and fixed < self._max_tokens
        )
        target = self._choose_cutoff(history, start, fixed, ctx) if foldable else start
        if target > start:
            folded = await self._summarize(state.text, history[cutoff:target], ctx)
            if folded is not None:
                state.text, state.upto = folded, target
                state.summaries += 1
                cutoff = target
                strategy = "summarize"
            else:
                state.dropped_upto = target
                strategy, reason = "drop", f"{reason}; summarization unavailable"
            start = target
            layout = _Layout.of(layout.head, history, cutoff, start, state.text)
            compacted = layout.view()
            self._record(ctx, state, strategy, reason, view, compacted)
            view = compacted

        if self._max_tokens is not None and self._over_tokens(view, ctx):
            shrunk = self._truncate(layout, self._max_tokens, ctx)
            if shrunk.tail != layout.tail:
                truncated = shrunk.view()
                self._record(ctx, state, "truncate", _TOKEN_BUDGET, view, truncated)
                view = truncated
        if self._over_tokens(view, ctx):
            if not state.overflowed:
                state.overflowed = True
                reason = "no fallback brings the prompt under the token budget"
                self._record(ctx, state, "overflow", reason, view, view)
        else:
            state.overflowed = False
        return view

    def _estimator(self, ctx: Context) -> TokenEstimator:
        return (
            self._estimator_override
            or ctx.get(TokenEstimator)
            or self._default_estimator
        )

    def _estimate(self, messages: list[Message], ctx: Context) -> int:
        """Estimate the prompt tokens of ``messages``, anchored to reported usage."""
        estimate = self._estimator(ctx).estimate(messages)
        usage = self._slot(ctx).usage
        if usage.input_tokens and self._anchor_eligible:
            # A fixed shift, not a ratio, so overhead like tool schemas doesn't scale.
            return max(0, usage.input_tokens + estimate - usage.sent_estimate)
        return estimate

    def _on_model_call(self, event: Event, ctx: Context) -> None:
        if isinstance(event, ModelCallStarted):
            estimate = self._estimator(ctx).estimate(event.history)
            self._slot(ctx).usage.pending_estimate = estimate

    def _on_response(self, event: Event, ctx: Context) -> None:
        if not isinstance(event, ResponseReceived):
            return
        usage = event.response.usage
        tokens = sum(
            int(value)
            for value in (usage.get(key) for key in _INPUT_USAGE_KEYS)
            if isinstance(value, int | float)
        )
        if tokens > 0:
            state = self._slot(ctx).usage
            state.input_tokens = tokens
            state.sent_estimate = state.pending_estimate

    def _over_budget(
        self, history: list[Message], start: int, view: list[Message], ctx: Context
    ) -> str | None:
        """Name the budget ``view`` exceeds, or None if it fits."""
        live = len(history) - start
        if self._max_messages is not None and live > self._max_messages:
            return _MESSAGE_BUDGET
        if self._over_tokens(view, ctx):
            return _TOKEN_BUDGET
        return None

    def _over_tokens(self, view: list[Message], ctx: Context) -> bool:
        if self._max_tokens is None:
            return False
        return self._estimate(view, ctx) > self._max_tokens

    def _head_limit(self, layout: _Layout, ctx: Context) -> int:
        """A char limit fitting the head user message into half the token budget.

        Returns ``0`` when the head already fits or has no user message to cut.
        """
        head = layout.head
        if self._max_tokens is None or not head or head[-1].role != "user":
            return 0
        excess = self._estimate(head, ctx) - int(self._max_tokens * _HEAD_SHARE)
        if excess <= 0:
            return 0
        message = head[-1]
        tokens = self._estimator(ctx).estimate([message])
        keep = int(len(message.content) * max(tokens - excess, 0) / max(tokens, 1))
        keep = max(keep, _MIN_TRUNCATED_CHARS)
        return keep if keep < len(message.content) else 0

    @staticmethod
    def _cut_head(head: list[Message], limit: int) -> list[Message]:
        if not limit or not head or head[-1].role != "user":
            return head
        last = head[-1]
        return [*head[:-1], replace(last, content=_truncate_text(last.content, limit))]

    def _choose_cutoff(
        self, history: list[Message], start: int, fixed: int, ctx: Context
    ) -> int:
        """Pick where the verbatim tail begins, at or after ``start``.

        Keeps at most ``keep_recent`` messages and, under a token budget, only as
        many as fit half the room left beside the ``fixed`` head and notes, but
        always the newest message. The other half is room to grow before the
        next compaction. The index is moved to the nearest cut that orphans no
        tool result.
        """
        last = len(history) - 1
        if last <= start:
            return start
        target = min(max(start, len(history) - self._keep_recent), last)
        if self._max_tokens is not None:
            estimator = self._estimator(ctx)
            tail_budget = (self._max_tokens - fixed) // 2
            sizes = [estimator.estimate([m]) for m in history[target:]]
            tail = sum(sizes)
            while target < last and tail > tail_budget:
                tail -= sizes.pop(0)
                target += 1
        return self._safe_cut(history, start, target)

    @staticmethod
    def _safe_cut(history: list[Message], start: int, target: int) -> int:
        """The cut nearest ``target`` (later preferred) keeping tool pairs whole.

        Returns ``start`` when no such cut exists after it.
        """
        blocked = SummarizingContextManager._blocked_cuts(history)
        for idx in range(target, len(history)):
            if idx not in blocked:
                return idx
        for idx in range(target - 1, start, -1):
            if idx not in blocked:
                return idx
        return start

    @staticmethod
    def _blocked_cuts(history: list[Message]) -> set[int]:
        """Indices where starting the kept tail would orphan a tool result.

        A tail may not begin with a tool result, nor anywhere between an
        assistant tool call and the last result answering it.
        """
        blocked: set[int] = set()
        call_index: dict[object, int] = {}
        for i, m in enumerate(history):
            if m.role == "tool":
                blocked.add(i)
                parent = call_index.get(m.tool_use_id)
                if parent is not None:
                    blocked.update(range(parent + 1, i + 1))
            for call in m.tool_calls:
                if call.get("id") is not None:
                    call_index[call["id"]] = i
        return blocked

    def _truncate(self, layout: _Layout, budget: int, ctx: Context) -> _Layout:
        """Cut tail messages until the layout's view fits ``budget``.

        Oversized tool results are capped first, then the largest remaining
        messages are shortened, content and string tool-call arguments alike;
        each keeps its start and end. The head and the notes are not cut here.
        """
        estimator = self._estimator(ctx)
        tail = list(layout.tail)
        cap = max(budget // 8, 1)
        for i, m in enumerate(tail):
            if m.role == "tool":
                tokens = estimator.estimate([m])
                if tokens > cap:
                    tail[i] = self._shrink(m, cap, tokens)

        out = replace(layout, tail=tail)
        overflow = self._estimate(out.view(), ctx) - budget
        stuck: set[int] = set()
        while overflow > 0:
            candidates = [
                (estimator.estimate([m]), i)
                for i, m in enumerate(tail)
                if i not in stuck and self._shrinkable(m)
            ]
            if not candidates:
                break
            tokens, i = max(candidates)
            shrunk = self._shrink(tail[i], tokens - overflow, tokens)
            if estimator.estimate([shrunk]) >= tokens:
                stuck.add(i)
                continue
            tail[i] = shrunk
            out = replace(layout, tail=tail)
            overflow = self._estimate(out.view(), ctx) - budget
        return out

    @staticmethod
    def _shrinkable(message: Message) -> bool:
        if len(message.content) > _MIN_TRUNCATED_CHARS:
            return True
        return any(
            isinstance(value, str) and len(value) > _MIN_TRUNCATED_CHARS
            for call in message.tool_calls
            if isinstance(arguments := call.get("arguments"), dict)
            for value in arguments.values()
        )

    @staticmethod
    def _shrink(message: Message, target_tokens: int, tokens: int) -> Message:
        """Scale the message's content and long string arguments by the ratio.

        Tool-call arguments stay a dict with the same keys, so the call remains
        well-formed; only this view changes, never the stored history.
        """
        ratio = max(target_tokens, 0) / max(tokens, 1)

        def cut(text: str) -> str:
            if len(text) <= _MIN_TRUNCATED_CHARS:
                return text
            keep = max(int(len(text) * ratio), _MIN_TRUNCATED_CHARS)
            return _truncate_text(text, keep)

        calls = [
            {
                **call,
                "arguments": {
                    key: cut(value) if isinstance(value, str) else value
                    for key, value in arguments.items()
                },
            }
            if isinstance(arguments := call.get("arguments"), dict)
            else call
            for call in message.tool_calls
        ]
        return replace(message, content=cut(message.content), tool_calls=calls)

    def _record(
        self,
        ctx: Context,
        state: SummaryState,
        strategy: str,
        reason: str,
        before: list[Message],
        after: list[Message],
    ) -> None:
        if strategy != "summarize":
            state.fallbacks += 1
            state.last_fallback = f"{strategy}: {reason}"
        ctx.emit(
            ContextCompacted(
                strategy=strategy,
                reason=reason,
                messages_before=len(before),
                messages_after=len(after),
                tokens_before=self._estimate(before, ctx),
                tokens_after=self._estimate(after, ctx),
            )
        )

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
        Long messages are shortened to their start and end.
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
        return _truncate_text(f"{m.role}: {' '.join(parts)}", _RENDER_CHARS)

    def _chunk_chars(self) -> int:
        if self._max_tokens is None:
            return _DEFAULT_CHUNK_CHARS
        # Half the budget, at ~4 chars per token, leaves room for prompt and recap.
        return max(self._max_tokens * 2, _RENDER_CHARS)

    @staticmethod
    def _chunks(lines: list[str], limit: int) -> list[list[str]]:
        chunks: list[list[str]] = [[]]
        size = 0
        for line in lines:
            if chunks[-1] and size + len(line) > limit:
                chunks.append([])
                size = 0
            chunks[-1].append(line)
            size += len(line) + 1
        return chunks

    async def _summarize(
        self, prior: str, messages: list[Message], ctx: Context
    ) -> str | None:
        """Fold ``messages`` into ``prior``, or None if summarization failed.

        A long slice is summarized in sequential chunks, each merged into the
        running recap, so no single request outgrows the budget.
        """
        model = ctx.get(Model)
        if model is None or not messages:
            return None
        lines = [line for line in (self._render(m) for m in messages) if line]
        if not lines:
            return prior or None
        recap = prior
        for chunk in self._chunks(lines, self._chunk_chars()):
            folded = await self._summarize_chunk(recap, "\n".join(chunk), model, ctx)
            if folded is None:
                return None
            recap = folded
        return recap

    async def _summarize_chunk(
        self, prior: str, transcript: str, model: Model, ctx: Context
    ) -> str | None:
        body = (
            transcript
            if not prior
            else f"Existing recap:\n{prior}\n\nNew messages:\n{transcript}"
        )
        request = [
            Message(role="system", content=_summary_prompt(self._summary_limit())),
            Message(role="user", content=body),
        ]
        try:
            # Summarization is a plain completion with no tools to offer.
            response = await timed_complete(model, request, [], ctx)
        except Exception:
            return None
        text = (response.text or "").strip()
        return self._cap_summary(text, ctx) if text else None

    def _summary_limit(self) -> int:
        if self._max_tokens is None:
            return self._max_summary_tokens
        # A recap that fills the budget would leave no room for the tail.
        cap = max(int(self._max_tokens * _SUMMARY_SHARE), 1)
        return min(self._max_summary_tokens, cap)

    def _cap_summary(self, text: str, ctx: Context) -> str:
        """Enforce ``max_summary_tokens`` so re-summarizing can't grow unbounded."""
        tokens = self._estimator(ctx).estimate(
            [Message(role="assistant", content=text)]
        )
        limit = self._summary_limit()
        if tokens <= limit:
            return text
        keep = int(len(text) * limit / tokens)
        cut = text[:keep]
        newline = cut.rfind("\n")
        if newline > keep // 2:  # end on a whole line when that loses little
            cut = cut[:newline]
        return f"{cut.rstrip()}\n{_SUMMARY_TRUNCATED}"
