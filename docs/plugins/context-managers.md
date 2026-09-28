# Context managers

A context manager shapes the history sent to the model on each turn without
changing the session's stored history. Several can be registered; they run in
registration order, each receiving the previous one's output.

## `SummarizingContextManager`

Keeps long conversations inside a message budget and a token budget:

```python
from nexus_ai_harness.plugins.context_manager import SummarizingContextManager

harness.use(SummarizingContextManager(max_messages=40, keep_recent=12))
harness.use(SummarizingContextManager(max_tokens=150_000))  # e.g. a 200k window
```

- **Head.** The leading system messages and the first user message are always
  kept verbatim.
- **Budgets.** It compacts once more than `max_messages` messages follow the
  head, or once the estimated prompt exceeds `max_tokens`. Either may be
  `None`, but not both. Below both budgets the history passes through
  unchanged.
- **Summary.** Compaction folds the oldest turns after the head into a model
  summary and keeps the newest `keep_recent` messages (fewer if they don't fit
  half of `max_tokens`). The summary is cached and reused until the budget is
  hit again. It is capped at `max_summary_tokens`: the summarizer is asked to
  stay under it, and longer output is truncated. Large slices are summarized
  in chunks.
- **Role alternation.** The summary is merged into the head user message, and
  so is a kept tail that opens on a user turn, so two user turns are never
  adjacent. Strict-alternation backends accept the result.
- **Tool pairs.** A cut never separates a tool result from the assistant tool
  call it answers, however the results are interleaved.

### Fallbacks

If no model is registered or summarization fails, it falls back to
deterministic steps instead of sending an over-budget history:

1. **Drop.** The same turns are omitted and a note in the head user message
   says how many. The drop is kept on later turns, and summarization is
   retried the next time the budget is exceeded; a later summary also covers
   the dropped turns.
2. **Truncate.** If the prompt still exceeds `max_tokens`, oversized tool
   results are capped, then the largest remaining messages are shortened.
   Each keeps its start and end around a marker. This also handles a single
   message too large for the budget.

Every compaction emits a `ContextCompacted` event (`strategy` is `"summarize"`,
`"drop"`, or `"truncate"`; `event.fallback` is true for the last two). Counts
and the latest fallback reason are kept in `ctx.state(SummaryState)`.

### Token estimates

Tokens are estimated by the `estimator` argument, else a registered
`TokenEstimator` plugin, else `CharTokenEstimator` (about four characters per
token). After the harness starts, estimates are anchored to the
`input_tokens` the provider reported for the previous call, so tool schemas
and tokenizer differences are accounted for. To plug in a real tokenizer,
implement the protocol:

```python
from nexus_ai_harness.protocols import TokenEstimator

class TiktokenEstimator(TokenEstimator):
    def estimate(self, messages):
        return sum(len(enc.encode(m.content)) + 4 for m in messages)

harness.use(TiktokenEstimator())
```

### In `default_harness`

`default_harness` registers `SummarizingContextManager(max_tokens=100_000)`
alongside the default 40-message budget. Short conversations are unaffected.
Pass `max_context_tokens=` to match your model's context window, or `None` to
budget by message count only.
