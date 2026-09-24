# e2e tests

Data-driven end-to-end tests. Each case is a JSON file under `cases/`; the
runner composes a real harness (a Python builder), drives the conversation, and
asserts on the outcome. `test_e2e.py` discovers every JSON file (recursively) and
runs one parametrized pytest per case. Deterministic scripted cases live under
`cases/scripted/` and run in CI; the live cases at the top of `cases/` are
opt-in (see below).

## Running

```bash
pytest tests/test_e2e.py                 # scripted cases only (default, in CI)
NEXUS_E2E_LIVE=1 pytest tests/test_e2e.py # also run live cases (real providers)
```

## Two modes, one schema

- **Scripted** (`"model": "scripted"`) replays `scripted_responses` through an
  in-memory model. Deterministic and free — assert exactly (tool calls, stop
  reason, output substrings). Runs in CI.
- **Live** (`"model": "provider:model-id"`) calls a real backend. Nondeterministic,
  so assert on *behavior* (which tools ran, stop reason, cost) plus loose factual
  anchors (`output_contains_any`), never full sentences. Skipped unless
  `NEXUS_E2E_LIVE=1`.

The design principle: for an agent, the **trajectory is the ground truth**, not
the prose. Prefer `tool_calls` / `stop_reason` / `max_cost_usd` (deterministic
even live) and use text checks only for factual tokens.

## Case schema

| Field | Type | Meaning |
|---|---|---|
| `name` | string (required) | Unique case id; shown as the pytest id. |
| `input` | string (required) | Opening user message. |
| `description` | string | Human note. |
| `tags` | string[] | Free-form labels. |
| `harness` | string | Builder name from `harnesses.py` (default `"default"`). |
| `model` | string | `"scripted"` or `"provider:model-id"`. |
| `scripted_responses` | object[] | Scripted mode only; one per model turn (see below). |
| `user_turns` | string[] | Follow-up user messages (multi-turn, one session). |
| `env` | object | Env vars set for the run, then restored. |
| `expect` | object | Assertions (see below). |
| `timeout_s` | number | Per-case timeout (default 120). |
| `skip` | bool | Skip this case. |

A `scripted_responses` entry: `{ "text": "...", "tool_calls": [{ "name": "...",
"arguments": {...}, "id"?: "..." }], "cost"?: 0.0 }`. The last entry repeats if
the loop needs more turns.

### `expect` fields (all optional; active ones are AND-ed)

Behavioral (robust against a live model):
- `no_error` (default `true`) — the run raised no exception.
- `stop_reason` — exact match, e.g. `"completed"`.
- `stop_reason_contains` — substring match, e.g. `"guard:"` to assert a guard
  halt without pinning the exact wording/number.
- `tool_calls` — list of `{ "name", "arguments_contains"?: {...} }`; each must
  match some recorded call (or, with `tool_calls_ordered: true`, appear in order).
- `tool_names` — exact multiset of tool names called (use `[]` for "no tools").
- `tool_denied` — list of `{ "name" }` that must have been blocked by the
  permission layer.
- `max_cost_usd`, `max_iterations` — upper bounds.

Text matching (anchor on facts, not phrasing). For multi-turn cases these match
against the **whole transcript** (every turn's final text joined), so a fact
answered in an earlier turn still counts:
- `output_contains` — all substrings must appear.
- `output_contains_any` — at least one must appear.
- `output_not_contains` — none may appear (checked across all turns).
- `output_regex` — must match.

## Registered harnesses

| Name | What it adds |
|---|---|
| `default` | agentic loop + calculator/echo tools + auto-approve + generous guards |
| `tiny_budget` | `default` with `BudgetGuard(1e-9)` — any real call trips it |
| `one_iteration` | `default` with `MaxIterations(1)` |
| `deny_calculator` | `default` + `DenyList(["calculator"])` |
| `memory` | `default` + `FileMemoryStore` (temp dir) + remember/recall/forget tools |
| `subagent` | `default` + `Subagent` + `InProcessSpawner` |

## Adding a harness

Register a builder in `harnesses.py` and reference it by name from JSON:

```python
@e2e_harness("memory")
def _memory(model):
    return NexusAIHarness().use(AgenticLoop()).use(model).use(...)
```

Builders register the loop, tools, and guards — but not the model (the runner
supplies it) and not the metrics recorder (the runner adds it).
