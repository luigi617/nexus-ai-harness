# benchmarks

Run benchmarks against agents built on `nexus-ai-harness`.

## Install

```bash
pip install -e '.[benchmarks]'
```

## List available benchmarks

```bash
python -m benchmarks list
```

## Run a benchmark

```bash
python -m benchmarks run <benchmark> --model <provider:model-id> [options]
```

`--model` is `provider:model-id`; any harness backend works, e.g.
`bedrock:us.anthropic.claude-opus-4-8`, `anthropic:claude-opus-4-8`,
`openai:gpt-4o`, `gemini:...`, `groq:...`.

Common options:

| Option | Meaning | Default |
|--------|---------|---------|
| `--limit N` | Run only the first N tasks | all |
| `--k N` | Attempts per task (for pass^k) | 1 |
| `--concurrency N` | Parallel attempts | 4 |
| `--output PATH` | Write per-attempt results as JSONL | none |
| `--resume` | Skip attempts already in `--output` | off |
| `--json` | Also print the summary as JSON | off |

Each run prints **pass@1**, **pass^k**, error rate, total cost, avg cost/task,
avg tokens/task, and **cost per solved task**.

## bfcl — function-calling accuracy

```bash
python -m benchmarks run bfcl \
    --model <provider:model-id> \
    --limit 40 --concurrency 8 --output runs/bfcl.jsonl

# pass^k reliability: run each task 3 times
python -m benchmarks run bfcl --model <provider:model-id> --k 3 --limit 20
```

Data is fetched from the gorilla repo and cached under `data/` (no setup
needed). Optional env vars:

- `BFCL_DATA_DIR` — read from a local gorilla checkout instead of fetching.
- `BFCL_RAW_BASE` — override the upstream URL if the path moves.

## tau-bench — multi-turn tool use vs. a simulated user

Runs an agent model (your harness) against an LLM user-simulator. The simulator
is routed through litellm, so it works with any provider litellm supports — set
`TAU_USER_PROVIDER` and `TAU_USER_MODEL` to your provider and model, plus
whatever credentials that provider needs (an API key for hosted providers, or
the provider's own credential mechanism).

```bash
export TAU_USER_PROVIDER=<provider>          # any litellm provider
export TAU_USER_MODEL=<model-id>             # a model that provider serves
export <PROVIDER_CREDENTIALS>=...            # e.g. the provider's API key
python -m benchmarks run tau-bench \
    --model <provider:model-id> \
    --limit 20 --k 3 --output runs/tau.jsonl
```

Configure via env vars:

- `TAU_ENV` — `retail` (default) or `airline`.
- `TAU_USER_PROVIDER` / `TAU_USER_MODEL` — the user-simulator backend and model.
- `TAU_TASK_SPLIT` — task split (default `test`).
- `TAU_MAX_STEPS` — max agent steps per task (default 30).
