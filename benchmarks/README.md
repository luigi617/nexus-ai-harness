# benchmarks

Run benchmarks against agents built on `nexus-ai-harness`.

## Install

```bash
pip install -e '.[benchmarks]'
```

This installs every benchmark's dependencies. Suites import their deps at
module top, so `import benchmarks` (and `python -m benchmarks list`) requires
the extra. Some suites need more at run time: GPQA and GAIA use gated Hugging
Face datasets (authenticate with `huggingface-cli login`), and swe-bench needs
a running Docker daemon.

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

## tau2-bench — dual-control multi-turn tool use

The successor to tau-bench: the agent and a simulated user share control of the
environment, and tau2's evaluator scores the finished trajectory. Needs a
user-simulator model.

```bash
export TAU2_USER_MODEL=<model-id>        # the user-simulator backend
export <PROVIDER_CREDENTIALS>=...
python -m benchmarks run tau2-bench --model <provider:model-id> --limit 20 --k 3
```

- `TAU2_DOMAIN` — domain/task set (default `airline`).
- `TAU2_TASK_SPLIT` — task split (default: the domain's full set).
- `TAU2_USER_MODEL` — the user-simulator model (default `gpt-4o`).
- `TAU2_MAX_STEPS` — max agent steps per task (default 30).
- `TAU2_TIMEOUT_S` — per-task wall-clock limit (default 300).

## humaneval — code generation graded by execution

Single-turn: the model completes a function; the completion is run against the
problem's unit tests in a timeboxed subprocess.

```bash
python -m benchmarks run humaneval --model <provider:model-id> --limit 20
```

- `HUMANEVAL_TIMEOUT_S` — per-completion execution timeout (default 15).

## gpqa — graduate-level multiple-choice science QA

Single-turn: the model answers a shuffled multiple-choice question; graded by
exact letter match. The GPQA dataset is gated — authenticate with Hugging Face.

```bash
huggingface-cli login                    # gated dataset
python -m benchmarks run gpqa --model <provider:model-id> --limit 50
```

- `GPQA_CONFIG` — dataset config (default `gpqa_diamond`).

## gaia — general assistant tasks

Agentic: the model uses tools to answer, graded by GAIA's normalized exact
match. The harness ships filesystem/shell tools and (when a task has an attached
file) copies it into a sandboxed workspace; full GAIA also needs web/browse
tools you supply. The dataset is gated — authenticate with Hugging Face.

```bash
huggingface-cli login                    # gated dataset
python -m benchmarks run gaia --model <provider:model-id> --limit 20
```

- `GAIA_CONFIG` — dataset config (default `2023_all`).
- `GAIA_SPLIT` — split (default `validation`).
- `GAIA_MAX_STEPS` — max agent steps per task (default 30).
- `GAIA_TIMEOUT_S` — per-task wall-clock limit (default 600).

## swe-bench — fix a real GitHub issue

Agentic: the repo is cloned at its base commit into a sandboxed workspace, the
agent edits it with the filesystem/shell tools, and the model's `git diff` is
graded by the project's `FAIL_TO_PASS`/`PASS_TO_PASS` tests via the official
`swebench` Docker harness, so grading needs a running Docker daemon.

```bash
python -m benchmarks run swe-bench --model <provider:model-id> --limit 5 \
    --task-timeout 2400 --output runs/swe.jsonl
```

- `SWE_BENCH_DATASET` — dataset (default `princeton-nlp/SWE-bench_Verified`;
  set to `princeton-nlp/SWE-bench_Lite` for the smaller split).
- `SWE_BENCH_MAX_STEPS` — max agent steps per task (default 50).
- `SWE_BENCH_TIMEOUT_S` — per-task wall-clock limit (default 1800).
- `SWE_BENCH_ALLOW_NETWORK` — allow the agent's shell network access (default
  `1`; set `0` to deny).
