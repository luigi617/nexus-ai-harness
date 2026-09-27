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

Each entry prints the suite's config fields with their defaults — those are the
knobs you can override with `--set`.

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
| `--set KEY=VALUE` | Override a benchmark config field (repeatable) | none |
| `--output PATH` | Write per-attempt results as JSONL | none |
| `--resume` | Skip attempts already in `--output` | off |
| `--json` | Also print the summary as JSON | off |

Each run prints **pass@1**, **pass^k**, error rate, total cost, avg cost/task,
avg tokens/task, and **cost per solved task**.

## Configuring a run

Per-benchmark tuning lives on a typed config dataclass, not environment
variables. `python -m benchmarks list` prints each suite's fields and defaults.
Two ways to set them, and they compose (a `--set` flag wins over the file):

**Inline with `--set key=value`** (repeatable):

```bash
python -m benchmarks run tau-bench --model ... \
    --set env=airline --set max_steps=40 --set user_model=gpt-4o
```

`--set` handles strings, ints, floats, booleans (`true`/`false`/`1`/`0`),
optional `X | None` (pass `none`/`null` for `None`), and comma-separated lists
(e.g. `--set categories=simple,parallel`).

**From a YAML file with `--config path.yaml`**, keyed by benchmark name so one
file configures every suite:

```yaml
# benchmarks.yaml
tau-bench:
  env: airline
  max_steps: 40
  user_model: gpt-4o
swe-bench:
  dataset: princeton-nlp/SWE-bench_Lite
  max_steps: 60
bfcl:
  categories: [simple, parallel]   # YAML lists map to list fields directly
```

```bash
python -m benchmarks run tau-bench --model ... --config benchmarks.yaml
# override one field from the file for this run:
python -m benchmarks run tau-bench --model ... --config benchmarks.yaml --set max_steps=10
```

Only the running benchmark's section is read; a section it lacks is a no-op.
YAML values keep their native types (ints, bools, lists). Unknown keys and
uncoercible values fail loudly with a clear message.
`benchmarks/benchmarks.example.yaml` is a ready-to-copy template with every
suite's fields at their defaults.

Credentials and cache locations stay in the environment (they're not run
parameters): provider API keys, Hugging Face auth (`HF_TOKEN` or
`huggingface-cli login`), and cache dirs (`HF_HOME`).

## bfcl — function-calling accuracy

```bash
python -m benchmarks run bfcl --model <provider:model-id> \
    --limit 40 --concurrency 8 --output runs/bfcl.jsonl

# pass^k reliability: run each task 3 times
python -m benchmarks run bfcl --model <provider:model-id> --k 3 --limit 20

# offline: read from a local gorilla checkout instead of fetching upstream
python -m benchmarks run bfcl --model <provider:model-id> \
    --set data_dir=/path/to/gorilla
```

Data is fetched from the gorilla repo and cached under `data/` (no setup
needed).

## tau-bench — multi-turn tool use vs. a simulated user

Runs an agent model (your harness) against an LLM user-simulator. The simulator
is routed through litellm, so it works with any provider litellm supports — set
its provider/model with `--set` and export whatever credentials that provider
needs.

```bash
export <PROVIDER_CREDENTIALS>=...
python -m benchmarks run tau-bench --model <provider:model-id> \
    --set env=airline --set user_provider=openai --set user_model=gpt-4o \
    --limit 20 --k 3 --output runs/tau.jsonl
```

## tau2-bench — dual-control multi-turn tool use

The successor to tau-bench: the agent and a simulated user share control of the
environment, and tau2's evaluator scores the finished trajectory. Needs a
user-simulator model.

```bash
export <PROVIDER_CREDENTIALS>=...
python -m benchmarks run tau2-bench --model <provider:model-id> \
    --set domain=airline --set user_model=gpt-4o --limit 20 --k 3
```

## humaneval — code generation graded by execution

Single-turn: the model completes a function; the completion is run against the
problem's unit tests in a timeboxed subprocess.

```bash
python -m benchmarks run humaneval --model <provider:model-id> --limit 20 \
    --set timeout_s=30
```

## gpqa — graduate-level multiple-choice science QA

Single-turn: the model answers a shuffled multiple-choice question; graded by
exact letter match. The GPQA dataset is gated — authenticate with Hugging Face.

```bash
huggingface-cli login                    # gated dataset
python -m benchmarks run gpqa --model <provider:model-id> --limit 50 \
    --set subset=gpqa_diamond
```

## gaia — general assistant tasks

Agentic: the model uses tools to answer, graded by GAIA's normalized exact
match. The harness ships filesystem/shell tools and (when a task has an attached
file) copies it into a sandboxed workspace; full GAIA also needs web/browse
tools you supply. The dataset is gated — authenticate with Hugging Face.

```bash
huggingface-cli login                    # gated dataset
python -m benchmarks run gaia --model <provider:model-id> --limit 20 \
    --set split=validation --set max_steps=40
```

## swe-bench — fix a real GitHub issue

Agentic: the repo is cloned at its base commit into a sandboxed workspace, the
agent edits it with the filesystem/shell tools, and the model's `git diff` is
graded by the project's `FAIL_TO_PASS`/`PASS_TO_PASS` tests via the official
`swebench` Docker harness, so grading needs a running Docker daemon.

```bash
# Verified (default); switch to Lite for a smaller split
python -m benchmarks run swe-bench --model <provider:model-id> --limit 5 \
    --set dataset=princeton-nlp/SWE-bench_Lite \
    --task-timeout 2400 --output runs/swe.jsonl
```
