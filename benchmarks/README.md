# benchmarks

Run benchmarks against agents built on `nexus-ai-harness`.

## Install

```bash
pip install -e . --group benchmarks
```

This installs every benchmark's dependencies. Some suites need more at run time: GPQA and GAIA use gated Hugging
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
problem's unit tests in an isolated, timeboxed process.

```bash
python -m benchmarks run humaneval --model <provider:model-id> --limit 20

# force the Docker backend and a tighter memory cap
python -m benchmarks run humaneval --model <provider:model-id> \
    --set isolation=docker --set memory_limit_mb=512
```

Set these with `--set key=value` or in the `humaneval:` section of a `--config`
YAML file (see `benchmarks.example.yaml`):

- `timeout_s` — per-completion wall-clock timeout (default 15). It also sets
  the CPU-time limit (rounded up, plus one second).
- `isolation` — how model-generated code runs (default `auto`):
  - `sandbox` — the harness's `WorkspaceSandbox`, rooted at a throwaway
    directory with network denied. On macOS, `sandbox-exec` enforces this in
    the kernel (writes are confined to the directory; network is blocked).
    Elsewhere, write and network confinement are best-effort until the sandbox
    gains a kernel backend for that platform. On POSIX the child also gets
    resource limits (CPU time, address space, file size, process count, no core
    dumps) and a scrubbed environment (no API keys or other host variables).
    Limits the OS can't enforce are skipped, for example `RLIMIT_AS` on macOS.
    The macOS profile narrows only writes and network: it does not stop the
    code from exec'ing another binary or asking launchd, for example through
    `/usr/bin/open`, to start an app outside the sandbox. Use `docker` for
    code you don't trust.
  - `docker` — a throwaway container (`--network none`, read-only root, a
    small `/tmp` tmpfs, all capabilities dropped, an unprivileged user, and
    memory, CPU, and pids limits). Pulls the image if it is missing. Falls
    back to `sandbox` with a warning if Docker is unusable.
  - `auto` — picks the safest backend that's ready: `docker` if the daemon is
    reachable and the image is already local (it never pulls), otherwise
    `sandbox`. If that `sandbox` can't confine writes and network on this
    platform, a warning says so and suggests `isolation=docker`.
  - `none` — the unconfined legacy subprocess. Use it only for debugging.

  The mode is resolved once per run, when the first harness is built and off
  the event loop, so a Docker probe or pull doesn't stall other attempts.
  Checks also run on a worker thread.
- `memory_limit_mb` — memory cap in MiB (default 1024, minimum 64 — sandbox
  mode's `RLIMIT_AS` needs well above Docker's own 6 MiB floor just to start
  the interpreter; `none` for unlimited).
- `max_file_size_mb` — largest file the code may write, in MiB (default 16,
  minimum 1; `none` for unlimited).
- `max_processes` — extra processes the code may spawn (default 0, must not be
  negative; `none` for unlimited). In `docker` mode it bounds the container's
  pid count. In `sandbox` mode it becomes `RLIMIT_NPROC`, which the kernel
  counts per user (and per thread on Linux). So the default `0` forbids forking
  and threads, which HumanEval solutions never need. A positive value is added
  to the number of tasks your user already runs when the check starts. That
  count is shared, so the allowance is only approximate while other checks run
  concurrently.
- `docker_image` — the image for `docker` mode (default: the official
  `python:<major>.<minor>-slim` image matching the host interpreter).

Scoring is the same in every mode: a completion passes if and only if its
program exits 0 within the timeout. Each attempt's `detail.isolation` in the
`--output` JSONL records which backend actually ran. `detail.kernel_confined`
records whether that backend confined file writes and network in the kernel:
true for `docker` and for `sandbox` on macOS, false for `sandbox` elsewhere and
for `none`. If the Docker backend itself fails (exit 125-127, for example a
daemon error or an image without `timeout`), the attempt is recorded as an
error rather than scored as a model failure.

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
`swebench` Docker harness. Needs `datasets` + `swebench` + a running Docker
daemon (`pip install -e . --group benchmarks`).

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
