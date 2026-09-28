# AGENTS.md

Guidance for AI coding agents working in this repository. Human-facing docs live
in [README.md](README.md) and [docs/](docs/README.md); this file points you to
the right one for the task at hand.

## Project overview

Nexus AI Harness is a plugin-based framework for building LLM agents. Every part
of an agent (loop, model, memory, tools, permissions, subagents, and more) is a
**plugin** that implements a **protocol** and is registered on a harness with
`.use(...)`. Plugins are resolved by protocol type, so any piece can be swapped
without touching the rest.

## Which docs to read, and when

Read the relevant doc **before** writing code for that area. Do not guess an API;
confirm it against the doc or the source.

| If you are... | Read first |
|---|---|
| Getting oriented in the codebase | [docs/architecture.md](docs/architecture.md) |
| Writing or changing any plugin | [docs/writing-a-plugin.md](docs/writing-a-plugin.md) |
| Writing comments, docstrings, or formatting code | [docs/code-style.md](docs/code-style.md) |
| Working on a model backend | [docs/plugins/models.md](docs/plugins/models.md) |
| Working on model selection | [docs/plugins/routers.md](docs/plugins/routers.md) |
| Working on long-term memory | [docs/plugins/memory.md](docs/plugins/memory.md) |
| Working on task delegation | [docs/plugins/subagents.md](docs/plugins/subagents.md) |
| Working on on-demand instructions | [docs/plugins/skills.md](docs/plugins/skills.md) |
| Working on logging, events, or usage metrics | [docs/plugins/observability.md](docs/plugins/observability.md) |
| Working on structured scoring | [docs/plugins/evaluators.md](docs/plugins/evaluators.md) |
| Steering or stopping a running agent | [docs/plugins/interventions.md](docs/plugins/interventions.md) |

## Project structure

Source lives under `src/nexus_ai_harness/`:

- `core/`: plain data types (messages, responses, run state).
- `protocols/`: the interfaces (abstract base classes) plugins implement.
- `plugins/`: concrete plugin implementations.
- `services/`: shared logic reused across plugins and the harness.
- `harness/`: the harness, session, run context, and registry.
- `graph/`: graph tracing support.

Tests are in `tests/`; benchmark suites are in `benchmarks/`.

## Build and test commands

Install with dev tools:

```bash
pip install -e . --group dev
```

Before finishing a change, run all checks and fix any failures:

```bash
ruff check .          # lint
ruff format .         # format
mypy                  # type check
pytest                # tests
```

## Code style

Full conventions are in [docs/code-style.md](docs/code-style.md). Key points:

- `ruff format` owns formatting; line length is 88.
- Start every module with `from __future__ import annotations`.
- Use absolute imports (`from nexus_ai_harness.protocols.model import Model`).
- Comments say **why**, never restate the code, and stay to one line.
- Public APIs need docstrings; follow the Google-style format in the doc.
- Depend on protocols, not concrete implementations, wherever a seam exists.

## Testing instructions

- Run the full suite with `pytest`; it must pass before a change is done.
- Some benchmark tests require the `benchmarks` extra. The tau2 suite runs only
  on Python 3.12 (it cannot import on 3.13 or install on 3.14) and is skipped
  automatically elsewhere; do not force it to run.
- Add or update tests for any behavior you change.

## Commit and pull request guidelines

- Every PR needs a changelog fragment at `.changes/unreleased/<PR#>.json`; CI
  fails without one. Label a PR `skip-changelog` only for changes users won't
  notice. Do not edit `CHANGELOG.md` directly. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security considerations

- Never commit API keys. Models read keys from environment variables; use `.env`
  (copied from `.env.example`), which is gitignored.
- Write and shell tools mutate state and stay gated behind the permission
  approver. Do not auto-approve them or widen the sandbox without good reason.
