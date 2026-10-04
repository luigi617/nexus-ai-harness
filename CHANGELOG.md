# Changelog

All notable changes to this project are documented here. The project uses
[Semantic Versioning](https://semver.org/).

This file is generated from the JSON fragments in `.changes/unreleased/` when a
release is prepared; don't edit release sections by hand. See
[CONTRIBUTING.md](CONTRIBUTING.md#changelog).

## [0.1.2] - 2026-10-04

- Enforce sandbox isolation on Linux with bubblewrap, add pluggable isolation backends with isolation/require_isolation options, and warn when commands run without OS isolation. ([#32](https://github.com/luigi617/nexus-ai-harness/pull/32))
- Make the file memory store crash-safe with atomic, locked writes, keep note text exactly as saved, and rank recall results with whole-word BM25 search. ([#33](https://github.com/luigi617/nexus-ai-harness/pull/33))
- Add library logging, a ModelCallCompleted event with call ids and durations on model and tool call events, token and cost totals on RunResult (including summarizer and router calls), and an opt-in LoggingHook for structured event logs. ([#34](https://github.com/luigi617/nexus-ai-harness/pull/34))
- Run HumanEval completions in isolation (WorkspaceSandbox with resource limits, or Docker) instead of an unconfined subprocess, configurable via the new `isolation` option. ([#35](https://github.com/luigi617/nexus-ai-harness/pull/35))
- Version session snapshots with migrations, persist opt-in plugin state and pending interventions, add session forking, and make AutoSave save at loop boundaries instead of every message (use AutoSave(every_message=True) for the old behavior); snapshot files are now owner-only. ([#36](https://github.com/luigi617/nexus-ai-harness/pull/36))
- Model backends now retry rate limits, 5xx responses, and dropped connections with backoff (honoring Retry-After) and raise typed ModelAPIError/RateLimitError/ContextLengthExceeded errors, and the built-in loops end a run with stop_reason "model_error" instead of raising when a model call still fails; Bedrock now uses botocore's standard retry mode, overriding AWS_MAX_ATTEMPTS/AWS_RETRY_MODE. ([#37](https://github.com/luigi617/nexus-ai-harness/pull/37))
- Budget the context by tokens (default_harness now compacts above 100k estimated tokens), fall back to dropping or truncating turns instead of sending an over-budget history when summarization fails, merge the summary into the first user message to keep role alternation, and add the ContextCompacted event and TokenEstimator protocol. ([#38](https://github.com/luigi617/nexus-ai-harness/pull/38))
- Add coding tools for agents working in a repository: edit_file for exact-string edits, grep and glob for code search, ranged and line-numbered read_file, and a shell that supports pipes and redirects and remembers its working directory and environment between calls. ([#39](https://github.com/luigi617/nexus-ai-harness/pull/39))

## [0.1.1] - 2026-09-28

- No user-facing changes.

## [0.1.0] - 2026-09-28

- Initial release: a plugin-based harness for building LLM agents, with plugins for loops, models, memory, tools, permissions, routers, subagents, and more. ([#29](https://github.com/luigi617/nexus-ai-harness/pull/29))
