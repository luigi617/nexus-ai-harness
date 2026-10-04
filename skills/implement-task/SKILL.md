---
name: implement-task
description: Implement a task in this repo end to end — learn repo conventions from AGENTS.md/CONTRIBUTING.md/README.md/docs, make the change, verify it, and (for code changes) open a PR on a new branch using the PR template.
---

# Implement a task

A procedure for an AI coding agent to implement a task in this repository
correctly on the first pass, by grounding every decision in the repo's own
documentation rather than guessing. Not tied to any particular agent harness
or tool set — follow it with whatever read/write/shell/git/GitHub tools are
available.

## When to use

The user (or caller) hands you a task description for this repo: a bug fix,
a new feature, a refactor, a docs change, anything that may touch the code.

## Steps

### 1. Learn how the repo works

Read, in this order, before writing any code:

1. `AGENTS.md` — doc map and project structure
2. `CONTRIBUTING.md` — build/lint/test commands, changelog requirement
3. `README.md` — what the project is, how it's used
4. Everything under `docs/`, in particular:
   - `docs/architecture.md` — before touching anything, to understand the
     plugin/protocol/harness/services layering
   - `docs/writing-a-plugin.md` — if the task adds or changes a plugin
   - `docs/code-style.md` — comment/docstring/formatting conventions
   - `docs/plugins/*.md` — the specific plugin family being touched (models,
     routers, memory, subagents, skills, evaluators, interventions)

Do not guess an API from memory — confirm it against the doc or the source
under `src/nexus_ai_harness/`.

### 2. Classify the task

- **Code change** (bug fix, new plugin/protocol, refactor, etc.) → follow
  Steps 3–7.
- **Non-code** (answer a question, investigate, one-off read) → do the work
  and report back; skip branching/PR steps entirely.

### 3. Branch

Before branching, check the working tree is clean (`git status`); stash or
flag anything unexpected rather than discarding it.

```bash
git checkout main && git pull
git checkout -b <short-descriptive-branch-name>
```

### 4. Implement

- Follow `docs/code-style.md` exactly (comments explain *why* only, one line;
  `from __future__ import annotations`; absolute imports; depend on protocols
  not concrete implementations where a seam exists).
- Match existing patterns in neighboring files under `core/`, `protocols/`,
  `plugins/`, `services/`, `harness/`, or `graph/` — whichever layer the task
  touches.
- Add or update tests under `tests/` for any behavior change (required by
  `AGENTS.md`).
- Keep the diff scoped to the task — no speculative abstractions or unrelated
  cleanup.

### 5. Verify

Run the full check suite and fix any failures before moving on:

```bash
ruff check .
ruff format .
mypy
pytest
```

If the change needs the `benchmarks` extra or touches a specific plugin
family, run the relevant subset too. Don't force the tau2 suite outside
Python 3.12 — it's expected to skip there.

### 6. Self-review

Re-read your own diff (`git diff main...HEAD`) as if reviewing someone else's
change:

- Does it actually solve the stated task, with no leftover TODOs or
  half-finished branches of logic?
- Does every changed public API have a docstring per `docs/code-style.md`?
- Are protocols depended on instead of concrete plugins where that seam
  exists?
- Will you need a changelog fragment (next step covers this)?
- Rerun `pytest` once more after any fix-ups from this review.

### 7. Commit, push, and open the PR

```bash
git add <files>
git commit -m "<concise, why-focused message>"
git push -u origin <branch-name>
gh pr create --title "<title>" --body "$(cat .github/PULL_REQUEST_TEMPLATE.md)"
```

Fill in the template's `## Related issue`, `## Summary`, and `## Type of
change` sections with real content — don't leave placeholders — and check off
the `## Checklist` items that are genuinely true. If `gh` isn't available,
push the branch and open the PR through whatever Git hosting interface is
available, using the same template content.

Then add the changelog fragment named after the real PR number:

```bash
gh pr view --json number -q .number   # get <PR#>
```

Write `.changes/unreleased/<PR#>.json`:

```json
{
  "id": <PR#>,
  "description": "<one-line, user-facing description>"
}
```

Label the PR `skip-changelog` instead of adding a fragment only if the change
is invisible to users (CI, refactor, typo). Commit and push the fragment as a
follow-up commit on the same branch:

```bash
git add .changes/unreleased/<PR#>.json
git commit -m "Add changelog fragment for #<PR#>"
git push
```

Report the PR URL back. Do not merge it yourself.
