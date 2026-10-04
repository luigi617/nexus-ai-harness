---
name: review-pr
description: Review a PR in this repo against its own conventions (AGENTS.md, CONTRIBUTING.md, README.md, docs/) for bugs and potential issues, and leave inline comments on the specific lines.
---

# Review a PR

A procedure for an AI coding agent to review a pull request against this
repo's own documented conventions and leave inline comments on the specific
lines. Not tied to any particular agent harness or tool set — follow it with
whatever read/shell/`gh`/GitHub-API access is available.

## When to use

The user (or caller) hands you a PR number or URL in this repo to review for
bugs, potential issues, and convention violations.

## Steps

### 1. Learn how the repo works

Read before looking at the diff:

1. `AGENTS.md` — doc map, project structure, build/test commands, security
   considerations (never commit API keys; write/shell tools stay gated behind
   the permission approver)
2. `CONTRIBUTING.md` — required checks (`ruff check`, `ruff format`, `mypy`,
   `pytest`) and the per-PR changelog fragment requirement
3. `README.md` — what the project is for
4. Everything under `docs/`:
   - `docs/architecture.md` — plugin/protocol/harness layering the PR must
     respect
   - `docs/writing-a-plugin.md` — if the PR adds/changes a plugin
   - `docs/code-style.md` — comment, docstring, and formatting rules
   - `docs/plugins/*.md` — the specific plugin family the PR touches

### 2. Fetch the PR

```bash
gh pr view <PR#> --json title,body,headRefName,baseRefName,files
gh pr diff <PR#>
```

Check out the branch locally if you need to run checks or read full files
(not just the diff hunks):

```bash
gh pr checkout <PR#>
```

If `gh` isn't available, use whatever Git hosting API or web interface is
available to get the same diff and metadata.

### 3. Review against repo conventions

For every changed file, check:

- **Correctness** — logic bugs, off-by-one errors, missing edge cases, race
  conditions, incorrect protocol implementations.
- **Architecture fit** (`docs/architecture.md`) — does new code depend on
  protocols rather than concrete plugins where a seam exists? Is the
  loop/model/memory/tools/permissions/subagents boundary respected?
- **Plugin conventions** (`docs/writing-a-plugin.md`, relevant
  `docs/plugins/*.md`) — if a plugin was added or changed, does it correctly
  implement its protocol?
- **Code style** (`docs/code-style.md`) — `from __future__ import
  annotations` present, absolute imports, one-line why-only comments, public
  APIs have Google-style docstrings.
- **Tests** — is there a new/updated test for every behavior change
  (required by `AGENTS.md`)? Do the tests actually exercise the change, not
  just assert mocks?
- **Security** (`AGENTS.md`) — no committed secrets/API keys; any new
  write/shell-capable tool stays gated behind the permission approver and
  doesn't widen the sandbox without justification.
- **CI/process** — is there a `.changes/unreleased/<PR#>.json` fragment, or
  is the PR labeled `skip-changelog` for a user-invisible change?

Optionally confirm checks actually pass on the checked-out branch:

```bash
ruff check . && ruff format --check . && mypy && pytest
```

### 4. Leave inline comments

For each confirmed finding, comment on its exact line. With the GitHub CLI,
`gh pr review` only supports whole-PR comments through flags, so use the API
for line-anchored comments:

```bash
gh api repos/{owner}/{repo}/pulls/<PR#>/comments \
  -f body="<finding: what's wrong, why, and the fix>" \
  -f commit_id="<head sha from gh pr view --json headRefOid>" \
  -f path="<file path>" \
  -F line=<line number> \
  -f side=RIGHT
```

Then submit an overall review summarizing the findings and verdict:

```bash
gh pr review <PR#> --comment --body "<summary of findings>"
```

Use `--request-changes` instead of `--comment` only if you found a real bug,
security issue, or convention violation that should block merge — not for
nitpicks. If `gh` isn't available, post equivalent line comments and a
summary review through whatever Git hosting API or web interface is
available. Never approve or merge the PR yourself.

### 5. Report

Summarize to the user what was found and what was posted — don't just say
"done," list the actual findings and link to the PR.
