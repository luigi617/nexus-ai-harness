---
name: address-pr-feedback
description: Given a PR, understand what it's doing (using repo conventions per the PR review skill), address the open review feedback on its branch, and push the fixes.
---

# Address PR feedback

A procedure for an AI coding agent to pick up an existing pull request,
understand its intent and the repo conventions it must follow, resolve the
outstanding review feedback on its branch, and push the result. Not tied to
any particular agent harness or tool set — follow it with whatever
read/write/shell/git/GitHub tools are available.

## When to use

The user (or caller) hands you a PR number or URL in this repo whose reviewers
left comments that still need addressing.

## Steps

### 1. Learn how the repo works

Read the same documents the `review-pr` skill uses to judge a PR, so your
fixes land on the right side of the repo's own conventions:

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

### 2. Understand the PR

```bash
gh pr view <PR#> --json title,body,headRefName,baseRefName,files,commits
gh pr diff <PR#>
gh pr checkout <PR#>
```

Read the PR title, description, and diff until you understand *what the PR is
trying to do*, not just the mechanics of the patch — you'll need that intent
to resolve feedback correctly rather than mechanically.

### 3. Collect the feedback

Pull every piece of outstanding review feedback — inline comments, review
bodies, and `--request-changes` reviews:

```bash
gh api repos/{owner}/{repo}/pulls/<PR#>/comments
gh api repos/{owner}/{repo}/pulls/<PR#>/reviews
```

For each comment:

- Note the file, line, and what the reviewer is asking for.
- Skip comments already marked resolved/outdated or that a later commit on
  the branch already addressed — check the current diff before acting.
- If a comment is ambiguous or asks for something outside the PR's scope,
  use judgment grounded in the repo docs from Step 1; don't guess wildly.

### 4. Address the feedback

On the checked-out PR branch:

- Make the smallest change that correctly resolves each comment — no
  speculative abstractions or unrelated cleanup beyond what was asked.
- Follow `docs/code-style.md` exactly (comments explain *why* only, one line;
  `from __future__ import annotations`; absolute imports; depend on protocols
  not concrete implementations where a seam exists).
- Add or update tests under `tests/` for any behavior change (required by
  `AGENTS.md`).
- If a comment is about missing tests, docs, or a changelog fragment, add
  them rather than just replying.

### 5. Verify

Run the full check suite and fix any failures before moving on:

```bash
ruff check .
ruff format .
mypy
pytest
```

Don't force the tau2 suite outside Python 3.12 — it's expected to skip there.

### 6. Commit and push

```bash
git add <files>
git commit -m "<concise, why-focused message>"
git push
```

Keep commits scoped to feedback fixes; don't fold in unrelated changes.

### 7. Reply and report

Reply to each addressed review comment (or post one summary comment) stating
what changed and where:

```bash
gh pr comment <PR#> --body "<summary of what was fixed, per comment>"
```

Report back to the user: which comments were addressed, which (if any) were
skipped and why, and confirm the push succeeded. Do not resolve conversations,
approve, or merge the PR yourself.
