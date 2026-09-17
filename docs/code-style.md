# Code style

The coding conventions for this repo.

## Formatting and tooling

- `ruff format` owns formatting; don't hand-format against it. Line length is 88.
- `ruff check` must pass (lint, import order, docstring format).
- `mypy` must pass. Annotate public function signatures; prefer `X | None` over
  `Optional[X]`.
- Start every module with `from __future__ import annotations`.

## Imports

- Use absolute imports (`from protocols.model import Model`), not relative ones.
- Let `ruff` order imports.

## Structure

- Protocols (interfaces) live in `protocols/`, concrete plugins in `plugins/`,
  shared value/data types in `core/`, orchestration in `harness/` and
  `services/`.
- Depend on protocols, not concrete implementations, wherever a seam exists.

## Comments

The one-line rule: **comments say _why_; they never restate the code.**

- **Explain intent, not mechanics.** A comment states *why* the code exists or
  what non-obvious thing it guards against. Never narrate what the code does. 
  If a comment only restates the line below it, delete it.
- **One line maximum.** A comment is a single line. If an operation seems to
  need a paragraph to explain, that is a signal to extract a well-named function
  or simplify — do that instead of writing a longer comment.
- **Comment only the non-obvious.** If you would have to explain it at code
  review, comment it; otherwise leave it out. Don't comment self-evident code.
- **Inline format.** Start an inline comment at least two spaces after the code,
  then `#` and one space.

```python
# Good — intent the reader can't infer from the code:
child_session._interrupt = self._session._interrupt  # interrupting root stops subagents

# Bad — restates the code:
i += 1  # increment i by one
```

## Docstrings

A docstring describes **what a component does and how to use it** — enough for a
reader to use it without reading its body. It does not expose internal
implementation details. Use a docstring, not a comment, for a public contract.

### When a docstring is required

Write one when the component is any of: **public API**, **nontrivial in size**,
or has **non-obvious logic**. Trivial private helpers may omit it. Omit a
docstring that would only repeat the name, and skip them for test modules,
`TestCase` subclasses, and `test_*` methods.

### Format

- Triple double-quotes: `"""..."""`.
- A one-line summary on the **same line as the opening quotes**, ending in `.`,
  `?`, or `!`, and no longer than 88 characters.
- For more detail, add a blank line after the summary, then the body aligned to
  the first quote.
- Pick descriptive (`Returns the token.`) or imperative (`Return the token.`)
  voice and keep it consistent within a file. A `@property` uses a noun phrase
  (`"""The resolved model."""`).

### Sections — functions and methods

Use these headings (each ending in a colon) when they apply:

- `Args:` — each parameter by name; include the type only if it isn't
  annotated; list variadics as `*args` / `**kwargs`.
- `Returns:` (or `Yields:` for generators) — describe the semantics of the
  value. Omit it when the function only returns `None`, or when the summary
  already states what is returned.
- `Raises:` — exceptions that are part of the interface. Do **not** document
  exceptions raised only when the API is used incorrectly.

```python
def route(self, history: list[Message], ctx: Context) -> Model:
    """Pick the model that should handle this turn.

    Args:
        history: The conversation so far, oldest message first.
        ctx: The active run context, used to resolve candidate models.

    Returns:
        The chosen model, drawn from ``ctx.all(Model)``.

    Raises:
        LookupError: If no model is registered.
    """
```

### Sections — classes

- Start with a one-line summary of what an *instance* represents. Don't state
  that it is a class.
- Document public attributes (not properties) in an `Attributes:` section, using
  the same format as `Args:`.
- An exception class describes what the exception *represents*, not the
  situation in which it is raised.

## Punctuation and grammar

Write comments and docstrings as narrative text: proper capitalization and
punctuation, complete sentences preferred.
