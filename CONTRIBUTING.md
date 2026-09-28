# Contributing

Install the project with its dev tools:

```bash
pip install -e . --group dev
```

Before opening a PR, make sure the checks pass:

```bash
ruff check .          # lint
ruff format .         # format
mypy                  # type check
pytest                # tests
```

Follow the [code style](docs/code-style.md) for comments and docstrings.

## Changelog

Every PR adds a changelog fragment named after the PR number, e.g. `.changes/unreleased/42.json`:

```json
{
  "id": 42,
  "description": "Add retry support to the Bedrock model."
}
```

CI fails without one; label the PR `skip-changelog` for changes users won't
notice (CI, refactors, typos). Don't edit `CHANGELOG.md` directly.
