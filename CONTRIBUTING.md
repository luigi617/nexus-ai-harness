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

## Releasing

1. Move the `[Unreleased]` entries in [CHANGELOG.md](CHANGELOG.md) under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading and merge that to `main`.
2. Tag the release on `main` and push the tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The tag sets the package version. The release workflow fails if CHANGELOG.md
has no section for that version; otherwise it publishes to PyPI and creates the
GitHub release with that section as its notes.
