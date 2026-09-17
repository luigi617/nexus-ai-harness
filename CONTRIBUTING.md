# Contributing

Install the project with its dev tools:

```bash
pip install -e ".[dev]"
```

Before opening a PR, make sure the checks pass:

```bash
ruff check .          # lint
ruff format .         # format
mypy                  # type check
pytest                # tests
```

Follow the [code style](docs/code-style.md) for comments and docstrings.
