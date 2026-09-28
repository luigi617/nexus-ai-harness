# Changelog fragments

Every PR adds one file to `unreleased/`, named after the PR number, e.g. `.changes/unreleased/42.json`:

```json
{
  "id": 42,
  "description": "Add retry support to the Bedrock model."
}
```

When a release is prepared, `scripts/changelog.py release` turns all fragments
into a new `CHANGELOG.md` section and deletes them. See
[CONTRIBUTING.md](../CONTRIBUTING.md#changelog).
