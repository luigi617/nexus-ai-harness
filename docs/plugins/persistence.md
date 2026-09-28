# Persistence

Persistence saves a session to disk so you can resume it later, even from a new
process, or branch it to try something else without losing the original.

## Saving and resuming

Register a `SessionStore` and the `AutoSave` hook:

```python
from nexus_ai_harness.plugins.persistence import AutoSave, FileSessionStore, resume

store = FileSessionStore("~/.nexus-ai-harness/sessions")   # one JSON file per session
harness = harness.use(store).use(AutoSave())

result = await harness.run("Remember the code ORCA-1.")

# Later, perhaps in another process:
session = resume(store, result.session.id)
await harness.run("What was the code?", session=session)
```

`resume` keeps the session's id, so continuing it updates the same saved file.

### When AutoSave writes

`AutoSave` saves at loop boundaries: when an iteration starts or completes and
just before each model call (each only if a message was added since the last
save), and always when the session ends. Writes grow with the number of turns
rather than the number of messages, and a resumed session never starts partway
through an iteration.

The model-call boundary is the one every loop passes through. `ChatLoop` emits
no iteration events, but the user's message is still on disk before the model
is asked, so a model call that raises doesn't lose it. If a run raises, messages
added after the last boundary (for example, an assistant reply whose tool calls
then failed) aren't saved; the snapshot stays at the last clean boundary.

| Option | Default | Effect |
|---|---|---|
| `every_message` | `False` | Also save after every added message (the behavior before this option existed). |
| `strict` | `False` | Raise save failures instead of warning, aborting the run. |

A failed save doesn't stop the run, but it isn't hidden either: it raises a
`PersistenceWarning` and emits a `SessionSaveFailed` event (with the session id
and the error), and the next boundary tries again, including when the failed
save was the final one at session end. Subagent sessions are never saved.

### How FileSessionStore writes

Each save serializes the snapshot first, so a value that can't be written as
JSON raises before anything on disk changes. The store then writes a uniquely
named temp file, fsyncs it, and atomically renames it over the old snapshot,
then fsyncs the directory where the platform allows. A crash, or two saves of
the same session at once, never leaves a partial file. Snapshot files are
readable by their owner only.

## Forking a session

To branch a saved session, fork it. The fork gets a new id and records where it
came from in `parent_id`; the original is left as it was:

```python
from nexus_ai_harness.plugins.persistence import fork_session

branch = fork_session(store, session_id)             # saved right away
await harness.run("Try the other approach.", session=branch)
assert branch.parent_id == session_id
```

`fork_session` returns `None` if the source doesn't exist and raises
`ValueError` rather than overwrite an existing `new_id`. It writes the fork with
`SessionStore.create`, which refuses an id that's already stored.
`FileSessionStore` makes that atomic: it hard-links the new file into place, so
two racing forks to the same id can't both succeed, and on case-insensitive
filesystems `V2` won't replace `v2`. `FileSessionStore` also rejects ids that
would map to a different file name, such as `./other`. To branch a live,
in-memory session, call `session.fork()`. It deep-copies the history and state
and carries over pending interventions.

Don't confuse this with `ctx.fork()`, which starts a subagent on an empty
session.

## Persisting plugin state

A snapshot always holds the history and the run's stop reason. Plugin state
(`ctx.state(cls)`) and pending interventions (`session.submit(...)`) are saved
only if their class opts in with `@persistable`, under a name that must never
change:

```python
from dataclasses import dataclass
from nexus_ai_harness.core.persistable import persistable

@persistable("my_plugin.counter")
@dataclass
class CounterState:
    calls: int = 0
```

A dataclass must hold only JSON values (strings, numbers, booleans, `None`,
lists, and dicts). For anything else, define `to_dict(self)` and a
`from_dict(cls, data)` classmethod instead. Fields that were removed from the
class since the snapshot was written are ignored when it loads.

Built-in state that opts in: `SummaryState` (the summarizing context manager's
cached summary) and the `InjectMessage` intervention.

Other state is skipped on purpose. If opted-in state can't be serialized or
restored, including when a custom `to_dict` or `from_dict` raises, that entry is
skipped with a `PersistenceWarning` instead of breaking the save. The rest of the
snapshot is still written, and the plugin asking for restored state gets a
default instance.
Saved state is restored the first time `ctx.state(cls)` asks for it, so it
doesn't matter whether the plugin was imported before `resume`. State that no
code asks for is carried into the next save unchanged.

## Snapshot format and versions

A snapshot is a JSON object:

```json
{
  "version": 2,
  "id": "sess_...",
  "parent_id": null,
  "history": [{"role": "user", "content": "hi", "id": "msg_...", "...": "..."}],
  "stop_reason": "completed",
  "state": {"summarizing.summary": {"upto": 0, "text": ""}},
  "interventions": [{"type": "interventions.inject_message", "data": {"...": "..."}}]
}
```

Files without a `version` come from before versioning and load as version 1.
When it loads, a snapshot is upgraded step by step to the current
`SCHEMA_VERSION` using the registered migrations. A snapshot newer than this
release supports raises `SnapshotVersionError`, a `ValueError`, instead of
quietly dropping data it doesn't understand.

To change the format, bump `SCHEMA_VERSION` in
`plugins/persistence/schema.py` and register a migration from the previous
version:

```python
from nexus_ai_harness.plugins.persistence import register_migration

@register_migration(2)
def _v2_to_v3(data: dict) -> dict:
    return {**data, "new_field": data.get("new_field", [])}
```

## Writing a store

Subclass `SessionStore` and implement `save`, `load`, `list_ids`, and `delete`.
They work with plain snapshot dicts. `create` has a default that checks `load`
and then calls `save`. That default isn't safe against concurrent writers, so
override it if your backend can create a key atomically. Persist JSON only, never pickle, and treat
the session id as untrusted input. `resume`, `fork_session`, and `AutoSave` work
with any store.
