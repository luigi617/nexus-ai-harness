# Memory

Memory lets an agent save notes and recall them in later sessions. Register a
memory store plus the memory tools, and the agent can use them on its own.

```python
from nexus_ai_harness.plugins.memory import FileMemoryStore
from nexus_ai_harness.plugins.tools import Remember, Recall, Forget

harness.use(FileMemoryStore())          # defaults to ~/.nexus-ai-harness/memory
harness.use(Remember()).use(Recall()).use(Forget())
```

Pass a path to store memories elsewhere: `FileMemoryStore("./my-memory")`.

## The tools

- **`remember`**: save a note.
- **`recall`**: search saved notes.
- **`forget`**: delete a note.

The default harness includes memory already, so you only wire it up when
building your own.

## How `FileMemoryStore` stores notes

Each note is one markdown file, `<dir>/<id>.md`, with a short header:

```text
---
id: "mem_ab12"
created_at: 1700000000.0
format: 2
---
the note text
```

- **The file name is the id.** The `id` line in the header is only there for
  people reading the file. Ids can't be blank, can't contain control
  characters, and can't point outside the store directory.
- **Text is saved exactly as given.** Only the header at the very top of the
  file is parsed, and it ends at the first `---` line, so a note can contain
  its own `---` lines without breaking. The store adds one newline after the
  text and strips exactly one when reading, so nothing else changes. (The
  `remember` tool still trims surrounding whitespace before saving.)
- **Older files still load.** Files without `format: 2`, written by earlier
  versions or by hand, load as before: line endings become `\n` and
  surrounding whitespace is trimmed.
- **Saves are atomic.** Each save goes to a temporary file, is flushed to disk,
  and then replaces the old file in one step, so a crash never leaves a
  half-written note. Saves and deletes also lock `<dir>/.lock` with `flock`, so
  several processes can share one directory safely. Where `flock` isn't
  available (for example on Windows), only threads within a single process are
  coordinated. New files are created readable by their owner only.

## How `recall` ranks results

Search splits the query and each note into words, ignoring case and
punctuation, and ranks notes with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25):
notes that mention rarer query words, or mention them more often, rank higher.
Words must match whole, so `java` does not match `javascript`. Notes with no
matching words are left out. Ties go to the newest note, then to the id in
alphabetical order. A blank query returns the newest notes.

Parsed notes are cached in memory. Each file is re-read only when its size,
modification time, or inode changes, so repeated searches are cheap, and edits
made by other processes still show up.
