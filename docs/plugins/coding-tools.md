# Coding tools

The coding tools let an agent explore, change, and run code in a workspace
folder. Register a sandbox rooted at the folder, then the tools you want:

```python
from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
from nexus_ai_harness.plugins.tools import (
    EditFile, Glob, Grep, ListDir, ReadFile, Shell, WriteFile,
)

harness.use(WorkspaceSandbox("./repo"))
for tool in (ReadFile(), ListDir(), Grep(), Glob(), WriteFile(), EditFile(), Shell()):
    harness.use(tool)
```

`default_harness(model, workspace="./repo")` wires all of them for you.

Every path is confined to the sandbox root (symlinks included); without a
sandbox the tools confine to the process's working directory instead.

## The tools

| Tool | What it does | Mutates? |
|---|---|---|
| `read_file` | Read lines `offset` to `offset + limit` (default the first 2000), numbered like `cat -n`. | No |
| `list_dir` | List a directory; directories end in `/`. | No |
| `grep` | Regex search (Python `re`) over files; prints `path:line:text`. | No |
| `glob` | Find files by pattern, such as `**/*.py`, sorted. | No |
| `write_file` | Create or overwrite a whole file. | Yes |
| `edit_file` | Replace exact text in a file, one or several edits at a time. | Yes |
| `shell` | Run a shell command; `cd` and `export` carry over. | Yes |

The default harness trusts the read-only tools and asks the approver before
every call to a mutating one. Keep it that way unless you have a reason not to.

### `read_file`

Long files are read a range at a time. When more lines remain, the result ends
with a note telling the model which `offset` to pass next. Very long lines are
clipped, and a result never cuts a line or a character in half. Memory stays
bounded even for a huge single-line file. For files over 20 MB, reading stops
once the range is filled, so the note says more lines exist instead of giving
the total. Pass `line_numbers: false` to get the raw text.

Lines are counted at `\n`, `\r\n`, and a lone `\r`, the same way in
`read_file`, `grep`, and `edit_file`, so a line number from one works in the
others. Form feeds don't start a new line. FIFOs and device files are refused
rather than read.

### `edit_file`

`old_string` must match the file exactly and occur exactly once, unless
`replace_all` is set. If the text is missing or ambiguous, the tool returns an
error saying so and writes nothing. Pass `edits` to apply several replacements
in order. They are all-or-nothing: if one fails, the file is left untouched. The
file is replaced atomically and keeps its permissions. For a CRLF file, the
model's `\n` line endings still match.

### `grep` and `glob`

Both are pure Python, so they need no `grep` or `rg` binary. They skip binary
files and junk directories (`.git`, `node_modules`, `__pycache__`, `.venv`,
caches) unless the pattern names one, and they cap their results. For `grep`,
a `glob` without a `/` (such as `*.py`) matches file names at any depth. For
`glob`, `*` stays within one directory and `**` spans any number of them.

### `shell`

Commands run through `bash` (or `sh` if `bash` is missing), so pipes,
redirects, `&&`, and globs all work. Each session remembers its working
directory and exported variables between calls, so `cd src` in one call is
where the next call starts. If a command leaves the workspace root, the next
call starts back at the root.

Standard error is merged into standard output in the order it was written, and
standard input is closed. Long output keeps its beginning and end. The limits
are configurable:

```python
Shell(default_timeout=30, max_timeout=600, max_output=30_000, persist_env=True)
```

On timeout, the whole process group is killed (children included) and the
result starts with `exit=124 (timed out after Ns)`, with or without a sandbox.

A background job that still writes to the output (`server &`) would otherwise
keep the call open until the timeout. Instead, once the command itself exits,
the tool waits a second for remaining output and then stops the job, saying so
in the result. To leave a process running, redirect its output:
`server > server.log 2>&1 &`. Under the bubblewrap backend, redirecting isn't
enough: its own PID namespace kills every process in the sandbox, backgrounded
or not, the moment the shell exits, so no command run through it outlives the
call that started it.

Parallel tool calls are safe. Edits to the same file are serialized, so both
edits land. A shell call only updates the session's directory when it
actually changed it.

## Command policy for shell strings

`WorkspaceSandbox(allowed_commands=..., denied_commands=...)` checks programs by
name. A shell string can hold many programs, so the sandbox scans it first and
checks each one it finds. The scan covers:

- Every command in a pipeline, `;`/`&&`/`||` list, subshell, `if`/`while`/`for`
  body, and `{ ...; }` group.
- Programs behind wrappers: `env`, `nohup`, `nice`, `timeout`, `stdbuf`,
  `xargs`, `exec`, `command`, and `find -exec`.
- Redirections, variable assignments, and heredoc bodies are treated as data
  and skipped.

Some constructs can run code the scan can't see: command or process
substitution (`$(...)`, backticks, `<(...)`), `eval`/`source`, arithmetic,
`[[ ]]`, `case`, unquoted heredocs that contain expansions, `env -S`, `set -a`,
and a program name stored in a variable or produced by a glob. The policy
handles them like this:

- **With an allowlist**, these constructs are refused. So is any command that
  sets a variable, whether by `NAME=value`, `export`/`declare`/`local`/
  `typeset`/`readonly`, `read`/`mapfile`/`readarray`/`getopts`, a `for`/
  `select` loop variable, or `{NAME}>file`, because a variable like `PATH` or
  `GIT_EXTERNAL_DIFF` can make an allowed name run something else. Output
  redirections are checked too, because off macOS nothing else stops the shell
  itself from writing a file: a target must resolve inside the root, and
  `/dev/tcp`/`/dev/udp` are refused unless `allow_network` is set. A relative
  target after `cd` is checked against every directory the `cd` chain could
  leave the shell in. That only works for plain relative `cd` targets without
  `..`, and not inside `while` loops or functions; anything else is refused.
- **With only a denylist**, these constructs are allowed. A denylist is a
  best-effort guard for shell strings, not a boundary.

The policy matches names, not what they resolve to. `./ls` passes an allowlist
that contains `ls`. It also can't limit what an allowed interpreter such as
`python` or `bash` does with its own arguments, or where an allowed program such
as `cp` writes. Redirect targets are checked when the command is checked, so a
symlink the command itself creates is not seen. For hard isolation, rely on the
sandbox's write confinement and network denial, and on the approver.

The interpreter itself needs no allowlist entry. If you put `sh` or `bash` in
`denied_commands`, shell strings are turned off entirely. A custom `Sandbox`
that doesn't implement `run_shell` still works: the tool falls back to running
the command as a single argument vector through `run_command`.
