# Sandbox

A sandbox confines what tools can touch. `WorkspaceSandbox` pins file access
to one workspace folder and runs shell commands under OS-level isolation, so a
command can only write inside the workspace and can't reach the network.

```python
from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox

harness.use(WorkspaceSandbox("./workspace"))
```

## What it enforces

On every platform:

- **Path confinement.** File tools resolve paths under the root and reject
  anything that escapes it, including `..`, absolute paths, and symlinks.
- **Command policy.** Optional `allowed_commands` and `denied_commands`, matched
  against the program name.
- **A clean process.** Commands run without a shell, from the root, with a
  scrubbed environment and a mandatory timeout.

Where an isolation backend is available, the kernel also enforces:

- **Write confinement.** Writes are allowed only in the root and a private
  scratch folder exposed as `$TMPDIR`.
- **No network**, unless you pass `allow_network=True`.

Reads are not restricted, except that the Linux backend hides some host
folders (see below). To stop a command from reading and leaking files, limit
which commands may run.

## Isolation backends

The backend is picked automatically for the platform:

| Platform | Backend | How it isolates |
|---|---|---|
| macOS | `sandbox-exec` | A Seatbelt profile that denies writes and network. |
| Linux | `bubblewrap` | `bwrap` with a read-only `/`, writable binds of the root and scratch folder, private empty `/tmp` and `/run`, fresh `/dev` and `/proc`, separate PID and IPC namespaces, and no network. |

On Linux, install bubblewrap (for example `apt install bubblewrap`). It also
needs unprivileged user namespaces, which some containers disable. On Ubuntu
24.04 and later, AppArmor may block them for `bwrap` unless a profile allows
it. The sandbox checks that `bwrap` actually works before choosing it.

### Hidden folders on Linux

A read-only mount does not stop a program from connecting to a Unix socket
file. A reachable socket for D-Bus, Docker, or an ssh agent would let a command
ask a host service to write files or open connections for it. So the Linux
backend replaces these folders with private, empty ones:

- `/tmp`, which also gives commands a writable scratch `/tmp` that is thrown
  away afterwards;
- `/run`, and `/var/run` when it is a real folder rather than a link to `/run`;
- `$XDG_RUNTIME_DIR`, when it lives somewhere else.

Host files in those folders are invisible to commands, not just read-only. The
workspace root and `$TMPDIR` stay visible even when they live under `/tmp`.
Links at the top of `/run` that point elsewhere, such as NixOS's
`/run/current-system`, are recreated so programs reached through them still
run.

If a command needs something from a hidden folder, such as a virtualenv under
`/tmp`, expose it read-only:

```python
from nexus_ai_harness.plugins.sandbox import BubblewrapBackend, WorkspaceSandbox

WorkspaceSandbox(
    "./workspace",
    isolation=BubblewrapBackend(expose=["/tmp/venv"]),
)
```

Unix sockets in other places, such as your home folder or `/var/lib`, are still
reachable. If that matters, deny the programs that use them with
`denied_commands`, or plug in a stricter backend.

Check which backend is in use:

```python
sandbox = WorkspaceSandbox("./workspace")
sandbox.isolation_level     # "sandbox-exec", "bubblewrap", or "none"
sandbox.isolation_enforced  # False when writes and network are not confined
```

### When no backend is available

The sandbox still runs commands, but writes and network are not confined. It
does not fail silently: `isolation_enforced` is `False`, and an
`IsolationWarning` is emitted once per process.

To fail instead, require isolation. The sandbox then raises
`IsolationUnavailable` when it is created:

```python
WorkspaceSandbox("./workspace", require_isolation=True)
```

### Choosing a backend

Pass `isolation` to override the automatic choice. A named backend must be
available, or the sandbox raises `IsolationUnavailable`:

```python
WorkspaceSandbox("./workspace", isolation="bubblewrap")
WorkspaceSandbox("./workspace", isolation="none")  # opt out, no warning
```

To use your own mechanism, subclass `IsolationBackend` and pass an instance.
`wrap` turns a command into one that runs under your confinement:

```python
import shutil

from nexus_ai_harness.plugins.sandbox import IsolationBackend, IsolationSpec

class Firejail(IsolationBackend):
    name = "firejail"

    def available(self) -> bool:
        return shutil.which("firejail") is not None

    def wrap(self, argv, spec: IsolationSpec) -> list[str]:
        net = [] if spec.allow_network else ["--net=none"]
        writable = [f"--read-write={spec.root}", f"--read-write={spec.tmpdir}"]
        return ["firejail", "--quiet", "--read-only=/", *writable, *net, *argv]

WorkspaceSandbox("./workspace", isolation=Firejail(), require_isolation=True)
```
