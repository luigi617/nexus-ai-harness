from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.lifecycle import Lifecycle
from nexus_ai_harness.protocols.sandbox import (
    Sandbox,
    SandboxResult,
    SandboxViolation,
    ShellResult,
)
from nexus_ai_harness.services.process import (
    TIMEOUT_NOTE,
    TIMEOUT_RETURNCODE,
    run_process,
)
from nexus_ai_harness.services.shell import (
    WRITE_REDIRECTS,
    CommandScan,
    ShellInvocation,
    default_interpreter,
    scan_command,
)

# Exit code used when a command is killed for exceeding its timeout, matching
# the conventional shell code for a process terminated by ``timeout(1)``.
_TIMEOUT_RETURNCODE = TIMEOUT_RETURNCODE

_SHELL_NAMES = frozenset({"sh", "bash"})
"""Interpreter names whose denial also disables :meth:`run_shell`."""

_MAX_TRACKED_CDS = 4
"""Most ``cd`` commands a string may hold and still have redirects verified."""

_SAFE_DEVICES = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})
"""Write targets outside the root that reach no file."""


class WorkspaceSandbox(Sandbox, Lifecycle):
    """A :class:`Sandbox` confined to a single workspace root directory.

    Paths are resolved and rejected if they escape the root (symlinks included);
    this path confinement holds on every platform. Commands are checked against
    an optional allowlist and a denylist before running. On macOS the process is
    wrapped in a ``sandbox-exec`` profile confining writes to the root (plus a
    private scratch dir exposed as ``$TMPDIR``) and denying network by default.
    Off macOS (or where ``sandbox-exec`` is absent)
    OS-level isolation is best-effort: the command still runs with ``cwd`` at the
    root, a scrubbed environment, ``shell=False``, and a mandatory timeout, but
    write confinement and network denial are not kernel-enforced.

    Scope of confinement, by design:

    * The macOS profile narrows *writes* and network only — *reads* are not
      restricted (denying them would break loading the program's own binary), so
      callers who must prevent exfiltration should also restrict which commands
      may run via ``allowed_commands``/``denied_commands``.
    * Command policy matches ``argv[0]`` (and its basename); it does not see
      through wrapper interpreters such as ``bash -c``/``python -c``.
    * Shell strings (:meth:`run_shell`) are checked by statically scanning
      them: every program in every pipeline, list, subshell, and compound
      command is checked, looking through ``env``/``nohup``/``timeout``/
      ``xargs``/``exec``/``command``/``find -exec``. Constructs whose programs
      the scan cannot see (command or process substitution, ``eval``/``source``,
      unquoted heredocs with expansions, arithmetic, a program name held in a
      variable, ``env -S``, ``set -a``) are refused when an allowlist is set,
      as are variable assignments (``PATH=...``, ``export``, ``for`` loop
      variables) and output redirections that could land outside the root or
      on ``/dev/tcp``; all are allowed otherwise, so a denylist is a
      best-effort guard for shell strings. Policy matches names only; it
      cannot tell what a name (or ``./name``) resolves to, nor what an allowed
      program does with its own arguments (``cp x /etc`` is the program's
      write, not the shell's).
    * Subprocesses are confined via :meth:`run_command`; the ``resolve_path``
      check applies to the filesystem tools, not to arbitrary shell argv.
    * A timeout kills the command's whole process group, so a shell's
      children cannot outlive it.

    It is a :class:`~protocols.lifecycle.Lifecycle`: the private scratch dir it
    exposes as ``$TMPDIR`` is removed when the harness stops, so repeated tasks
    (e.g. a benchmark run) don't leak temp directories.

    Attributes:
        allow_network: Whether the sandboxed process may reach the network.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        allow_network: bool = False,
        allowed_commands: Iterable[str] | None = None,
        denied_commands: Iterable[str] = (),
    ) -> None:
        """Create a sandbox rooted at ``root``, creating the directory if absent.

        Args:
            root: The confinement root; expanded and resolved to a real path.
            allow_network: If ``False`` (default), network access is denied when
                the platform supports enforcing it.
            allowed_commands: If not ``None``, only these program names (matched
                against ``argv[0]``) may run; anything else is denied.
            denied_commands: Program names that are always denied, even if they
                appear in ``allowed_commands``.
        """
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        # A dedicated scratch dir exposed as $TMPDIR: outside the root so it
        # never pollutes the listing, private so the profile's write grant stays narrow.
        self._tmpdir = Path(tempfile.mkdtemp(prefix="nexus-sandbox-")).resolve()
        self.allow_network = allow_network
        self._allowed = (
            frozenset(allowed_commands) if allowed_commands is not None else None
        )
        self._denied = frozenset(denied_commands)

    @property
    def root(self) -> Path:
        """The resolved confinement root directory."""
        return self._root

    async def start(self, ctx: Context) -> None:
        """Recreate the private scratch dir, so the sandbox is restart-safe."""
        self._tmpdir.mkdir(parents=True, exist_ok=True)

    async def stop(self) -> None:
        """Remove the private scratch dir the sandbox handed out as ``$TMPDIR``."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def resolve_path(self, path: str) -> Path:
        """Resolve ``path`` under the root, rejecting anything that escapes it.

        Args:
            path: A relative or absolute path supplied by a tool or the model.

        Returns:
            The resolved absolute path, guaranteed to be inside the root.

        Raises:
            SandboxViolation: If the resolved path lands outside the root,
                whether via ``..`` segments, an absolute path, or a symlink.
        """
        candidate = (self._root / path).resolve()
        if not candidate.is_relative_to(self._root):
            raise SandboxViolation(f"path escapes sandbox root: {path!r}")
        return candidate

    def check_command(self, argv: list[str]) -> None:
        """Validate ``argv`` against the allowlist and denylist.

        Args:
            argv: The argument vector; ``argv[0]`` is the program name.

        Raises:
            SandboxViolation: If ``argv`` is empty, the program is denied, or an
                allowlist is configured and the program is not on it.
        """
        if not argv:
            raise SandboxViolation("empty command")
        program = argv[0]
        # Match argv[0] and its basename so a bare-name policy ("rm") also catches
        # "/bin/rm"; a coarse guard that does not see through "bash -c ...".
        names = {program, os.path.basename(program)}
        if names & self._denied:
            raise SandboxViolation(f"command not permitted: {program!r}")
        if self._allowed is not None and not (names & self._allowed):
            raise SandboxViolation(f"command not in allowlist: {program!r}")

    def run_command(
        self,
        argv: list[str],
        *,
        timeout: float,
        input: str | None = None,  # noqa: A002
    ) -> SandboxResult:
        """Run ``argv`` confined to the root and capture its output.

        The command is validated by :meth:`check_command`, then run with
        ``shell=False``, ``cwd`` set to the root, a scrubbed minimal
        environment, and standard input closed unless ``input`` is given. A
        timeout kills the process group and is converted into a
        :class:`SandboxResult` with ``timed_out`` set and a non-zero return code
        rather than being raised, so callers get uniform output.

        Args:
            argv: The argument vector to execute; ``argv[0]`` is the program.
            timeout: Mandatory wall-clock limit in seconds.
            input: Optional text written to the process's standard input.

        Returns:
            The command's exit status and captured output.

        Raises:
            SandboxViolation: If :meth:`check_command` rejects ``argv``.
        """
        self.check_command(argv)
        result = self._execute(argv, timeout=timeout, input=input)
        if result.timed_out:
            note = f"{TIMEOUT_NOTE} after {timeout}s"
            result.stderr = f"{result.stderr}\n{note}".strip()
        return result

    def check_shell(self, command: str, *, cwd: str | None = None) -> None:
        """Validate a shell command string against the allowlist and denylist.

        Every program the string runs that a static scan can identify is passed
        to :meth:`check_command`. When an allowlist is configured, the string is
        also refused if it can run code the scan cannot see (see the class
        docs), sets a variable (``NAME=``, ``export``, a ``for`` loop variable),
        or redirects output to a file outside the root or to ``/dev/tcp``,
        since each could make an allowed program, or the shell itself, do what
        the allowlist was meant to prevent.

        Args:
            command: The shell command string.
            cwd: The directory the command starts in, for resolving relative
                redirection targets (default: the root).

        Raises:
            SandboxViolation: If any identified program is not permitted, or an
                allowlist is set and the string cannot be fully verified.
        """
        scan = scan_command(command)
        for program in scan.programs:
            self.check_command([program])
        if self._allowed is None:
            return
        if scan.opaque:
            raise SandboxViolation(
                "cannot verify shell command against the allowlist: "
                f"{scan.opaque[0]}; simplify the command"
            )
        if scan.assignments:
            # PATH, LD_PRELOAD, GIT_EXTERNAL_DIFF... can make a permitted program
            # run arbitrary code, which would hollow out the allowlist.
            raise SandboxViolation(
                "setting environment variables (or a loop variable) is not "
                f"permitted with a command allowlist: {scan.assignments[0]!r}"
            )
        start = self.resolve_path(cwd) if cwd else self._root
        for op, target in scan.redirects:
            self._check_redirect(op, target, self._possible_cwds(start, scan))

    def _possible_cwds(self, start: Path, scan: CommandScan) -> list[Path] | None:
        """Every directory a redirection in ``scan`` might resolve against.

        Each ``cd`` may succeed or fail (``cd a || true``) or be undone by a
        subshell, so the cwd is ``start`` plus any in-order subset of the
        ``cd`` targets. That is only tractable when each target is literal,
        relative, and free of ``..``, nothing repeats, and there are few of
        them; otherwise ``None`` (unknown).
        """
        dirs = scan.directories
        if not dirs:
            return [start]
        if scan.repeats or len(dirs) > _MAX_TRACKED_CDS:
            return None
        parts: list[Path] = []
        for d in dirs:
            if d is None or os.path.isabs(d) or ".." in Path(d).parts:
                return None
            parts.append(Path(d))
        cwds = []
        for mask in range(1 << len(parts)):
            chosen = [p for i, p in enumerate(parts) if mask >> i & 1]
            cwds.append(self.resolve_path(str(start.joinpath(*chosen))))
        return cwds

    def _check_redirect(self, op: str, target: str, cwds: list[Path] | None) -> None:
        """Refuse a redirection that writes outside the root or opens a socket.

        Only called under an allowlist: the argv-only policy this replaces never
        let a permitted program write arbitrary files or connect out, and off
        macOS nothing else enforces that for the shell's own redirections.
        """
        if target.startswith(("/dev/tcp/", "/dev/udp/")) and not self.allow_network:
            raise SandboxViolation(f"network redirection is not permitted: {target!r}")
        if op not in WRITE_REDIRECTS or target in _SAFE_DEVICES:
            return
        if target.startswith("/dev/fd/"):
            return
        if os.path.isabs(target):
            self.resolve_path(target)
            return
        if cwds is None:
            raise SandboxViolation(
                "cannot verify a relative redirection target after this 'cd' "
                f"with a command allowlist: {target!r}; cd to a plain relative "
                "directory or use a path from the start directory"
            )
        for cwd in cwds:
            self.resolve_path(str(cwd / target))

    def run_shell(
        self,
        command: str,
        *,
        timeout: float,
        cwd: str | None = None,
        env: Mapping[str, str | None] | None = None,
    ) -> ShellResult:
        """Run a shell command string confined to the root.

        The string is validated by :meth:`check_shell` and run by ``bash`` (or
        ``sh``) under the same confinement as :meth:`run_command`. The
        interpreter itself needs no allowlist entry, since the programs it runs
        are what the policy checks, but denying ``sh`` or ``bash`` disables
        shell strings entirely. Standard error is merged into standard output.

        Args:
            command: The shell command string.
            timeout: Mandatory wall-clock limit in seconds.
            cwd: Directory to start in, confined to the root (default: root).
            env: Variables to set (or unset, when ``None``) before running.

        Returns:
            The exit status, combined output, final confined cwd, and the
            environment changes the command made.

        Raises:
            SandboxViolation: If the policy rejects the command or ``cwd``
                escapes the root.
        """
        interpreter = default_interpreter()
        names = {interpreter, os.path.basename(interpreter)} | _SHELL_NAMES
        if names & self._denied:
            raise SandboxViolation("shell commands are not permitted")
        start = self.resolve_path(cwd) if cwd else self._root
        self.check_shell(command, cwd=str(start))
        invocation = ShellInvocation.build(
            command, cwd=str(start), env=env, interpreter=interpreter
        )
        raw = self._execute(invocation.argv, timeout=timeout)
        return invocation.parse(raw, confine=self.resolve_path, root=self._root)

    def _execute(
        self,
        argv: list[str],
        *,
        timeout: float,
        input: str | None = None,  # noqa: A002
    ) -> SandboxResult:
        """Run already-validated ``argv`` under the platform confinement."""
        return run_process(
            self._wrap(argv),
            cwd=self._root,
            env=self._env(),
            timeout=timeout,
            input=input,
        )

    # --- confinement helpers -------------------------------------------------

    def _env(self) -> dict[str, str]:
        """Build a scrubbed minimal environment for the sandboxed process.

        ``TMPDIR`` is the sandbox's private scratch dir (writes to it are allowed
        by the macOS profile), so a subprocess that follows ``$TMPDIR``
        (compilers, ``tempfile``, ``mktemp``) can write scratch files without
        those landing in — or being visible in — the user's workspace root.
        """
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
            "HOME": str(self._root),
            "TMPDIR": str(self._tmpdir),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }

    def _wrap(self, argv: list[str]) -> list[str]:
        """Wrap ``argv`` in ``sandbox-exec`` on macOS when it is available."""
        if sys.platform != "darwin":
            return list(argv)
        sandbox_exec = shutil.which("sandbox-exec")
        if sandbox_exec is None:
            return list(argv)
        return [sandbox_exec, "-p", self._profile(), *argv]

    def _profile(self) -> str:
        """Build the macOS ``sandbox-exec`` profile string.

        The profile allows everything by default, then narrows file writes to
        the root, the process temp dir, and ``/dev`` (for stdio), and denies
        network unless :attr:`allow_network` is set.
        """
        lines = [
            "(version 1)",
            "(allow default)",
            "(deny file-write*)",
            f'(allow file-write* (subpath "{self._sbpl(self._root)}"))',
            f'(allow file-write* (subpath "{self._sbpl(self._tmpdir)}"))',
            '(allow file-write* (subpath "/dev"))',
        ]
        if not self.allow_network:
            lines.append("(deny network*)")
        return "\n".join(lines)

    @staticmethod
    def _sbpl(path: Path) -> str:
        r"""Escape ``path`` for use inside an SBPL double-quoted string literal.

        SBPL (TinyScheme) escapes with backslash, so backslashes must be doubled
        before quotes are escaped — otherwise a path byte like ``\"`` would
        terminate the literal early and let trailing bytes parse as directives.
        """
        return str(path).replace("\\", "\\\\").replace('"', '\\"')
