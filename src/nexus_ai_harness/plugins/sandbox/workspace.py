from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Iterable
from pathlib import Path

from nexus_ai_harness.plugins.sandbox.isolation import (
    IsolationBackend,
    IsolationSpec,
    IsolationWarning,
    SandboxExecBackend,
    sbpl_escape,
    select_backend,
)
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.lifecycle import Lifecycle
from nexus_ai_harness.protocols.sandbox import Sandbox, SandboxResult, SandboxViolation

# Exit code used when a command is killed for exceeding its timeout, matching
# the conventional shell code for a process terminated by ``timeout(1)``.
_TIMEOUT_RETURNCODE = 124

# Set after the first degraded-isolation warning so a process warns only once.
_warned_unenforced = False


class WorkspaceSandbox(Sandbox, Lifecycle):
    """A :class:`Sandbox` confined to a single workspace root directory.

    Paths are resolved and rejected if they escape the root (symlinks included);
    this path confinement holds on every platform. Commands are checked against
    an optional allowlist and a denylist before running, then run under an
    :class:`~nexus_ai_harness.plugins.sandbox.isolation.IsolationBackend` that
    confines writes to the root (plus a private scratch dir exposed as
    ``$TMPDIR``) and denies network by default:

    * macOS: a ``sandbox-exec`` profile.
    * Linux: ``bwrap`` (bubblewrap) with a read-only ``/``, writable binds of
      the root and scratch dir, a private ``/tmp``, and an unshared network.

    When no backend is usable (Linux without a working ``bwrap``, other
    platforms) the command still runs with ``cwd`` at the root, a scrubbed
    environment, ``shell=False``, and a mandatory timeout, but write confinement
    and network denial are not kernel-enforced. That degradation is observable:
    :attr:`isolation_enforced` is ``False`` and an :class:`IsolationWarning` is
    emitted once per process. Pass ``require_isolation=True`` to raise instead.

    Scope of confinement, by design:

    * Isolation narrows *writes* and network only — *reads* are not
      restricted (denying them would break loading the program's own binary), so
      callers who must prevent exfiltration should also restrict which commands
      may run via ``allowed_commands``/``denied_commands``.
    * Command policy matches ``argv[0]`` (and its basename); it does not see
      through wrapper interpreters such as ``bash -c``/``python -c``.
    * Subprocesses are confined via :meth:`run_command`; the ``resolve_path``
      check applies to the filesystem tools, not to arbitrary shell argv.

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
        isolation: IsolationBackend | str = "auto",
        require_isolation: bool = False,
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
            isolation: The OS isolation backend. ``"auto"`` (default) picks the
                platform's backend and degrades to none, with a warning, when it
                is unavailable. A name (``"sandbox-exec"``, ``"bubblewrap"``,
                ``"none"``) or an ``IsolationBackend`` instance forces a choice;
                ``"none"`` opts out of OS isolation without a warning.
            require_isolation: If ``True``, raise rather than run commands
                without kernel-enforced isolation.

        Raises:
            IsolationUnavailable: If an explicitly chosen backend is unavailable,
                or ``require_isolation`` is set and no backend is usable.
            ValueError: If ``isolation`` is an unknown name, or a non-enforcing
                backend is combined with ``require_isolation``.
        """
        # Selected first so a failed strict check leaves no scratch dir behind.
        self._isolation = select_backend(isolation, require=require_isolation)
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        # A dedicated scratch dir exposed as $TMPDIR: outside the root so it
        # never pollutes the listing, private so the backend's write grant stays narrow.
        self._tmpdir = Path(tempfile.mkdtemp(prefix="nexus-sandbox-")).resolve()
        self.allow_network = allow_network
        self._allowed = (
            frozenset(allowed_commands) if allowed_commands is not None else None
        )
        self._denied = frozenset(denied_commands)
        if isolation == "auto" and not self._isolation.enforced:
            _warn_unenforced_once()

    @property
    def root(self) -> Path:
        """The resolved confinement root directory."""
        return self._root

    @property
    def isolation(self) -> IsolationBackend:
        """The OS isolation backend commands run under."""
        return self._isolation

    @property
    def isolation_level(self) -> str:
        """The active backend's name: ``"sandbox-exec"``, ``"bubblewrap"``, etc."""
        return self._isolation.name

    @property
    def isolation_enforced(self) -> bool:
        """Whether writes and network are kernel-confined for commands."""
        return self._isolation.enforced

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
        ``shell=False``, ``cwd`` set to the root, and a scrubbed minimal
        environment. A timeout is converted into a :class:`SandboxResult` with a
        non-zero return code rather than being raised, so callers get uniform
        output.

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
        command = self._wrap(argv)
        try:
            completed = subprocess.run(
                command,
                cwd=self._root,
                env=self._env(),
                input=input,
                capture_output=True,
                text=True,
                errors="replace",  # non-UTF-8 output must not raise mid-capture
                timeout=timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            note = f"sandbox: command timed out after {timeout}s"
            return SandboxResult(
                returncode=_TIMEOUT_RETURNCODE,
                stdout=stdout,
                stderr=f"{stderr}\n{note}".strip(),
            )
        return SandboxResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    # --- confinement helpers -------------------------------------------------

    def _env(self) -> dict[str, str]:
        """Build a scrubbed minimal environment for the sandboxed process.

        ``TMPDIR`` is the sandbox's private scratch dir (writes to it are allowed
        by every isolation backend), so a subprocess that follows ``$TMPDIR``
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
        """Wrap ``argv`` with the selected isolation backend."""
        return self._isolation.wrap(argv, self._spec())

    def _spec(self) -> IsolationSpec:
        return IsolationSpec(
            root=self._root, tmpdir=self._tmpdir, allow_network=self.allow_network
        )

    def _profile(self) -> str:
        """Build the macOS ``sandbox-exec`` profile string for this sandbox."""
        return SandboxExecBackend().profile(self._spec())

    @staticmethod
    def _sbpl(path: Path) -> str:
        """Escape ``path`` for use inside an SBPL double-quoted string literal."""
        return sbpl_escape(path)


def _warn_unenforced_once() -> None:
    """Warn, once per process, that auto-selection found no isolation backend."""
    global _warned_unenforced
    if _warned_unenforced:
        return
    _warned_unenforced = True
    warnings.warn(
        f"WorkspaceSandbox: no OS isolation backend is available on {sys.platform}; "
        "write confinement and network denial are NOT enforced for sandboxed "
        "commands. Install bubblewrap (bwrap) on Linux, pass "
        "require_isolation=True to fail instead, or isolation='none' to opt out "
        "explicitly.",
        IsolationWarning,
        stacklevel=3,
    )
