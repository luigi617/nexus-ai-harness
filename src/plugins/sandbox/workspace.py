from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from protocols.sandbox import Sandbox, SandboxResult, SandboxViolation

# Exit code used when a command is killed for exceeding its timeout, matching
# the conventional shell code for a process terminated by ``timeout(1)``.
_TIMEOUT_RETURNCODE = 124


class WorkspaceSandbox(Sandbox):
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
    * Subprocesses are confined via :meth:`run_command`; the ``resolve_path``
      check applies to the filesystem tools, not to arbitrary shell argv.

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
