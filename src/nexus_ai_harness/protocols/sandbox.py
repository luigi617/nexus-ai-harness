from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from nexus_ai_harness.protocols.plugin import Plugin


class SandboxViolation(Exception):
    """Raised when an operation escapes the sandbox's confinement.

    A path that resolves outside the sandbox root, or a command that fails the
    sandbox's command policy, raises this rather than being executed. Tools that
    call a sandbox should catch it and return an ``"error: ..."`` string so the
    model can react, per the tool error convention.
    """


@dataclass
class SandboxResult:
    """The outcome of a command run inside the sandbox.

    Attributes:
        returncode: The process exit status; ``0`` conventionally means success.
        stdout: Captured standard output.
        stderr: Captured standard error.
        timed_out: Whether the command was killed for exceeding its timeout.
    """

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class ShellResult(SandboxResult):
    """The outcome of a shell command string run inside the sandbox.

    Standard error is interleaved into ``stdout`` in the order it was written;
    ``stderr`` only carries notes from the sandbox itself (such as a timeout).

    Attributes:
        cwd: The confined working directory the command finished in, or
            ``None`` when it could not be observed (e.g. the command timed out).
        env: Environment changes the command made, to carry into the next call:
            a value to set, or ``None`` for a variable it unset.
    """

    cwd: str | None = None
    env: dict[str, str | None] = field(default_factory=dict)


class Sandbox(Plugin):
    """An OS confinement boundary for filesystem and subprocess access.

    A sandbox owns a single root directory and a command policy. Tools that
    touch the real filesystem or spawn processes route through it so path and
    command access stay confined: paths are resolved and rejected if they escape
    the root (symlinks included), commands are checked against the policy before
    they run, and every run carries a mandatory timeout. Network access is off by
    default. The sandbox enforces confinement only; permission prompting is a
    separate concern handled by the permission layer.
    """

    @abstractmethod
    def resolve_path(self, path: str) -> Path:
        """Resolve ``path`` to a real path confined to the sandbox root.

        Resolves symlinks and relative segments, then rejects anything that
        lands outside the root — including absolute paths and symlink escapes.

        Args:
            path: A model- or tool-supplied path, absolute or relative to root.

        Returns:
            The resolved, confined absolute path.

        Raises:
            SandboxViolation: If the resolved path escapes the sandbox root.
        """

    @abstractmethod
    def check_command(self, argv: list[str]) -> None:
        """Validate a command against the sandbox's command policy.

        Args:
            argv: The argument vector (never a shell string), ``argv[0]`` being
                the program to run.

        Raises:
            SandboxViolation: If the command is not permitted by the policy.
        """

    @abstractmethod
    def run_command(
        self,
        argv: list[str],
        *,
        timeout: float,
        input: str | None = None,  # noqa: A002
    ) -> SandboxResult:
        """Run a command inside the sandbox and capture its output.

        Passes ``argv`` directly to the OS — never through a shell — with a
        scrubbed, network-off-by-default environment.

        Args:
            argv: The argument vector to execute; ``argv[0]`` is the program.
            timeout: Mandatory wall-clock limit in seconds; the process is killed
                if it is exceeded.
            input: Optional text written to the process's standard input.

        Returns:
            The command's exit status and captured output.

        Raises:
            SandboxViolation: If :meth:`check_command` rejects ``argv``.
        """

    def run_shell(
        self,
        command: str,
        *,
        timeout: float,
        cwd: str | None = None,
        env: Mapping[str, str | None] | None = None,
    ) -> ShellResult:
        """Run a shell command string (pipes, redirects, ``&&``) in the sandbox.

        Optional: the default raises :class:`NotImplementedError`, so a sandbox
        that only supports argv execution keeps working and callers fall back to
        :meth:`run_command`. An implementation must apply its command policy to
        every program the string runs that it can identify, and refuse strings
        it cannot verify when that policy is an allowlist.

        Args:
            command: The shell command string, as the model wrote it.
            timeout: Mandatory wall-clock limit in seconds.
            cwd: Directory to start in, confined to the root (default: root).
            env: Variables to set (or unset, when ``None``) before running.

        Returns:
            The exit status, combined output, and resulting cwd/env changes.

        Raises:
            SandboxViolation: If the policy rejects the command or ``cwd``
                escapes the root.
            NotImplementedError: If this sandbox cannot run shell strings.
        """
        raise NotImplementedError(f"{type(self).__name__} does not run shell strings")
