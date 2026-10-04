from __future__ import annotations

import math
import os
import shlex
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from nexus_ai_harness.plugins.tools._paths import confine, display
from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.sandbox import (
    Sandbox,
    SandboxResult,
    SandboxViolation,
    ShellResult,
)
from nexus_ai_harness.protocols.tool import Tool
from nexus_ai_harness.services.process import TIMEOUT_NOTE, run_process
from nexus_ai_harness.services.shell import ShellInvocation

_DEFAULT_TIMEOUT = 30.0
"""Wall-clock limit applied when the model omits a timeout."""

_MAX_TIMEOUT = 600.0
"""Default hard cap so the model can't request an unbounded run."""

_MAX_OUTPUT = 30_000
"""Default cap on returned output so a chatty command can't blow up context."""

_SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
"""The only environment variables handed to an unsandboxed subprocess."""

_MAX_ENV_VARS = 256
"""Cap on distinct persisted variables, so a session's argv stays bounded."""


@dataclass
class ShellState:
    """The shell session carried between ``shell`` calls within one session.

    Attributes:
        cwd: The confined absolute directory the next command starts in, or
            ``None`` for the root.
        env: Variables the model set (a value) or unset (``None``) so far.
        seq: Ticket counter; each call claims the next value when it starts.
        cwd_seq: The ticket of the call that last set ``cwd``, so a call
            issued earlier can't clobber one issued later that finishes first.
    """

    cwd: str | None = None
    env: dict[str, str | None] = field(default_factory=dict)
    seq: int = 0
    cwd_seq: int = 0
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


def _truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    """Cap ``text`` at ``limit`` chars, keeping its head and tail.

    The tail usually holds the error or summary a command ends with, so the
    middle is dropped, cutting at line breaks where possible.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    head = text[:half]
    cut = head.rfind("\n")
    if cut > half // 2:
        head = head[:cut]
    tail = text[-half:]
    cut = tail.find("\n")
    if 0 <= cut < half // 2:
        tail = tail[cut + 1 :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... ({omitted} chars omitted) ...\n{tail}"


def _scrubbed_env() -> dict[str, str]:
    """Return a minimal environment for an unsandboxed subprocess."""
    return {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}


def _apply_env_changes(
    env: dict[str, str | None], changes: dict[str, str | None]
) -> None:
    """Merge ``changes`` into ``env``, then drop the oldest entries over the cap.

    Every call re-embeds all of ``env`` in its argv, so an unbounded variable
    count would eventually make the command line too long to exec.
    """
    env.update(changes)
    while len(env) > _MAX_ENV_VARS:
        env.pop(next(iter(env)))


def _coerce_timeout(
    value: object,
    default: float = _DEFAULT_TIMEOUT,
    cap: float = _MAX_TIMEOUT,
) -> float:
    """Coerce a model-supplied timeout to a positive, capped number of seconds."""
    # bool is a subclass of int, so exclude it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return min(default, cap)
    number = float(value)
    # Reject NaN/inf (JSON allows the NaN literal): NaN slips past ``<= 0`` and
    # would disable the "mandatory" timeout, and inf never fires.
    if not math.isfinite(number) or number <= 0:
        return min(default, cap)
    return min(number, cap)


class Shell(Tool):
    """Run a shell command string, confined to the sandbox root when one is set.

    Commands run through ``bash`` (``sh`` if absent), so pipes, redirects,
    ``&&``/``||``, globs, and quoting behave as in a terminal. The working
    directory and exported variables persist across calls within a session: a
    ``cd src`` in one call is where the next call starts, confined to the
    workspace root. Standard error is merged into standard output in the order
    it was written.

    With a :class:`Sandbox`, the command runs through ``Sandbox.run_shell``,
    which applies the sandbox's command policy to the programs in the string;
    a sandbox that cannot run shell strings falls back to running the string as
    a single argument vector. Without one, it runs in the process cwd with a
    scrubbed environment. Either way it stays gated behind the permission
    layer like any other tool.

    Attributes:
        default_timeout: Seconds allowed when the model omits ``timeout``.
        max_timeout: The most seconds the model may request.
        max_output: Chars of output returned before the middle is dropped.
        persist_env: Whether exported variables carry into the next call.
    """

    name = "shell"
    description = (
        "Run a shell command (bash syntax: pipes, redirects, &&, globs) and "
        "return its exit code and combined stdout/stderr. The working directory "
        "and exported variables persist between calls, so `cd subdir` carries "
        "over; the directory stays within the workspace root. Each call has a "
        "timeout (default 30 s; pass `timeout` for longer runs such as test "
        "suites). Commands must not wait for interactive input (stdin is "
        "closed). To leave a process running in the background, redirect its "
        "output (`cmd > log.txt 2>&1 &`); a background job still attached to "
        "the output is stopped when the command finishes. Very long output "
        "keeps its beginning and end."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "The command to run, e.g. 'cd src && pytest -x 2>&1 | tail'."
                ),
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Wall-clock limit in seconds (default 30, capped by the "
                    "tool's configured maximum, 600 unless changed)."
                ),
            },
        },
        "required": ["command"],
    }

    def __init__(
        self,
        *,
        default_timeout: float = _DEFAULT_TIMEOUT,
        max_timeout: float = _MAX_TIMEOUT,
        max_output: int = _MAX_OUTPUT,
        persist_env: bool = True,
    ) -> None:
        """Configure limits and state carry-over for the shell tool.

        Args:
            default_timeout: Seconds allowed when the model omits ``timeout``.
            max_timeout: Upper bound on any requested timeout, in seconds.
            max_output: Chars of output returned before truncating the middle.
            persist_env: If ``True`` (default), variables a command exports are
                set again for the next call in the same session.
        """
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.max_output = max_output
        self.persist_env = persist_env

    def run(self, arguments: dict, ctx: Context) -> str:
        command = str(arguments.get("command", ""))
        if not command.strip():
            return "error: empty command"
        timeout = _coerce_timeout(
            arguments.get("timeout"), self.default_timeout, self.max_timeout
        )
        state = ctx.state(ShellState)
        with state.lock:
            started = state.cwd
            ticket = state.seq
            state.seq += 1
        env = dict(state.env) if self.persist_env else None
        sandbox: Sandbox | None = ctx.get(Sandbox)
        result: SandboxResult
        try:
            if sandbox is None:
                base = Path(os.getcwd()).resolve()
                result = self._run_local(command, timeout, started, env, base)
            else:
                base = sandbox.resolve_path(".")
                result = self._run_sandboxed(sandbox, command, timeout, started, env)
        except SandboxViolation as exc:
            return f"error: {exc}"
        except (OSError, ValueError) as exc:  # ValueError e.g. embedded NUL
            return f"error: {exc}"

        moved = ""
        if isinstance(result, ShellResult):
            with state.lock:
                # A call only overwrites cwd if its ticket is at least as new as
                # the last one applied, so a call issued earlier can't clobber a
                # later one just because its own command happened to finish last.
                if (
                    result.cwd is not None
                    and result.cwd != (started or str(base))
                    and ticket >= state.cwd_seq
                ):
                    moved = f"\n(cwd is now {display(Path(result.cwd), base)})"
                    state.cwd = None if result.cwd == str(base) else result.cwd
                    state.cwd_seq = ticket
                if self.persist_env:
                    _apply_env_changes(state.env, result.env)
        return self._format(result, timeout) + moved

    @staticmethod
    def _run_sandboxed(
        sandbox: Sandbox,
        command: str,
        timeout: float,
        cwd: str | None,
        env: dict[str, str | None] | None,
    ) -> SandboxResult:
        """Run via the sandbox's shell support, or as argv if it has none."""
        try:
            return sandbox.run_shell(command, timeout=timeout, cwd=cwd, env=env)
        except NotImplementedError:
            argv = shlex.split(command)
            if not argv:
                raise ValueError("empty command") from None
            try:
                result = sandbox.run_command(argv, timeout=timeout)
            except FileNotFoundError:
                # Match the bash-wrapper path's shape for a missing program.
                return SandboxResult(returncode=127, stdout="", stderr="")
            if result.timed_out:
                # The header already reports the timeout; drop the duplicate note
                # so every path shows the same shape.
                result.stderr = "\n".join(
                    line
                    for line in result.stderr.splitlines()
                    if not line.startswith(TIMEOUT_NOTE)
                )
            return result

    @staticmethod
    def _run_local(
        command: str,
        timeout: float,
        cwd: str | None,
        env: dict[str, str | None] | None,
        base: Path,
    ) -> ShellResult:
        """Run through a local shell in the (confined) session cwd."""
        start = base
        if cwd:
            try:
                start = confine(cwd, base)
            except SandboxViolation:  # the process cwd moved; restart at the root
                start = base
        invocation = ShellInvocation.build(command, cwd=str(start), env=env)
        raw = run_process(
            invocation.argv, cwd=base, env=_scrubbed_env(), timeout=timeout
        )
        return invocation.parse(raw, confine=lambda p: confine(p, base), root=base)

    def _format(self, result: SandboxResult, timeout: float) -> str:
        """Render the exit status and output in one shape for every path."""
        header = f"exit={result.returncode}"
        if result.timed_out:
            header += f" (timed out after {timeout:g}s)"
        output = result.stdout
        extra = result.stderr.strip()
        if extra:
            label = "" if isinstance(result, ShellResult) else "[stderr]\n"
            joiner = "\n" if output and not output.endswith("\n") else ""
            output = f"{output}{joiner}{label}{extra}"
        return f"{header}\n{_truncate(output, self.max_output)}"
