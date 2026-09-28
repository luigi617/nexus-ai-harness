from __future__ import annotations

import math
import os
import shlex
import subprocess
from typing import ClassVar

from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.sandbox import Sandbox, SandboxResult, SandboxViolation
from nexus_ai_harness.protocols.tool import Tool

_DEFAULT_TIMEOUT = 30.0
"""Wall-clock limit applied when the model omits a timeout."""

_MAX_TIMEOUT = 120.0
"""Hard cap so the model can't request an unbounded run."""

_MAX_OUTPUT = 10_000
"""Cap on captured stdout/stderr so a chatty command can't blow up context."""

_SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR")
"""The only environment variables handed to an unsandboxed subprocess."""


def _truncate(text: str) -> str:
    """Cap ``text`` at ``_MAX_OUTPUT`` chars with a clear marker when clipped."""
    if len(text) <= _MAX_OUTPUT:
        return text
    return text[:_MAX_OUTPUT] + "\n... (truncated)"


def _scrubbed_env() -> dict[str, str]:
    """Return a minimal environment for an unsandboxed subprocess."""
    return {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}


def _coerce_timeout(value: object) -> float:
    """Coerce a model-supplied timeout to a positive, capped number of seconds."""
    # bool is a subclass of int, so exclude it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _DEFAULT_TIMEOUT
    number = float(value)
    # Reject NaN/inf (JSON allows the NaN literal): NaN slips past ``<= 0`` and
    # would disable the "mandatory" timeout, and inf never fires.
    if not math.isfinite(number) or number <= 0:
        return _DEFAULT_TIMEOUT
    return min(number, _MAX_TIMEOUT)


class Shell(Tool):
    """Run a shell command as an argv vector, confined to the sandbox root."""

    name = "shell"
    description = (
        "Run a command and return its exit code and output. The command string "
        "is split into an argument vector and executed directly (never through "
        "a shell), with a mandatory timeout. Output is truncated if very large."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to run, e.g. 'ls -l src'.",
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Wall-clock limit in seconds (default 30, capped at 120)."
                ),
            },
        },
        "required": ["command"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        command = str(arguments.get("command", ""))
        timeout = _coerce_timeout(arguments.get("timeout"))
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return f"error: {exc}"
        if not argv:
            return "error: empty command"

        sandbox: Sandbox | None = ctx.get(Sandbox)
        try:
            result = self._run(argv, timeout, sandbox)
        except SandboxViolation as exc:
            return f"error: {exc}"
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {timeout:g}s"
        except (OSError, ValueError) as exc:  # ValueError e.g. embedded NUL in argv
            return f"error: {exc}"

        out = _truncate(result.stdout)
        err = _truncate(result.stderr)
        return f"exit={result.returncode}\n{out}\n{err}"

    @staticmethod
    def _run(argv: list[str], timeout: float, sandbox: Sandbox | None) -> SandboxResult:
        """Execute ``argv`` via the sandbox, or directly with a scrubbed env."""
        if sandbox is not None:
            return sandbox.run_command(argv, timeout=timeout)
        # shell=False (argv list) so no shell interpretation; errors="replace" so
        # non-UTF-8 output is captured rather than raising mid-capture.
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=os.getcwd(),
            env=_scrubbed_env(),
            check=False,
        )
        return SandboxResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
