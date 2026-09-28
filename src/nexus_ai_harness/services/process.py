from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path

from nexus_ai_harness.protocols.sandbox import SandboxResult

TIMEOUT_RETURNCODE = 124
"""Exit code reported for a timed-out command, matching ``timeout(1)``."""

_DRAIN_TIMEOUT = 5.0
"""How long to wait for output pipes to close after killing a timed-out group."""


def run_process(
    argv: list[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    timeout: float,
    input: str | None = None,  # noqa: A002
    merge_stderr: bool = False,
) -> SandboxResult:
    """Run ``argv`` to completion (or timeout) and capture its output as text.

    The process runs in its own session so a timeout kills the whole process
    group: a shell's children (``sleep 100 | cat``) would otherwise hold the
    output pipes open and hang the capture long after the shell itself died.
    Standard input is closed when ``input`` is ``None`` so a command that reads
    stdin sees EOF instead of stealing the host terminal. Output is decoded as
    UTF-8 with replacement characters, so binary output never raises.

    Args:
        argv: The argument vector; executed directly, never through a shell.
        cwd: Working directory for the process.
        env: The complete environment for the process.
        timeout: Wall-clock limit in seconds.
        input: Optional text written to the process's standard input.
        merge_stderr: If ``True``, standard error is interleaved into
            ``stdout`` in the order it was written and ``stderr`` is empty.

    Returns:
        The exit status and captured output. On timeout ``timed_out`` is set,
        ``returncode`` is :data:`TIMEOUT_RETURNCODE`, and the output holds
        whatever was captured before the kill.

    Raises:
        OSError: If the program cannot be started (e.g. it does not exist).
        ValueError: If ``argv`` is malformed (e.g. contains a NUL byte).
    """
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=_DRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:  # a daemonized grandchild kept a pipe open
            stdout, stderr = "", ""
        return SandboxResult(
            returncode=TIMEOUT_RETURNCODE,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=True,
        )
    except BaseException:
        _kill_group(proc)  # never leak a running process group on interrupt
        proc.wait()
        raise
    return SandboxResult(
        returncode=proc.returncode, stdout=stdout or "", stderr=stderr or ""
    )


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Kill ``proc`` and every process in its group, tolerating a dead group."""
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # Windows has no process groups; kill the leader only
        proc.kill()
        return
    try:
        killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
