from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import IO

from nexus_ai_harness.protocols.sandbox import SandboxResult

TIMEOUT_RETURNCODE = 124
"""Exit code reported for a timed-out command, matching ``timeout(1)``."""

TIMEOUT_NOTE = "sandbox: command timed out"
"""Prefix of the note ``WorkspaceSandbox.run_command`` adds to a timeout's stderr."""

LINGER_NOTE = (
    "note: background processes still holding the output open were stopped; "
    "redirect a background job's output (cmd > log 2>&1 &) to keep it running"
)
"""Note added when the command exited but a child kept its output pipe open."""

_DRAIN_TIMEOUT = 5.0
"""How long to wait for output pipes to close after killing a process group."""

_LINGER_GRACE = 1.0
"""How long output may stay open after the process exits before it is cut off."""

_READ_CHUNK = 65_536


class _Reader:
    """Drain one output pipe on a daemon thread, keeping what has been read."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._chunks: list[bytes] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Chunked reads so partial output survives a pipe that never closes.
        # The reader closes its own stream: closing it from another thread
        # mid-read could let the fd number be reused under this thread.
        try:
            with contextlib.suppress(OSError, ValueError):
                while chunk := os.read(self._stream.fileno(), _READ_CHUNK):
                    self._chunks.append(chunk)
        finally:
            with contextlib.suppress(OSError):
                self._stream.close()

    def join(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for EOF; return whether it arrived."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def text(self) -> str:
        """Return the output so far, decoded as ``text=True`` subprocesses do."""
        data = b"".join(self._chunks).decode("utf-8", errors="replace")
        return data.replace("\r\n", "\n").replace("\r", "\n")


def _feed(stream: IO[bytes], data: str) -> None:
    """Write ``data`` to the process's stdin, then close it."""
    with contextlib.suppress(BrokenPipeError, OSError, ValueError):
        stream.write(data.encode("utf-8"))
    with contextlib.suppress(BrokenPipeError, OSError, ValueError):
        stream.close()


def _join_all(readers: list[_Reader], timeout: float) -> bool:
    """Wait for every reader to hit EOF within ``timeout`` seconds in total."""
    deadline = time.monotonic() + timeout
    return all(r.join(max(deadline - time.monotonic(), 0.0)) for r in readers)


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
    Likewise, once the process exits, output left open by a background child
    (``server &``) is read for a short grace period and then the rest of the
    group is stopped, so the call returns promptly with the real exit code
    and :data:`LINGER_NOTE` rather than waiting out the timeout. Standard
    input is closed when ``input`` is ``None`` so a command that reads stdin
    sees EOF instead of stealing the host terminal. Output is decoded as UTF-8
    with replacement characters, so binary output never raises.

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
        start_new_session=True,
    )
    try:
        return _collect(proc, timeout, input)
    except BaseException:
        _kill_group(proc)  # never leak a running process group on interrupt
        proc.wait()
        raise


def _collect(
    proc: subprocess.Popen[bytes],
    timeout: float,
    input: str | None,  # noqa: A002
) -> SandboxResult:
    """Wait for ``proc`` and gather its output, handling timeouts and lingerers."""
    if proc.stdout is None:  # always piped above; narrows the type
        raise RuntimeError("stdout was not captured")
    out = _Reader(proc.stdout)
    err = _Reader(proc.stderr) if proc.stderr is not None else None
    readers = [out] if err is None else [out, err]
    if input is not None and proc.stdin is not None:
        threading.Thread(target=_feed, args=(proc.stdin, input), daemon=True).start()
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        proc.wait()
        _join_all(readers, _DRAIN_TIMEOUT)  # a daemonized grandchild may never close
        return SandboxResult(
            returncode=TIMEOUT_RETURNCODE,
            stdout=out.text(),
            stderr=err.text() if err else "",
            timed_out=True,
        )
    note = ""
    if not _join_all(readers, _LINGER_GRACE):
        _kill_group(proc)
        _join_all(readers, _DRAIN_TIMEOUT)
        note = LINGER_NOTE
    stderr = err.text() if err else ""
    if note:
        stderr = f"{stderr}\n{note}".strip()
    return SandboxResult(returncode=returncode, stdout=out.text(), stderr=stderr)


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Kill ``proc`` and every process in its group, tolerating a dead group."""
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # Windows has no process groups; kill the leader only
        proc.kill()
        return
    try:
        killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
