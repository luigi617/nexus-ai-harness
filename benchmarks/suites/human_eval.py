from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.config import BenchmarkConfig
from benchmarks.core.registry import register
from benchmarks.core.task import Score, Task
from nexus_ai_harness.harness import NexusAIHarness
from nexus_ai_harness.harness.result import RunResult
from nexus_ai_harness.harness.session import Session
from nexus_ai_harness.plugins.loops import ChatLoop
from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
from nexus_ai_harness.protocols.lifecycle import Lifecycle
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.sandbox import Sandbox, SandboxViolation

_INSTRUCTION = (
    "Complete the following Python function. Return the full function "
    "definition including its signature, as Python code and nothing else."
)
# The model is told to return full code, so grab the first fenced block if the
# reply is wrapped in Markdown/prose; otherwise treat the whole reply as code.
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)

logger = logging.getLogger(__name__)

#: Isolation modes accepted by ``HumanEvalConfig.isolation``.
ISOLATION_MODES = ("auto", "docker", "sandbox", "none")

# Docker's cgroup --memory accepts as little as 6 MiB, but sandbox mode's
# RLIMIT_AS caps virtual address space, and CPython needs well above that just
# to start: 6 MiB fails before the program runs; 64 MiB leaves real headroom.
_MIN_MEMORY_MB = 64


class IsolationError(RuntimeError):
    """The isolation backend itself failed, so the completion was never judged.

    Raised instead of returning a failing result so the runner records the
    attempt as an error rather than blaming the model.
    """


@dataclass
class HumanEvalConfig(BenchmarkConfig):
    """HumanEval knobs.

    Attributes:
        timeout_s: Per-completion execution timeout, in seconds. It also sets the
            child's CPU-time limit (rounded up, plus one second).
        isolation: How model-generated code is executed. ``"auto"`` (default)
            picks the safest available backend: ``"docker"`` when a Docker daemon
            is reachable and ``docker_image`` is already present locally, else
            ``"sandbox"``. ``"docker"`` requests a container (pulling the image
            if needed) and falls back to ``"sandbox"`` when Docker is unusable.
            ``"sandbox"`` runs under :class:`WorkspaceSandbox` rooted at a
            throwaway directory with network denied, plus POSIX resource limits
            and a scrubbed environment. ``"none"`` is the unconfined legacy
            subprocess, for debugging only.
        memory_limit_mb: Address-space (sandbox) or cgroup memory (docker) cap
            in MiB, at least 64 (sandbox mode's ``RLIMIT_AS`` needs well above
            Docker's own 6 MiB floor just to start the interpreter); ``None``
            leaves memory unlimited. Silently skipped where the OS cannot
            enforce it (``RLIMIT_AS`` on macOS).
        max_file_size_mb: Largest file the child may write, in MiB, at least 1;
            ``None`` leaves it unlimited.
        max_processes: Extra processes (non-negative) the child may spawn;
            ``None`` leaves it unlimited. In docker mode it bounds the
            container's pid count. In sandbox mode it becomes ``RLIMIT_NPROC``,
            which the kernel counts per user (and, on Linux, per thread), so the
            default ``0`` forbids forking and threads, and a positive value is
            added to the user's task count at launch; that count is shared, so
            the allowance is approximate while other checks run concurrently.
        docker_image: Image for docker mode; ``None`` uses the official
            ``python:<major>.<minor>-slim`` image matching the host interpreter.
    """

    timeout_s: float = 15.0
    isolation: str = "auto"
    memory_limit_mb: int | None = 1024
    max_file_size_mb: int | None = 16
    max_processes: int | None = 0
    docker_image: str | None = None

    def __post_init__(self) -> None:
        if self.isolation not in ISOLATION_MODES:
            raise ValueError(
                f"isolation: expected one of {list(ISOLATION_MODES)}, "
                f"got {self.isolation!r}"
            )
        # 0 and negatives mean different things per backend (RLIMIT_NPROC -1 is
        # unlimited, --pids-limit 1 is not), so reject them outright.
        if not self.timeout_s > 0:
            raise ValueError(f"timeout_s: must be > 0, got {self.timeout_s!r}")
        _check_minimum("memory_limit_mb", self.memory_limit_mb, _MIN_MEMORY_MB)
        _check_minimum("max_file_size_mb", self.max_file_size_mb, 1)
        _check_minimum("max_processes", self.max_processes, 0)


def _check_minimum(name: str, value: int | None, minimum: int) -> None:
    if value is not None and value < minimum:
        raise ValueError(f"{name}: must be >= {minimum} or none, got {value!r}")


@register
class HumanEval(Benchmark):
    """HumanEval: complete a Python function, graded by executing its unit tests."""

    name = "humaneval"
    description = "HumanEval: function completion graded by executing unit tests"
    config_type = HumanEvalConfig

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """Store the run config; the isolation backend is resolved on first use."""
        super().__init__(config)
        self._isolation: str | None = None
        # build_harness runs on several worker threads at once; one probe/pull.
        self._isolation_lock = threading.Lock()

    @property
    def isolation(self) -> str:
        """The concrete isolation mode in use (``docker``, ``sandbox``, or ``none``).

        Resolved once and cached, so Docker is probed (and pulled) at most once
        per run. The runner first touches it from :meth:`build_harness`, off the
        event loop, since probing or pulling may block for minutes.
        """
        with self._isolation_lock:
            if self._isolation is None:
                self._isolation = resolve_isolation(
                    self.config.isolation, self._docker_image()
                )
                _log_resolved(self.config.isolation, self._isolation)
            return self._isolation

    def load_tasks(self, *, limit: int | None = None) -> list[Task]:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return [
            Task(
                task_id=row["task_id"],
                prompt=f"{_INSTRUCTION}\n\n{row['prompt']}",
                metadata={
                    "prompt": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                },
            )
            for row in dataset
        ]

    def build_harness(self, task: Task, model: Model, session: Session) -> Episode:
        # Called off the event loop, so a Docker probe or pull can't stall it.
        _ = self.isolation
        harness = NexusAIHarness().use(ChatLoop()).use(model)
        return Episode(harness=harness, initial_input=task.prompt)  # single-turn

    def grade(self, task: Task, result: RunResult) -> Score | Awaitable[Score]:
        """Execute the completion against the task's tests.

        Inside a running event loop (the runner) the check runs on a worker
        thread and an awaitable is returned, so a slow container start or a
        program running out its timeout doesn't stall other attempts. Called
        from sync code it returns the :class:`Score` directly.

        Raises:
            IsolationError: If the isolation backend failed before the program
                could be judged (for example the Docker daemon errored).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._grade_sync(task, result)
        return asyncio.to_thread(self._grade_sync, task, result)

    def _grade_sync(self, task: Task, result: RunResult) -> Score:
        completion = _extract_code(result.output)
        program = _assemble_program(
            task.metadata["prompt"],
            completion,
            task.metadata["test"],
            task.metadata["entry_point"],
        )
        isolation = self.isolation
        passed = _run_check(
            program,
            self.config.timeout_s,
            isolation=isolation,
            limits=ResourceLimits(
                memory_mb=self.config.memory_limit_mb,
                file_size_mb=self.config.max_file_size_mb,
                processes=self.config.max_processes,
            ),
            docker_image=self._docker_image(),
        )
        return Score(
            passed=passed,
            detail={
                "task_id": task.task_id,
                "entry_point": task.metadata["entry_point"],
                "isolation": isolation,
                "kernel_confined": kernel_confined(isolation),
            },
        )

    def _docker_image(self) -> str:
        return self.config.docker_image or _default_docker_image()


# --- code extraction and assembly (offline-testable) ----------------------


def _extract_code(text: str) -> str:
    """Pull runnable Python out of a model reply, stripping Markdown/prose.

    Returns the first fenced code block's contents when the reply is fenced,
    else the whole reply trimmed of surrounding whitespace.
    """
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip("\n")
    return text.strip()


def _defines_entry_point(code: str, entry_point: str) -> bool:
    """Whether ``code`` already contains a ``def entry_point(...)``."""
    pattern = rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\("
    return re.search(pattern, code) is not None


def _assemble_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """Build the runnable program that defines the solution and invokes ``check``.

    When the completion already re-emits the signature (a full function/module),
    it is self-contained and used verbatim; otherwise it is treated as the body
    and appended to the prompt's signature, so both model styles run correctly.
    """
    if _defines_entry_point(completion, entry_point):
        body = completion
    else:
        body = prompt + completion
    return f"{body}\n{test}\n\ncheck({entry_point})\n"


# --- isolated execution of model-generated code ---------------------------

# Runs inside the sandboxed child: applies the limits, then execs the real
# program so the limits are inherited but the program never sees this wrapper.
_RLIMIT_BOOTSTRAP = """\
import json, os, sys
try:
    import resource
except ImportError:
    resource = None
if resource is not None:
    for name, value in json.loads(sys.argv[1]):
        which = getattr(resource, name, None)
        if which is None:
            continue
        try:
            _, hard = resource.getrlimit(which)
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            pass
os.execv(sys.executable, [sys.executable, "-c", sys.argv[2]])
"""

# Container start-up is not the program's time: the program itself is bounded
# by timeout(1) inside the container, the host wait adds this much slack.
_DOCKER_STARTUP_GRACE_S = 30.0
# The in-container timeout(1) and python processes count against --pids-limit.
_DOCKER_BASE_PIDS = 2
# Exit codes docker run (125) and timeout(1) (125-127) use for their own
# failures, e.g. a daemon error or an image without timeout(1).
_DOCKER_INFRA_EXIT_CODES = frozenset({125, 126, 127})
_DOCKER_PROBE_TIMEOUT_S = 15.0
_DOCKER_PULL_TIMEOUT_S = 600.0
_MIB = 1024 * 1024


@dataclass(frozen=True)
class ResourceLimits:
    """Resource caps applied to one execution of model-generated code.

    Attributes:
        memory_mb: Memory cap in MiB, or ``None`` for unlimited.
        file_size_mb: Largest writable file in MiB, or ``None`` for unlimited.
        processes: Extra processes the child may spawn, or ``None`` for
            unlimited.
    """

    memory_mb: int | None = 1024
    file_size_mb: int | None = 16
    processes: int | None = 0

    def rlimits(self, timeout: float, *, nproc_base: int = 0) -> list[tuple[str, int]]:
        """The ``resource`` limits to set in the child, as ``(name, value)`` pairs.

        Core dumps are always disabled and CPU time is capped just above the
        wall-clock ``timeout``, so a runaway loop dies even if the parent's
        timer is missed.

        Args:
            timeout: The wall-clock limit the CPU cap is derived from.
            nproc_base: Tasks already counted against the user's
                ``RLIMIT_NPROC``, including the child itself; a positive
                ``processes`` allowance is added on top of it. Ignored when
                ``processes`` is ``0``, since forking is then simply forbidden.
        """
        limits: list[tuple[str, int]] = [
            ("RLIMIT_CPU", _cpu_seconds(timeout)),
            ("RLIMIT_CORE", 0),
        ]
        if self.memory_mb is not None:
            limits.append(("RLIMIT_AS", self.memory_mb * _MIB))
        if self.file_size_mb is not None:
            limits.append(("RLIMIT_FSIZE", self.file_size_mb * _MIB))
        if self.processes is not None:
            nproc = nproc_base + self.processes if self.processes > 0 else 0
            limits.append(("RLIMIT_NPROC", nproc))
        return limits


def resolve_isolation(requested: str, docker_image: str) -> str:
    """Resolve a configured isolation mode to the concrete backend to use.

    Args:
        requested: One of :data:`ISOLATION_MODES`.
        docker_image: The image docker mode would run.

    Returns:
        ``"docker"``, ``"sandbox"``, or ``"none"``. ``"auto"`` becomes
        ``"docker"`` only when the daemon is reachable and the image is already
        local (so a run never blocks on an unrequested pull); ``"docker"``
        pulls a missing image and degrades to ``"sandbox"``, with a warning,
        when Docker is unusable.
    """
    if requested in ("sandbox", "none"):
        return requested
    explicit = requested == "docker"
    if not _docker_available():
        if explicit:
            logger.warning("humaneval: Docker unavailable; using sandbox isolation")
        return "sandbox"
    if _docker_image_present(docker_image):
        return "docker"
    if explicit:
        if _docker_pull(docker_image):
            return "docker"
        logger.warning(
            "humaneval: could not pull %s; using sandbox isolation", docker_image
        )
    return "sandbox"


def _run_check(
    program: str,
    timeout: float,
    *,
    isolation: str = "sandbox",
    limits: ResourceLimits | None = None,
    docker_image: str | None = None,
) -> bool:
    """Execute ``program`` in isolation; True iff it exits 0 within ``timeout``.

    Runs in a throwaway working directory so file-touching completions cannot
    pollute the repo, and swallows timeouts/spawn errors as a failing (not
    raising) result. A failure of the Docker backend itself raises
    :class:`IsolationError` instead, so it isn't scored as the model's.

    Args:
        program: The assembled Python program (solution plus ``check`` call).
        timeout: Wall-clock limit in seconds.
        isolation: A concrete mode from :func:`resolve_isolation`.
        limits: Resource caps; defaults to :class:`ResourceLimits` defaults.
        docker_image: Image for docker mode; defaults to the host-matched one.
    """
    limits = limits if limits is not None else ResourceLimits()
    if isolation == "docker":
        image = docker_image or _default_docker_image()
        return _run_in_docker(program, timeout, limits, image)
    if isolation not in ("sandbox", "none"):
        raise ValueError(f"unknown isolation mode: {isolation!r}")
    workdir = tempfile.mkdtemp(prefix="humaneval_")
    try:
        if isolation == "sandbox":
            return _run_in_sandbox(program, timeout, limits, workdir)
        return _run_unconfined(program, timeout, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_in_sandbox(
    program: str, timeout: float, limits: ResourceLimits, workdir: str
) -> bool:
    """Run under a network-denied :class:`WorkspaceSandbox` with rlimits applied."""
    sandbox = WorkspaceSandbox(
        workdir, allow_network=False, allowed_commands=[sys.executable]
    )
    try:
        argv = _sandbox_argv(program, timeout, limits)
        return _sandbox_passes(sandbox, argv, timeout)
    finally:
        _stop_sync(sandbox)


def _sandbox_passes(sandbox: Sandbox, argv: list[str], timeout: float) -> bool:
    try:
        return sandbox.run_command(argv, timeout=timeout).returncode == 0
    except (SandboxViolation, OSError):
        return False


def _sandbox_argv(program: str, timeout: float, limits: ResourceLimits) -> list[str]:
    """The argv that runs ``program``, behind the rlimit bootstrap on POSIX."""
    if os.name != "posix":  # no ``resource`` module, and execv there is a respawn
        return [sys.executable, "-c", program]
    nproc_base = 0
    if limits.processes:
        # RLIMIT_NPROC is a per-user total, so "N extra" is the current count
        # plus the child plus N; an unknown count falls back to the strict cap.
        current = _user_task_count()
        nproc_base = current + 1 if current is not None else 0
    encoded = json.dumps(limits.rlimits(timeout, nproc_base=nproc_base))
    return [sys.executable, "-c", _RLIMIT_BOOTSTRAP, encoded, program]


def _user_task_count() -> int | None:
    """Tasks the kernel counts against this real uid's ``RLIMIT_NPROC``.

    Linux counts threads, read from ``/proc``; elsewhere ``ps`` counts
    processes. Returns ``None`` if the count can't be taken.
    """
    uid = os.getuid()
    proc = Path("/proc")
    if sys.platform.startswith("linux") and proc.is_dir():
        total = 0
        for status in proc.glob("[0-9]*/status"):
            try:
                fields = dict(
                    line.split(":", 1)
                    for line in status.read_text().splitlines()
                    if ":" in line
                )
                if int(fields["Uid"].split()[0]) == uid:
                    total += int(fields["Threads"])
            except (OSError, KeyError, ValueError):
                continue  # the process exited mid-scan
        return total
    try:
        listing = subprocess.run(
            ["ps", "-A", "-o", "ruid="],
            capture_output=True,
            text=True,
            timeout=_DOCKER_PROBE_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if listing.returncode != 0:
        return None
    return listing.stdout.split().count(str(uid))


def kernel_confined(isolation: str) -> bool:
    """Whether ``isolation`` confines file writes and network in the kernel.

    Docker always does. ``sandbox`` does only where :class:`WorkspaceSandbox`
    has a kernel backend (today macOS ``sandbox-exec``); elsewhere it applies
    just resource limits and a scrubbed environment. ``none`` never does.

    Args:
        isolation: A concrete mode from :func:`resolve_isolation`.
    """
    if isolation == "docker":
        return True
    if isolation == "sandbox":
        return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None
    return False


def _log_resolved(requested: str, isolation: str) -> None:
    logger.info("humaneval: isolation=%s (requested %s)", isolation, requested)
    if isolation == "sandbox" and not kernel_confined(isolation):
        logger.warning(
            "humaneval: sandbox isolation here limits resources and scrubs the "
            "environment but does not confine file writes or network; use "
            "isolation=docker for untrusted code"
        )


def _run_unconfined(program: str, timeout: float, workdir: str) -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            cwd=workdir,
            timeout=timeout,
            check=False,
        )
        return completed.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _run_in_docker(
    program: str, timeout: float, limits: ResourceLimits, image: str
) -> bool:
    """Run in a throwaway, network-less, read-only container.

    Raises:
        IsolationError: If docker or the in-container ``timeout`` failed
            (exit 125-127) rather than the program.
    """
    name = f"humaneval-{uuid.uuid4().hex[:12]}"
    command = _docker_command(name, image, program, timeout, limits)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout + _DOCKER_STARTUP_GRACE_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # Killing the docker CLI leaves the container running; remove it by name.
        _docker(["rm", "-f", name], timeout=_DOCKER_PROBE_TIMEOUT_S)
        return False
    except OSError:
        return False
    if completed.returncode in _DOCKER_INFRA_EXIT_CODES:
        stderr = completed.stderr.decode(errors="replace").strip()
        raise IsolationError(
            f"docker run exited {completed.returncode}: {stderr[-500:]}"
        )
    return completed.returncode == 0


def _docker_command(
    name: str, image: str, program: str, timeout: float, limits: ResourceLimits
) -> list[str]:
    """Build the ``docker run`` argv for one isolated check."""
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--workdir",
        "/tmp",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpus",
        "1",
        "--env",
        "HOME=/tmp",
        "--ulimit",
        "core=0",
        "--ulimit",
        f"cpu={_cpu_seconds(timeout)}",
    ]
    if limits.memory_mb is not None:
        memory = f"{limits.memory_mb}m"
        command += ["--memory", memory, "--memory-swap", memory]  # no swap escape
    if limits.file_size_mb is not None:
        command += ["--ulimit", f"fsize={limits.file_size_mb * _MIB}"]
    if limits.processes is not None:
        command += ["--pids-limit", str(_DOCKER_BASE_PIDS + limits.processes)]
    return [
        *command,
        image,
        "timeout",
        "-s",
        "KILL",
        str(timeout),
        "python",
        "-c",
        program,
    ]


def _default_docker_image() -> str:
    """The official slim Python image matching the host interpreter's version."""
    return f"python:{sys.version_info.major}.{sys.version_info.minor}-slim"


def _cpu_seconds(timeout: float) -> int:
    return math.ceil(timeout) + 1


def _docker(args: list[str], *, timeout: float) -> bool:
    """Run a docker CLI subcommand; True iff it succeeded."""
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", *args], capture_output=True, timeout=timeout, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def _docker_available() -> bool:
    return _docker(["info"], timeout=_DOCKER_PROBE_TIMEOUT_S)


def _docker_image_present(image: str) -> bool:
    return _docker(["image", "inspect", image], timeout=_DOCKER_PROBE_TIMEOUT_S)


def _docker_pull(image: str) -> bool:
    return _docker(["pull", image], timeout=_DOCKER_PULL_TIMEOUT_S)


def _stop_sync(lifecycle: Lifecycle) -> None:
    """Run ``lifecycle.stop()`` to completion from sync code, best-effort.

    Grading is synchronous but may be called on the runner's event-loop thread,
    where ``asyncio.run`` refuses to nest, so it then runs on a helper thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        in_loop = False
    else:
        in_loop = True
    try:
        if in_loop:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(asyncio.run, lifecycle.stop()).result()
        else:
            asyncio.run(lifecycle.stop())
    except Exception:  # cleanup must not turn a graded result into an error
        logger.debug("humaneval: sandbox cleanup failed", exc_info=True)
