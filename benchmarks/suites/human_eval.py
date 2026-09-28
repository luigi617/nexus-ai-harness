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
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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
            in MiB; ``None`` leaves memory unlimited. Silently skipped where the
            OS cannot enforce it (``RLIMIT_AS`` on macOS).
        max_file_size_mb: Largest file the child may write, in MiB; ``None``
            leaves it unlimited.
        max_processes: Extra processes the child may spawn. In sandbox mode this
            is ``RLIMIT_NPROC``, which the kernel counts per user (and, on
            Linux, per thread), so the default ``0`` forbids forking and
            threads; in docker mode it bounds the container's pid count.
            ``None`` leaves it unlimited.
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


@register
class HumanEval(Benchmark):
    """HumanEval: complete a Python function, graded by executing its unit tests."""

    name = "humaneval"
    description = "HumanEval: function completion graded by executing unit tests"
    config_type = HumanEvalConfig

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """Store the run config; the isolation backend is resolved on first grade."""
        super().__init__(config)
        self._isolation: str | None = None

    @property
    def isolation(self) -> str:
        """The concrete isolation mode in use (``docker``, ``sandbox``, or ``none``).

        Resolved once and cached, so Docker is probed at most once per run.
        """
        if self._isolation is None:
            self._isolation = resolve_isolation(
                self.config.isolation, self._docker_image()
            )
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
        harness = NexusAIHarness().use(ChatLoop()).use(model)
        return Episode(harness=harness, initial_input=task.prompt)  # single-turn

    def grade(self, task: Task, result: RunResult) -> Score:
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

    def rlimits(self, timeout: float) -> list[tuple[str, int]]:
        """The ``resource`` limits to set in the child, as ``(name, value)`` pairs.

        Core dumps are always disabled and CPU time is capped just above the
        wall-clock ``timeout``, so a runaway loop dies even if the parent's
        timer is missed.
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
            limits.append(("RLIMIT_NPROC", self.processes))
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
    raising) result.

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
    encoded = json.dumps(limits.rlimits(timeout))
    return [sys.executable, "-c", _RLIMIT_BOOTSTRAP, encoded, program]


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
    """Run in a throwaway, network-less, read-only container."""
    name = f"humaneval-{uuid.uuid4().hex[:12]}"
    command = _docker_command(name, image, program, timeout, limits)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout + _DOCKER_STARTUP_GRACE_S,
            check=False,
        )
        return completed.returncode == 0
    except subprocess.TimeoutExpired:
        # Killing the docker CLI leaves the container running; remove it by name.
        _docker(["rm", "-f", name], timeout=_DOCKER_PROBE_TIMEOUT_S)
        return False
    except OSError:
        return False


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
