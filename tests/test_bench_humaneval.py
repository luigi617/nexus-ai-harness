from __future__ import annotations

import asyncio
import inspect
import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

import benchmarks.suites.human_eval as human_eval
from benchmarks.core.config import load_config_file
from benchmarks.core.registry import get_benchmark
from benchmarks.core.runner import Runner
from benchmarks.core.task import Task
from benchmarks.suites.human_eval import (
    HumanEval,
    HumanEvalConfig,
    ResourceLimits,
    _assemble_program,
    _docker_command,
    _extract_code,
    _run_check,
    resolve_isolation,
)
from nexus_ai_harness.core.response import Response
from nexus_ai_harness.harness.result import RunResult
from nexus_ai_harness.harness.session import Session
from tests.conftest import ScriptedModel

_POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX resource limits only")
_MACOS_SANDBOX = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="kernel-enforced sandbox needs macOS sandbox-exec",
)

_ADD_PROMPT = 'def add(a, b):\n    """Add two numbers."""\n'
_ADD_TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n"


def test_extract_code_from_python_fence():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
    assert _extract_code(text) == "def f():\n    return 1"


def test_extract_code_from_bare_fence():
    text = "```\ndef f():\n    return 1\n```"
    assert _extract_code(text) == "def f():\n    return 1"


def test_extract_code_plain_text_unwrapped():
    text = "def f():\n    return 1"
    assert _extract_code(text) == "def f():\n    return 1"


def test_extract_code_takes_first_fence_when_prose_wrapped():
    text = "Explanation first.\n```python\ndef f():\n    return 2\n```\nmore prose"
    assert _extract_code(text) == "def f():\n    return 2"


def test_run_check_passes_on_correct_program():
    program = "assert 1 + 1 == 2\n"
    assert _run_check(program, timeout=10) is True


def test_run_check_fails_on_failing_assert():
    program = "assert 1 + 1 == 3\n"
    assert _run_check(program, timeout=10) is False


def test_run_check_times_out_without_hanging():
    program = "while True:\n    pass\n"
    assert _run_check(program, timeout=1) is False


def test_assemble_prefers_full_code_when_signature_reemitted():
    # A self-contained completion must not be double-prefixed with the prompt.
    completion = "def add(a, b):\n    return a + b"
    program = _assemble_program(_ADD_PROMPT, completion, _ADD_TEST, "add")
    assert program.count("def add(") == 1


def test_assemble_appends_body_when_only_body_returned():
    completion = "    return a + b\n"
    program = _assemble_program(_ADD_PROMPT, completion, _ADD_TEST, "add")
    assert program.startswith(_ADD_PROMPT)
    assert "def add(" in program


def _grade(output: str) -> bool:
    task = Task(
        task_id="add_0",
        prompt="",
        metadata={"prompt": _ADD_PROMPT, "test": _ADD_TEST, "entry_point": "add"},
    )
    result = RunResult(output=output, session=Session())
    return HumanEval().grade(task, result).passed


def test_grade_passes_on_correct_completion():
    assert _grade("```python\ndef add(a, b):\n    return a + b\n```") is True


def test_grade_fails_on_incorrect_completion():
    assert _grade("```python\ndef add(a, b):\n    return a - b\n```") is False


# --- isolation config and mode resolution ----------------------------------


def _add_task() -> Task:
    return Task(
        task_id="add_0",
        prompt="",
        metadata={"prompt": _ADD_PROMPT, "test": _ADD_TEST, "entry_point": "add"},
    )


def _correct_result() -> RunResult:
    return RunResult(output="def add(a, b):\n    return a + b\n", session=Session())


def test_config_rejects_unknown_isolation_mode():
    with pytest.raises(ValueError, match="isolation"):
        get_benchmark("humaneval", {"isolation": "chroot"})


def test_config_isolation_knobs_settable_via_set_strings():
    bench = get_benchmark(
        "humaneval",
        {"isolation": "sandbox", "max_processes": "none", "memory_limit_mb": "256"},
    )
    assert bench.config.isolation == "sandbox"
    assert bench.config.max_processes is None
    assert bench.config.memory_limit_mb == 256


def test_config_defaults_to_auto_isolation():
    assert HumanEvalConfig().isolation == "auto"


@pytest.mark.parametrize("mode", ["sandbox", "none"])
def test_grade_reports_isolation_mode_in_detail(mode):
    score = HumanEval(HumanEvalConfig(isolation=mode)).grade(
        _add_task(), _correct_result()
    )
    assert score.passed is True
    assert score.detail["isolation"] == mode


def _fake_docker(monkeypatch, *, available, present, pull=False):
    calls: list[str] = []

    def record(name, value):
        def probe(*_args):
            calls.append(name)
            return value

        return probe

    monkeypatch.setattr(human_eval, "_docker_available", record("info", available))
    monkeypatch.setattr(human_eval, "_docker_image_present", record("inspect", present))
    monkeypatch.setattr(human_eval, "_docker_pull", record("pull", pull))
    return calls


@pytest.mark.parametrize("mode", ["sandbox", "none"])
def test_resolve_explicit_host_modes_never_probe_docker(monkeypatch, mode):
    calls = _fake_docker(monkeypatch, available=True, present=True)
    assert resolve_isolation(mode, "img") == mode
    assert calls == []


def test_resolve_auto_prefers_docker_when_image_is_local(monkeypatch):
    _fake_docker(monkeypatch, available=True, present=True)
    assert resolve_isolation("auto", "img") == "docker"


def test_resolve_auto_never_pulls_and_falls_back_to_sandbox(monkeypatch):
    calls = _fake_docker(monkeypatch, available=True, present=False, pull=True)
    assert resolve_isolation("auto", "img") == "sandbox"
    assert "pull" not in calls


def test_resolve_auto_without_docker_uses_sandbox(monkeypatch):
    _fake_docker(monkeypatch, available=False, present=False)
    assert resolve_isolation("auto", "img") == "sandbox"


def test_resolve_explicit_docker_pulls_missing_image(monkeypatch):
    calls = _fake_docker(monkeypatch, available=True, present=False, pull=True)
    assert resolve_isolation("docker", "img") == "docker"
    assert "pull" in calls


def test_resolve_explicit_docker_falls_back_with_warning(monkeypatch, caplog):
    _fake_docker(monkeypatch, available=False, present=False)
    with caplog.at_level("WARNING", logger=human_eval.__name__):
        assert resolve_isolation("docker", "img") == "sandbox"
    assert "Docker unavailable" in caplog.text


def test_resolve_explicit_docker_failed_pull_falls_back(monkeypatch, caplog):
    _fake_docker(monkeypatch, available=True, present=False, pull=False)
    with caplog.at_level("WARNING", logger=human_eval.__name__):
        assert resolve_isolation("docker", "img") == "sandbox"
    assert "could not pull img" in caplog.text


def test_isolation_resolved_once_per_benchmark(monkeypatch):
    calls = _fake_docker(monkeypatch, available=False, present=False)
    bench = HumanEval(HumanEvalConfig(isolation="auto"))
    for _ in range(2):
        assert bench.grade(_add_task(), _correct_result()).detail["isolation"] == (
            "sandbox"
        )
    assert calls == ["info"]


# --- resource limits ---------------------------------------------------------


def test_rlimits_disable_core_and_cap_cpu_above_timeout():
    limits = dict(ResourceLimits().rlimits(timeout=2.5))
    assert limits["RLIMIT_CORE"] == 0
    assert limits["RLIMIT_CPU"] == 4
    assert limits["RLIMIT_AS"] == 1024 * 1024 * 1024
    assert limits["RLIMIT_NPROC"] == 0


def test_rlimits_omit_unlimited_caps():
    limits = ResourceLimits(memory_mb=None, file_size_mb=None, processes=None)
    assert {name for name, _ in limits.rlimits(timeout=1)} == {
        "RLIMIT_CPU",
        "RLIMIT_CORE",
    }


@_POSIX
def test_sandbox_child_has_core_dumps_disabled():
    program = (
        "import resource\nassert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)\n"
    )
    assert _run_check(program, timeout=10) is True


@_POSIX
def test_sandbox_child_file_size_is_capped():
    program = "open('big.bin', 'wb').write(b'x' * (2 * 1024 * 1024))\n"
    capped = ResourceLimits(file_size_mb=1)
    assert _run_check(program, timeout=10, limits=capped) is False
    uncapped = ResourceLimits(file_size_mb=None)
    assert _run_check(program, timeout=10, limits=uncapped) is True


@_POSIX
@pytest.mark.skipif(
    os.name == "posix" and os.geteuid() == 0, reason="root ignores RLIMIT_NPROC"
)
def test_sandbox_child_cannot_fork_by_default():
    program = (
        "import os\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os._exit(0)\n"
        "os.waitpid(pid, 0)\n"
    )
    assert _run_check(program, timeout=10) is False
    unlimited = ResourceLimits(processes=None)
    assert _run_check(program, timeout=10, limits=unlimited) is True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS on Linux")
def test_sandbox_child_memory_is_capped():
    program = "block = bytearray(512 * 1024 * 1024)\n"
    assert (
        _run_check(program, timeout=10, limits=ResourceLimits(memory_mb=256)) is False
    )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS on Linux")
def test_sandbox_child_starts_at_minimum_memory_limit():
    # The floor must leave the interpreter itself enough address space to boot;
    # a too-low floor fails before the program runs at all.
    minimum = ResourceLimits(memory_mb=human_eval._MIN_MEMORY_MB)
    assert _run_check("pass\n", timeout=10, limits=minimum) is True


def test_sandbox_child_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("HUMANEVAL_TEST_SECRET", "hunter2")
    program = "import os\nassert 'HUMANEVAL_TEST_SECRET' not in os.environ\n"
    assert _run_check(program, timeout=10) is True


def test_sandbox_timeout_still_fails():
    assert _run_check("while True:\n    pass\n", timeout=1) is False


# --- kernel-enforced confinement (macOS sandbox-exec) -------------------------


@_MACOS_SANDBOX
def test_sandbox_blocks_writes_outside_workdir(tmp_path: Path):
    target = tmp_path / "escape.txt"
    program = f"open({str(target)!r}, 'w').write('pwned')\n"
    assert _run_check(program, timeout=10) is False
    assert not target.exists()
    # The unconfined legacy mode shows the write is otherwise possible.
    assert _run_check(program, timeout=10, isolation="none") is True


@_MACOS_SANDBOX
def test_sandbox_denies_network():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        program = (
            "import socket\n"
            f"socket.create_connection(('127.0.0.1', {port}), timeout=2).close()\n"
        )
        assert _run_check(program, timeout=10) is False
        assert _run_check(program, timeout=10, isolation="none") is True


# --- sandbox cleanup -----------------------------------------------------------


class _TrackingSandbox(human_eval.WorkspaceSandbox):
    instances: ClassVar[list[_TrackingSandbox]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopped = False
        _TrackingSandbox.instances.append(self)

    async def stop(self) -> None:
        await super().stop()
        self.stopped = True


def test_sandbox_is_stopped_after_sync_check(monkeypatch):
    _TrackingSandbox.instances = []
    monkeypatch.setattr(human_eval, "WorkspaceSandbox", _TrackingSandbox)
    assert _run_check("pass\n", timeout=10) is True
    assert [s.stopped for s in _TrackingSandbox.instances] == [True]


def test_sandbox_is_stopped_when_graded_inside_event_loop(monkeypatch):
    _TrackingSandbox.instances = []
    monkeypatch.setattr(human_eval, "WorkspaceSandbox", _TrackingSandbox)

    async def grade_in_loop() -> bool:
        return _run_check("pass\n", timeout=10)

    assert asyncio.run(grade_in_loop()) is True
    assert [s.stopped for s in _TrackingSandbox.instances] == [True]


def test_run_check_rejects_unknown_mode():
    with pytest.raises(ValueError, match="isolation"):
        _run_check("pass\n", timeout=1, isolation="chroot")


# --- docker backend ----------------------------------------------------------


def test_docker_command_is_locked_down():
    limits = ResourceLimits(memory_mb=256, file_size_mb=1, processes=0)
    command = _docker_command("c1", "python:3.12-slim", "pass\n", 15.0, limits)
    joined = " ".join(command)
    assert command[:2] == ["docker", "run"]
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "--memory 256m --memory-swap 256m" in joined
    assert "--pids-limit 2" in joined
    assert "--ulimit core=0" in joined
    assert "--ulimit cpu=16" in joined
    assert f"--ulimit fsize={1024 * 1024}" in joined
    assert command[-8:] == [
        "python:3.12-slim",
        "timeout",
        "-s",
        "KILL",
        "15.0",
        "python",
        "-c",
        "pass\n",
    ]


def test_docker_command_omits_unlimited_caps():
    limits = ResourceLimits(memory_mb=None, file_size_mb=None, processes=None)
    command = _docker_command("c1", "img", "pass\n", 5.0, limits)
    assert "--memory" not in command
    assert "--pids-limit" not in command
    assert not any(arg.startswith("fsize=") for arg in command)


def test_docker_default_image_matches_host_python():
    bench = HumanEval()
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert bench._docker_image() == f"python:{version}-slim"
    custom = HumanEval(HumanEvalConfig(docker_image="my/python:1"))
    assert custom._docker_image() == "my/python:1"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    image = human_eval._default_docker_image()
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return probe.returncode == 0


@pytest.mark.skipif(not _docker_ready(), reason="needs Docker with the python image")
def test_docker_backend_scores_and_denies_network():
    assert _run_check("assert 1 + 1 == 2\n", timeout=30, isolation="docker") is True
    assert _run_check("assert 1 + 1 == 3\n", timeout=30, isolation="docker") is False
    program = "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)\n"
    assert _run_check(program, timeout=30, isolation="docker") is False


@pytest.mark.parametrize("name", ["benchmarks.example.yaml", "benchmarks.bedrock.yaml"])
def test_shipped_yaml_humaneval_section_builds(name):
    path = Path(human_eval.__file__).parents[1] / name
    bench = get_benchmark("humaneval", load_config_file(path, "humaneval"))
    assert bench.config == HumanEvalConfig()


# --- review follow-ups: validation, off-loop grading, infra errors -------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("timeout_s", "0"),
        ("timeout_s", "-1"),
        ("memory_limit_mb", "0"),
        ("memory_limit_mb", "63"),
        ("max_file_size_mb", "0"),
        ("max_processes", "-1"),
    ],
)
def test_config_rejects_out_of_range_limits(key, value):
    with pytest.raises(ValueError, match=key):
        get_benchmark("humaneval", {key: value})


def test_config_accepts_boundary_limits():
    bench = get_benchmark(
        "humaneval",
        {"memory_limit_mb": "64", "max_file_size_mb": "1", "max_processes": "0"},
    )
    assert bench.config.memory_limit_mb == 64


def test_grade_inside_event_loop_runs_off_the_loop():
    sleepy = RunResult(
        output="import time\ndef add(a, b):\n    time.sleep(0.5)\n    return a + b\n",
        session=Session(),
    )
    bench = HumanEval(HumanEvalConfig(isolation="sandbox"))

    async def grade_while_ticking() -> tuple[bool, int]:
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        pending = bench.grade(_add_task(), sleepy)
        assert inspect.isawaitable(pending)
        score = await pending
        ticker.cancel()
        return score.passed, ticks

    passed, ticks = asyncio.run(grade_while_ticking())
    assert passed is True
    assert ticks > 10  # the loop kept running while the check executed


def test_build_harness_resolves_isolation_before_grading(monkeypatch):
    calls = _fake_docker(monkeypatch, available=False, present=False)
    bench = HumanEval(HumanEvalConfig(isolation="docker"))
    bench.build_harness(_add_task(), ScriptedModel(Response(text="x")), Session())
    assert calls == ["info"]
    bench.grade(_add_task(), _correct_result())
    assert calls == ["info"]


def test_isolation_resolved_once_across_threads(monkeypatch):
    calls: list[str] = []

    def slow_resolve(requested, image):
        calls.append(requested)
        time.sleep(0.2)
        return "sandbox"

    monkeypatch.setattr(human_eval, "resolve_isolation", slow_resolve)
    bench = HumanEval(HumanEvalConfig(isolation="docker"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        modes = list(pool.map(lambda _: bench.isolation, range(4)))
    assert modes == ["sandbox"] * 4
    assert calls == ["docker"]


class _SingleTaskHumanEval(HumanEval):
    def load_tasks(self, *, limit=None):
        return [_add_task()]


def test_runner_grades_humaneval_end_to_end():
    bench = _SingleTaskHumanEval(HumanEvalConfig(isolation="sandbox"))
    model = ScriptedModel(Response(text="def add(a, b):\n    return a + b\n"))
    report = asyncio.run(Runner(bench, model, k=1).run())
    [attempt] = report.attempts
    assert attempt.error is None
    assert attempt.passed is True
    assert attempt.score.detail["isolation"] == "sandbox"


def _fake_docker_run(monkeypatch, returncode: int, stderr: bytes = b""):
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, returncode, b"", stderr)

    monkeypatch.setattr(human_eval.subprocess, "run", run)


@pytest.mark.parametrize("code", [125, 126, 127])
def test_docker_infra_failure_raises_instead_of_failing(monkeypatch, code):
    _fake_docker_run(monkeypatch, code, b"timeout: not found")
    with pytest.raises(human_eval.IsolationError, match=f"exited {code}"):
        _run_check("pass\n", timeout=5, isolation="docker", docker_image="img")


@pytest.mark.parametrize(("code", "passed"), [(0, True), (1, False), (137, False)])
def test_docker_program_exit_codes_score_normally(monkeypatch, code, passed):
    _fake_docker_run(monkeypatch, code)
    assert _run_check("pass\n", timeout=5, isolation="docker") is passed


def test_runner_records_docker_infra_failure_as_error(monkeypatch):
    monkeypatch.setattr(human_eval, "resolve_isolation", lambda *_: "docker")
    bench = _SingleTaskHumanEval(HumanEvalConfig(isolation="docker"))
    model = ScriptedModel(Response(text="def add(a, b):\n    return a + b\n"))
    _fake_docker_run(monkeypatch, 125, b"docker: Error response from daemon")
    report = asyncio.run(Runner(bench, model, k=1).run())
    [attempt] = report.attempts
    assert attempt.passed is False
    assert attempt.error is not None
    assert "IsolationError" in attempt.error


@_POSIX
def test_sandbox_child_cpu_time_is_capped():
    program = (
        "import resource\n"
        "limit = resource.getrlimit(resource.RLIMIT_CPU)\n"
        "assert limit[0] == 11, limit\n"
    )
    assert _run_check(program, timeout=10) is True


def test_rlimits_add_process_allowance_to_user_count():
    allowed = dict(ResourceLimits(processes=3).rlimits(1, nproc_base=100))
    assert allowed["RLIMIT_NPROC"] == 103
    # Zero means no forking at all, whatever the user already runs.
    forbidden = dict(ResourceLimits(processes=0).rlimits(1, nproc_base=100))
    assert forbidden["RLIMIT_NPROC"] == 0


@_POSIX
@pytest.mark.skipif(
    os.name == "posix" and os.geteuid() == 0, reason="root ignores RLIMIT_NPROC"
)
def test_sandbox_positive_process_allowance_permits_fork():
    program = (
        "import os\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os._exit(0)\n"
        "os.waitpid(pid, 0)\n"
    )
    assert _run_check(program, timeout=10, limits=ResourceLimits(processes=4)) is True


@_POSIX
def test_user_task_count_sees_this_process():
    count = human_eval._user_task_count()
    assert count is not None
    assert count >= 1


def test_grade_reports_kernel_confinement():
    score = HumanEval(HumanEvalConfig(isolation="none")).grade(
        _add_task(), _correct_result()
    )
    assert score.detail["kernel_confined"] is False
    assert human_eval.kernel_confined("docker") is True


@_MACOS_SANDBOX
def test_macos_sandbox_is_kernel_confined():
    assert human_eval.kernel_confined("sandbox") is True


@pytest.mark.parametrize("enforced", [True, False])
def test_kernel_confined_sandbox_mirrors_select_backend(monkeypatch, enforced):
    # Covers platforms (e.g. Linux with bubblewrap) without needing one in CI.
    monkeypatch.setattr(
        human_eval,
        "select_backend",
        lambda _mode: SimpleNamespace(enforced=enforced),
    )
    assert human_eval.kernel_confined("sandbox") is enforced


def test_unconfined_sandbox_resolution_warns(monkeypatch, caplog):
    monkeypatch.setattr(human_eval, "kernel_confined", lambda _mode: False)
    bench = HumanEval(HumanEvalConfig(isolation="sandbox"))
    with caplog.at_level("WARNING", logger=human_eval.__name__):
        assert bench.isolation == "sandbox"
    assert "does not confine file writes or network" in caplog.text
