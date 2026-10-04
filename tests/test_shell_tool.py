from __future__ import annotations

import shlex
import sys
import threading
import time
from pathlib import Path

import pytest

from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
from nexus_ai_harness.plugins.tools import Shell
from nexus_ai_harness.plugins.tools import shell as shell_module
from nexus_ai_harness.plugins.tools.shell import (
    _DEFAULT_TIMEOUT,
    _MAX_TIMEOUT,
    ShellState,
    _coerce_timeout,
    _truncate,
)
from nexus_ai_harness.protocols.sandbox import Sandbox, SandboxResult, SandboxViolation
from nexus_ai_harness.services.shell import _denormalize_msys_path
from tests.conftest import make_ctx


def test_shell_echo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    result = Shell().run({"command": "echo hi"}, ctx)
    assert result == "exit=0\nhi\n"


def test_shell_reports_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    result = Shell().run({"command": "exit 3"}, ctx)
    assert result.startswith("exit=3")


def test_shell_empty_command_returns_error():
    ctx = make_ctx()
    assert Shell().run({"command": "   "}, ctx) == "error: empty command"


def test_shell_unbalanced_quotes_is_a_shell_syntax_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    result = Shell().run({"command": "echo 'unterminated"}, ctx)
    assert result.startswith("exit=")
    assert not result.startswith("exit=0")


def test_shell_missing_program_reports_127(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    result = Shell().run({"command": "definitely_not_a_real_binary_xyz"}, ctx)
    assert result.startswith("exit=127")


def test_shell_supports_pipes_redirects_and_lists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    assert "2" in Shell().run({"command": "printf 'a\\nb\\n' | wc -l"}, ctx)
    result = Shell().run({"command": "echo hi > f.txt && cat f.txt"}, ctx)
    assert result == "exit=0\nhi\n"
    assert (tmp_path / "f.txt").read_text() == "hi\n"
    assert "no" in Shell().run({"command": "false || echo no"}, ctx)
    assert "a.txt" in Shell().run({"command": "touch a.txt; ls *.txt"}, ctx)


def test_shell_merges_stderr_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    result = Shell().run({"command": "echo one; echo two >&2; echo three"}, ctx)
    assert result == "exit=0\none\ntwo\nthree\n"


def test_shell_cwd_persists_across_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    ctx = make_ctx()
    shell = Shell()

    moved = shell.run({"command": "cd sub"}, ctx)
    assert "(cwd is now sub)" in moved
    assert shell.run({"command": "pwd -P"}, ctx).strip().endswith("/sub")
    assert ctx.state(ShellState).cwd == str((tmp_path / "sub").resolve())

    shell.run({"command": "touch made_here"}, ctx)
    assert (tmp_path / "sub" / "made_here").exists()


def test_shell_subshell_cd_does_not_persist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    ctx = make_ctx()
    shell = Shell()

    shell.run({"command": "(cd sub)"}, ctx)
    assert ctx.state(ShellState).cwd is None


def test_shell_cwd_outside_root_is_reset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    shell = Shell()

    result = shell.run({"command": "cd /"}, ctx)
    assert "outside the workspace" in result
    assert ctx.state(ShellState).cwd is None
    pwd = shell.run({"command": "pwd -P"}, ctx)
    # pwd -P is the command's own output, not our tracked state: under Git
    # Bash on Windows it reports its own MSYS form (/c/...) regardless.
    last_line = _denormalize_msys_path(pwd.strip().splitlines()[-1])
    assert last_line == str(tmp_path.resolve())


def test_shell_env_persists_across_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    shell = Shell()

    shell.run({"command": "export GREETING='hello world'"}, ctx)
    assert shell.run({"command": 'echo "$GREETING"'}, ctx) == "exit=0\nhello world\n"
    shell.run({"command": "unset GREETING"}, ctx)
    assert shell.run({"command": 'echo "[$GREETING]"'}, ctx) == "exit=0\n[]\n"


def test_shell_env_persistence_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    shell = Shell(persist_env=False)

    shell.run({"command": "export GREETING=hi"}, ctx)
    assert shell.run({"command": 'echo "[$GREETING]"'}, ctx) == "exit=0\n[]\n"


def test_shell_state_is_per_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    shell = Shell()
    first, second = make_ctx(), make_ctx()

    shell.run({"command": "cd sub; export X=1"}, first)
    assert shell.run({"command": 'echo "[$X]"'}, second) == "exit=0\n[]\n"
    assert second.state(ShellState).cwd is None


def test_shell_stdin_is_closed(tmp_path, monkeypatch):
    # A command that reads stdin must see EOF, not block on the host terminal.
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    result = Shell().run({"command": "cat", "timeout": 10}, ctx)
    assert result == "exit=0\n"


def test_shell_timeout_result_shape_unsandboxed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    result = Shell().run({"command": "echo started; sleep 5", "timeout": 0.5}, ctx)
    assert result == "exit=124 (timed out after 0.5s)\nstarted\n"


def test_shell_timeout_result_shape_sandboxed(tmp_path):
    ctx = make_ctx(WorkspaceSandbox(tmp_path))
    result = Shell().run({"command": "echo started; sleep 5", "timeout": 0.5}, ctx)
    assert result == "exit=124 (timed out after 0.5s)\nstarted\n"


@pytest.mark.parametrize("sandboxed", [False, True])
def test_shell_timeout_kills_pipeline_children(tmp_path, monkeypatch, sandboxed):
    # A grandchild holding the output pipe must not stall the capture past the
    # timeout (killing only the shell left `sleep` running with the pipe open).
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx(WorkspaceSandbox(tmp_path)) if sandboxed else make_ctx()
    started = time.monotonic()
    result = Shell().run({"command": "sleep 30 | cat", "timeout": 0.5}, ctx)
    assert result.startswith("exit=124")
    assert time.monotonic() - started < 10


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows has no process-group kill; a hard-killed bash can still "
    "finish writing the state trailer before it dies",
)
def test_shell_timeout_does_not_update_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    ctx = make_ctx()
    Shell().run({"command": "cd sub; sleep 5", "timeout": 0.5}, ctx)
    assert ctx.state(ShellState).cwd is None


def test_shell_runs_through_sandbox(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    ctx = make_ctx(sandbox)

    result = Shell().run({"command": "echo sandboxed | tr a-z A-Z"}, ctx)
    assert result == "exit=0\nSANDBOXED\n"


def test_sandboxed_shell_persists_cwd_and_env(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    (Path(sandbox.root) / "pkg").mkdir()
    ctx = make_ctx(sandbox)
    shell = Shell()

    shell.run({"command": "cd pkg && export MODE=test"}, ctx)
    result = shell.run({"command": 'echo "$MODE"; pwd -P'}, ctx)
    assert result.startswith("exit=0\ntest\n")
    assert result.strip().endswith("/pkg")


def test_sandboxed_shell_cd_escape_resets_to_root(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path / "root")
    ctx = make_ctx(sandbox)
    shell = Shell()

    result = shell.run({"command": "cd .."}, ctx)
    assert "outside the workspace" in result
    assert shell.run({"command": "pwd -P"}, ctx).strip().endswith("/root")


def test_sandbox_allowlist_checks_every_program_in_a_pipeline(tmp_path):
    ctx = make_ctx(WorkspaceSandbox(tmp_path, allowed_commands=["ls", "wc"]))

    assert Shell().run({"command": "cd . && ls | wc -l"}, ctx).startswith("exit=0")
    denied = Shell().run({"command": "ls | cat"}, ctx)
    assert denied == "error: command not in allowlist: 'cat'"


def test_sandbox_allowlist_refuses_unverifiable_strings(tmp_path):
    ctx = make_ctx(WorkspaceSandbox(tmp_path, allowed_commands=["ls"]))

    result = Shell().run({"command": "ls $(cat secret)"}, ctx)
    assert result.startswith("error: cannot verify shell command against the")


def test_sandbox_denylist_checks_programs_inside_lists(tmp_path):
    ctx = make_ctx(WorkspaceSandbox(tmp_path, denied_commands=["rm"]))

    result = Shell().run({"command": "touch a && rm a"}, ctx)
    assert result == "error: command not permitted: 'rm'"
    assert (Path(tmp_path) / "a").exists() is False  # nothing ran


def test_sandbox_denylist_allows_unverifiable_strings(tmp_path):
    # Documented: a denylist is best-effort for shell strings.
    ctx = make_ctx(WorkspaceSandbox(tmp_path, denied_commands=["rm"]))
    assert Shell().run({"command": "echo $(echo hi)"}, ctx) == "exit=0\nhi\n"


def test_sandbox_denying_the_shell_disables_shell_strings(tmp_path):
    ctx = make_ctx(WorkspaceSandbox(tmp_path, denied_commands=["bash"]))
    result = Shell().run({"command": "echo hi"}, ctx)
    assert result == "error: shell commands are not permitted"


class _ArgvOnlySandbox(Sandbox):
    """A third-party sandbox without shell-string support."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self.argvs: list[list[str]] = []

    def resolve_path(self, path: str) -> Path:
        return (self._root / path).resolve()

    def check_command(self, argv: list[str]) -> None:
        if argv[0] == "forbidden":
            raise SandboxViolation("nope")

    def run_command(self, argv, *, timeout, input=None) -> SandboxResult:  # noqa: A002
        self.check_command(argv)
        self.argvs.append(argv)
        if argv[0] == "no_such_binary":
            raise FileNotFoundError(2, "No such file or directory", argv[0])
        return SandboxResult(returncode=0, stdout="out", stderr="warn")


def test_sandbox_without_shell_support_falls_back_to_argv(tmp_path):
    sandbox = _ArgvOnlySandbox(tmp_path)
    ctx = make_ctx(sandbox)

    result = Shell().run({"command": "echo 'a b'"}, ctx)
    assert sandbox.argvs == [["echo", "a b"]]
    assert result == "exit=0\nout\n[stderr]\nwarn"
    assert Shell().run({"command": "forbidden"}, ctx) == "error: nope"
    assert Shell().run({"command": "echo 'x"}, ctx).startswith("error:")


def test_sandbox_without_shell_support_reports_127_for_missing_program(tmp_path):
    sandbox = _ArgvOnlySandbox(tmp_path)
    ctx = make_ctx(sandbox)

    result = Shell().run({"command": "no_such_binary"}, ctx)
    assert result.startswith("exit=127")


class _NoShellWorkspace(WorkspaceSandbox):
    """A real sandbox that opts out of shell strings, to exercise the fallback."""

    def run_shell(self, command, *, timeout, cwd=None, env=None):
        raise NotImplementedError


def test_shell_timeout_result_shape_fallback(tmp_path):
    ctx = make_ctx(_NoShellWorkspace(tmp_path))
    result = Shell().run({"command": "sleep 5", "timeout": 0.5}, ctx)
    assert result == "exit=124 (timed out after 0.5s)\n"


@pytest.mark.parametrize("sandboxed", [False, True])
def test_shell_background_job_returns_promptly(tmp_path, monkeypatch, sandboxed):
    monkeypatch.chdir(tmp_path)
    sandbox = WorkspaceSandbox(tmp_path) if sandboxed else None
    ctx = make_ctx(sandbox) if sandboxed else make_ctx()
    started = time.monotonic()
    result = Shell().run({"command": "sleep 30 & echo started", "timeout": 20}, ctx)
    assert time.monotonic() - started < 10
    assert result.startswith("exit=0\nstarted\n")
    # Bubblewrap gives the command its own PID namespace, so the kernel kills
    # the backgrounded sleep the moment the foreground shell exits - there's
    # nothing left lingering to report. Elsewhere, the sleep outlives the
    # shell and must be reaped explicitly.
    if not (sandbox is not None and sandbox.isolation_level == "bubblewrap"):
        assert "redirect" in result


def test_shell_parallel_call_does_not_clobber_cwd(tmp_path, monkeypatch):
    # The loop runs one response's tool calls concurrently; a call that never
    # changed directory must not reset the cwd another call moved to.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    ctx = make_ctx()
    shell = Shell()
    slow = threading.Thread(
        target=shell.run, args=({"command": "sleep 1"}, ctx), daemon=True
    )
    slow.start()
    time.sleep(0.2)
    shell.run({"command": "cd sub"}, ctx)
    slow.join(10)
    assert ctx.state(ShellState).cwd == str((tmp_path / "sub").resolve())


def test_shell_concurrent_cd_resolves_by_issue_order_not_finish_order(
    tmp_path, monkeypatch
):
    # Two calls that both cd elsewhere race on finish order; the call issued
    # later must win even if it finishes first, so an earlier call that is
    # merely slow to finish can't clobber a more recent move.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slow_target").mkdir()
    (tmp_path / "fast_target").mkdir()
    ctx = make_ctx()
    shell = Shell()
    slow = threading.Thread(
        target=shell.run,
        args=({"command": "sleep 1 && cd slow_target"}, ctx),
        daemon=True,
    )
    slow.start()
    time.sleep(0.2)
    shell.run({"command": "cd fast_target"}, ctx)  # issued second, finishes first
    slow.join(10)
    assert ctx.state(ShellState).cwd == str((tmp_path / "fast_target").resolve())


def test_shell_persisted_env_caps_variable_count(tmp_path, monkeypatch):
    # Only each value's size was capped before; an uncapped variable count
    # would make the re-embedded argv grow without bound over a long session.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shell_module, "_MAX_ENV_VARS", 3)
    ctx = make_ctx()
    shell = Shell()
    for i in range(5):
        shell.run({"command": f"export V{i}=x"}, ctx)
    env = ctx.state(ShellState).env
    assert len(env) <= 3
    assert "V0" not in env
    assert "V4" in env


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), 0, -5, "30", None, True],
)
def test_coerce_timeout_rejects_bad_values(value):
    # NaN/inf must not slip past the guard and disable the mandatory timeout;
    # non-numbers and non-positive values fall back to the default.
    assert _coerce_timeout(value) == _DEFAULT_TIMEOUT


def test_coerce_timeout_caps_and_keeps_valid():
    assert _coerce_timeout(10) == 10.0
    assert _coerce_timeout(10_000) == _MAX_TIMEOUT == 600.0
    assert _coerce_timeout(10_000, 30, 1200) == 1200.0
    assert _coerce_timeout(None, 60, 20) == 20.0  # default never exceeds the cap


def test_shell_timeout_cap_is_configurable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    shell = Shell(max_timeout=0.5)
    result = shell.run({"command": "sleep 5", "timeout": 100}, ctx)
    assert result.startswith("exit=124 (timed out after 0.5s)")


def test_shell_nan_timeout_does_not_hang(tmp_path, monkeypatch):
    # A NaN timeout previously slipped past `<= 0`; the default now applies and
    # a fast command still completes normally.
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    result = Shell().run({"command": "echo ok", "timeout": float("nan")}, ctx)
    assert result == "exit=0\nok\n"


def test_shell_non_utf8_output_does_not_crash(tmp_path, monkeypatch):
    # A command emitting invalid UTF-8 must be captured (replacement chars),
    # not raise UnicodeDecodeError out of the tool.
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    script = "import sys; sys.stdout.buffer.write(b'\\xff\\xfe\\x00bad')"
    # sys.executable on Windows contains backslashes, which bash would
    # otherwise interpret as escapes; shlex.quote keeps it one literal word.
    executable = shlex.quote(sys.executable)
    result = Shell().run({"command": f"{executable} -c {script!r}"}, ctx)
    assert result.startswith("exit=0")
    assert "bad" in result


def test_shell_long_output_keeps_head_and_tail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    command = "for i in $(seq 1 5000); do echo line$i; done"
    result = Shell(max_output=2000).run({"command": command}, ctx)
    assert "line1\n" in result
    assert "line5000" in result
    assert "chars omitted" in result
    assert len(result) < 2200


def test_truncate_cuts_at_line_breaks():
    text = "".join(f"row{i}\n" for i in range(1000))
    clipped = _truncate(text, 200)
    head, _, tail = clipped.partition("\n... (")
    assert head.startswith("row0\n")
    assert all(line.startswith("row") for line in head.splitlines())
    assert tail.splitlines()[-1] == "row999"
    assert all(line.startswith("row") for line in tail.splitlines()[1:])
