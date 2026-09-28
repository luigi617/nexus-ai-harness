from __future__ import annotations

import sys

import pytest

from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
from nexus_ai_harness.plugins.tools import Shell
from nexus_ai_harness.plugins.tools.shell import _DEFAULT_TIMEOUT, _coerce_timeout
from tests.conftest import make_ctx


def test_shell_echo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    result = Shell().run({"command": "echo hi"}, ctx)
    assert result.startswith("exit=0")
    assert "hi" in result


def test_shell_reports_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    result = Shell().run({"command": "sh -c 'exit 3'"}, ctx)
    assert result.startswith("exit=3")


def test_shell_empty_command_returns_error():
    ctx = make_ctx()
    assert Shell().run({"command": "   "}, ctx) == "error: empty command"


def test_shell_unbalanced_quotes_returns_error():
    ctx = make_ctx()
    assert Shell().run({"command": "echo 'unterminated"}, ctx).startswith("error:")


def test_shell_missing_program_returns_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    assert (
        Shell()
        .run({"command": "definitely_not_a_real_binary_xyz"}, ctx)
        .startswith("error:")
    )


def test_shell_runs_through_sandbox(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    ctx = make_ctx(sandbox)

    result = Shell().run({"command": "echo sandboxed"}, ctx)
    assert result.startswith("exit=0")
    assert "sandboxed" in result


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
    assert _coerce_timeout(10_000) == 120.0  # capped at _MAX_TIMEOUT


def test_shell_nan_timeout_does_not_hang(tmp_path, monkeypatch):
    # A NaN timeout previously slipped past `<= 0`; the default now applies and
    # a fast command still completes normally.
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    result = Shell().run({"command": "echo ok", "timeout": float("nan")}, ctx)
    assert result.startswith("exit=0")
    assert "ok" in result


def test_shell_non_utf8_output_does_not_crash(tmp_path, monkeypatch):
    # A command emitting invalid UTF-8 must be captured (replacement chars),
    # not raise UnicodeDecodeError out of the tool.
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    script = "import sys; sys.stdout.buffer.write(b'\\xff\\xfe\\x00bad')"
    result = Shell().run({"command": f"{sys.executable} -c {script!r}"}, ctx)
    assert result.startswith("exit=0")
