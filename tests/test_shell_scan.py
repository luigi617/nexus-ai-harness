from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from nexus_ai_harness.protocols.sandbox import SandboxResult, SandboxViolation
from nexus_ai_harness.services.process import TIMEOUT_RETURNCODE, run_process
from nexus_ai_harness.services.shell import ShellInvocation, scan_command


def programs(command: str) -> list[str]:
    scan = scan_command(command)
    assert scan.opaque == [], scan.opaque
    return scan.programs


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls -l src", ["ls"]),
        ("cd src && pytest -x 2>&1 | tail -20", ["pytest", "tail"]),
        ("a | b || c && d; e & f", ["a", "b", "c", "d", "e", "f"]),
        ("ls\npwd\nrm x", ["ls", "rm"]),
        ('FOO=1 BAR="a b" python -c "print(1)" > out.txt', ["python"]),
        ("cat a >> b 2> c < d", ["cat"]),
        ("(cd a; make)", ["make"]),
        ("{ rm x; }", ["rm"]),
        ("if [ -f x ]; then cat x; else echo no; fi", ["cat"]),
        ("for f in *.py; do wc -l $f; done", ["wc"]),
        ("while read -r l; do grep x; done < f", ["read", "grep"]),
        ("! grep x f", ["grep"]),
        ("time -p make", ["make"]),
        ("function f { rm x; }", ["rm"]),
        ("'rm' x", ["rm"]),
        ("r\\m x", ["rm"]),
        ("/bin/rm x", ["/bin/rm"]),
        ("env A=1 nohup timeout 5 rm x", ["env", "nohup", "timeout", "rm"]),
        ("xargs -n 1 rm", ["xargs", "rm"]),
        ("exec python app.py", ["python"]),
        ("command -v python", ["python"]),
        ('find . -name "*.py" -exec grep -n foo {} \\;', ["find", "grep"]),
        ("vcs log | head -5 # rm everything", ["vcs", "head"]),
        ('export PATH="$PATH:/x"; python', ["python"]),
        ("echo ${HOME} $1 $?", []),
        ("cat > f.py <<'EOF'\nimport os\nrm $(x)\nEOF\npython f.py", ["cat", "python"]),
        ("cat <<-EOF\n\trm\n\tEOF\nls", ["cat", "ls"]),
        ("grep x <<< $HOME", ["grep"]),
    ],
)
def test_scan_finds_every_program(command, expected):
    assert programs(command) == expected


def test_scan_records_assignments():
    scan = scan_command("A=1 B+=2 cmd; C=3; export D E=5 -n; env F=6 prog")
    assert scan.assignments == ["A", "B", "C", "D", "E", "F"]
    assert scan.programs == ["cmd", "env", "prog"]
    assert scan_command('"A=1" cmd').assignments == []


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("echo $(rm -rf x)", "command substitution"),
        ("echo `rm x`", "command substitution"),
        ('echo "$(rm x)"', "command substitution"),
        ("diff <(ls) b", "process substitution"),
        ("cat <<EOF\nrm $(x)\nEOF", "heredoc"),
        ("eval ls", "evaluates code"),
        ("source ./env.sh", "evaluates code"),
        (". ./env.sh", "evaluates code"),
        ("$CMD arg", "computed at run time"),
        ("{rm,x}", "computed at run time"),
        ("r*m x", "computed at run time"),
        ("echo ${x:-y}", "complex parameter expansion"),
        ("echo $((1 + 2))", "arithmetic"),
        ("((i++))", "arithmetic"),
        ("[[ -n x ]]", "evaluates code"),
        ("case x in a) rm;; esac", "case"),
        ('echo "unterminated', "unterminated quote"),
        ("printf -v x hi", "printf -v"),
        ("[ -v 'a[0]' ]", "-v"),
    ],
)
def test_scan_reports_what_it_cannot_see(command, reason):
    scan = scan_command(command)
    assert any(reason in item for item in scan.opaque), scan.opaque


# --- ShellInvocation ------------------------------------------------------


def _run(invocation: ShellInvocation, cwd: Path, timeout: float = 10):
    return run_process(
        invocation.argv, cwd=cwd, env={"PATH": "/usr/bin:/bin"}, timeout=timeout
    )


def _confine(root: Path):
    def check(path: str) -> Path:
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise SandboxViolation(path)
        return resolved

    return check


def test_invocation_reports_cwd_and_env_changes(tmp_path):
    root = tmp_path.resolve()
    (root / "sub").mkdir()
    invocation = ShellInvocation.build(
        "cd sub; export NEW=1; unset OLD; echo done",
        cwd=str(root),
        env={"OLD": "x"},
    )
    result = invocation.parse(_run(invocation, root), confine=_confine(root), root=root)
    assert result.stdout == "done\n"
    assert result.cwd == str(root / "sub")
    assert result.env == {"NEW": "1", "OLD": None}


def test_invocation_keeps_exit_code_and_reports_state_after_exit(tmp_path):
    root = tmp_path.resolve()
    (root / "sub").mkdir()
    invocation = ShellInvocation.build("cd sub; exit 7", cwd=str(root))
    result = invocation.parse(_run(invocation, root), confine=_confine(root), root=root)
    assert result.returncode == 7
    assert result.cwd == str(root / "sub")


def test_invocation_syntax_error_still_reports_state(tmp_path):
    root = tmp_path.resolve()
    invocation = ShellInvocation.build("if then", cwd=str(root))
    result = invocation.parse(_run(invocation, root), confine=_confine(root), root=root)
    assert result.returncode != 0
    assert result.cwd == str(root)


def test_invocation_positional_args_are_not_leaked(tmp_path):
    root = tmp_path.resolve()
    invocation = ShellInvocation.build('echo "[$#]"', cwd=str(root), env={"A": "1"})
    result = invocation.parse(_run(invocation, root), confine=_confine(root), root=root)
    assert result.stdout == "[0]\n"


def test_invocation_parse_without_markers_returns_raw_output():
    invocation = ShellInvocation.build("true", cwd="/")
    raw = SandboxResult(returncode=126, stdout="boom", stderr="err")
    result = invocation.parse(raw, confine=Path, root=Path("/"))
    assert (result.returncode, result.stdout, result.stderr) == (126, "boom", "err")
    assert result.cwd is None
    assert result.env == {}


# --- run_process ----------------------------------------------------------


def test_run_process_times_out_and_kills_the_group(tmp_path):
    started = time.monotonic()
    result = run_process(
        ["/bin/sh", "-c", "echo early; sleep 30 | cat"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        timeout=0.5,
    )
    assert result.timed_out
    assert result.returncode == TIMEOUT_RETURNCODE
    assert "early" in result.stdout
    assert time.monotonic() - started < 10


def test_run_process_closes_stdin_and_can_merge_stderr(tmp_path):
    code = (
        "import sys; print(repr(sys.stdin.read()), flush=True); "
        "print('e', file=sys.stderr)"
    )
    result = run_process(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={},
        timeout=10,
        merge_stderr=True,
    )
    assert result.stdout.split() == ["''", "e"]
    assert result.stderr == ""
