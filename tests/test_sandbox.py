from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
from nexus_ai_harness.protocols.sandbox import Sandbox, SandboxResult, SandboxViolation


def test_satisfies_protocol(tmp_path):
    assert isinstance(WorkspaceSandbox(tmp_path), Sandbox)


def test_init_creates_missing_root(tmp_path):
    root = tmp_path / "workspace"
    sandbox = WorkspaceSandbox(root)
    assert root.is_dir()
    assert sandbox.root == root.resolve()


# --- resolve_path --------------------------------------------------------


def test_resolve_path_returns_path_under_root(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    resolved = sandbox.resolve_path("sub/file.txt")
    assert resolved.is_relative_to(tmp_path.resolve())
    assert resolved == (tmp_path.resolve() / "sub" / "file.txt")


def test_resolve_path_rejects_dotdot_escape(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path / "workspace")
    with pytest.raises(SandboxViolation):
        sandbox.resolve_path("../escape")


def test_resolve_path_rejects_absolute_outside_root(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    with pytest.raises(SandboxViolation):
        sandbox.resolve_path("/etc/passwd")


def test_resolve_path_rejects_symlink_escape(tmp_path):
    root = tmp_path / "workspace"
    sandbox = WorkspaceSandbox(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises(SandboxViolation):
        sandbox.resolve_path("link/secret.txt")


# --- check_command -------------------------------------------------------


def test_check_command_allows_allowlisted(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, allowed_commands=["echo"])
    sandbox.check_command(["echo", "hi"])  # does not raise


def test_check_command_rejects_non_allowlisted(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, allowed_commands=["echo"])
    with pytest.raises(SandboxViolation):
        sandbox.check_command(["rm", "-rf", "/"])


def test_check_command_denylist_always_blocks(tmp_path):
    sandbox = WorkspaceSandbox(
        tmp_path, allowed_commands=["rm"], denied_commands=["rm"]
    )
    with pytest.raises(SandboxViolation):
        sandbox.check_command(["rm", "x"])


def test_check_command_rejects_empty(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    with pytest.raises(SandboxViolation):
        sandbox.check_command([])


def test_check_command_denylist_matches_basename(tmp_path):
    # A denylist keyed on a bare name must still catch an absolute/relative path.
    sandbox = WorkspaceSandbox(tmp_path, denied_commands=["rm"])
    for argv in (["/bin/rm", "-rf", "x"], ["./rm"], ["rm"]):
        with pytest.raises(SandboxViolation):
            sandbox.check_command(argv)


def test_check_command_allowlist_matches_basename(tmp_path):
    # An allowlisted bare name permits the same program given by full path.
    sandbox = WorkspaceSandbox(tmp_path, allowed_commands=["echo"])
    sandbox.check_command(["/bin/echo", "hi"])  # does not raise
    with pytest.raises(SandboxViolation):
        sandbox.check_command(["/bin/cat", "x"])


# --- run_command ---------------------------------------------------------


def test_run_command_captures_stdout(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    result = sandbox.run_command(
        [sys.executable, "-c", "print('hello sandbox')"],
        timeout=30,
    )
    assert isinstance(result, SandboxResult)
    assert result.returncode == 0
    assert "hello sandbox" in result.stdout


def test_run_command_forwards_stdin(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    result = sandbox.run_command(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        timeout=30,
        input="piped",
    )
    assert result.stdout == "piped"


def test_run_command_denied_raises(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, denied_commands=["echo"])
    with pytest.raises(SandboxViolation):
        sandbox.run_command(["echo", "nope"], timeout=5)


def test_run_command_timeout_returns_nonzero(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    result = sandbox.run_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout=0.2,
    )
    assert result.returncode != 0
    assert "timed out" in result.stderr


def test_resolve_path_accepts_root_itself(tmp_path):
    # '' and '.' resolve to the root; both must be accepted (ListDir defaults to
    # '.'), since is_relative_to is true when the path equals the root.
    sandbox = WorkspaceSandbox(tmp_path)
    assert sandbox.resolve_path("") == tmp_path.resolve()
    assert sandbox.resolve_path(".") == tmp_path.resolve()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec only")
def test_profile_escapes_backslash_in_root(tmp_path):
    # A backslash is a legal path char; the SBPL profile must escape it (and any
    # quote) so the string literal can't terminate early or inject code.
    root = tmp_path / 'ws\\a"b'
    sandbox = WorkspaceSandbox(root)
    profile = sandbox._profile()
    # Backslashes are doubled and quotes escaped in the emitted subpath line.
    assert '(allow file-write* (subpath "' in profile
    escaped = str(root.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    assert escaped in profile


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec only")
def test_backslash_root_profile_is_parseable_on_macos(tmp_path):
    # With correct escaping, sandbox-exec accepts the profile and runs the
    # command rather than aborting with an SBPL parse error.
    sandbox = WorkspaceSandbox(tmp_path / "ws\\data")
    result = sandbox.run_command([sys.executable, "-c", "print('ok')"], timeout=30)
    assert result.returncode == 0
    assert "ok" in result.stdout


def test_env_tmpdir_is_real_and_not_inside_root(tmp_path):
    # TMPDIR is the host temp dir (writable via the profile), kept out of the
    # workspace so it never pollutes list_dir or collides with model files.
    sandbox = WorkspaceSandbox(tmp_path)
    tmpdir = Path(sandbox._env()["TMPDIR"])
    assert tmpdir.is_dir()
    assert not tmpdir.resolve().is_relative_to(tmp_path.resolve())


def test_scratch_dir_does_not_pollute_workspace_listing(tmp_path, monkeypatch):
    # Running a command must not create a scratch dir inside the root (regression
    # for a prior <root>/.tmp that showed up in list_dir).
    from nexus_ai_harness.plugins.tools import ListDir
    from tests.conftest import make_ctx

    sandbox = WorkspaceSandbox(tmp_path)
    sandbox.run_command([sys.executable, "-c", "print('hi')"], timeout=30)
    listing = ListDir().run({"path": "."}, make_ctx(sandbox))
    assert ".tmp" not in listing


def test_profile_allows_writes_to_tmpdir(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    profile = sandbox._profile()
    assert sandbox._sbpl(sandbox._tmpdir) in profile


def test_lifecycle_stop_removes_scratch_dir(tmp_path):
    import asyncio

    from nexus_ai_harness.protocols.lifecycle import Lifecycle

    sandbox = WorkspaceSandbox(tmp_path)
    assert isinstance(sandbox, Lifecycle)
    scratch = sandbox._tmpdir
    assert scratch.is_dir()
    asyncio.run(sandbox.stop())
    assert not scratch.exists()
    # start() is restart-safe: it recreates the scratch dir.
    asyncio.run(sandbox.start(None))
    assert scratch.is_dir()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec only")
def test_run_command_can_write_to_tmpdir_on_macos(tmp_path):
    # A subprocess that follows $TMPDIR must be able to write there under the
    # macOS profile (regression for TMPDIR pointing outside the writable root).
    sandbox = WorkspaceSandbox(tmp_path)
    script = (
        "import os, tempfile; "
        "f = tempfile.NamedTemporaryFile(dir=os.environ['TMPDIR'], delete=False); "
        "f.write(b'x'); f.close(); print('wrote', f.name)"
    )
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    assert result.returncode == 0, result.stderr
    assert "wrote" in result.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec only")
def test_run_command_confines_writes_on_macos(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    outside = tmp_path.parent / "escape_target.txt"
    script = f"open({str(outside)!r}, 'w').write('x')"
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    # sandbox-exec should block the write outside the root.
    assert result.returncode != 0
    assert not outside.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec only")
def test_run_command_allows_writes_inside_root_on_macos(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    target = tmp_path / "inside.txt"
    script = f"open({str(target)!r}, 'w').write('ok')"
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    assert result.returncode == 0
    assert target.read_text() == "ok"


# --- shell strings --------------------------------------------------------


def test_run_command_timeout_sets_timed_out(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    result = sandbox.run_command(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2
    )
    assert result.timed_out
    assert result.returncode == 124


def test_run_command_closes_stdin_without_input(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    code = "import sys; sys.stdout.write(repr(sys.stdin.read()))"
    result = sandbox.run_command([sys.executable, "-c", code], timeout=30)
    assert result.stdout == "''"


def test_check_shell_allowlist(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, allowed_commands=["python", "pytest", "env"])
    sandbox.check_shell("cd src && pytest -x 2>&1 > log.txt; echo done")
    with pytest.raises(SandboxViolation, match="not in allowlist: 'tail'"):
        sandbox.check_shell("pytest | tail")
    with pytest.raises(SandboxViolation, match="cannot verify"):
        sandbox.check_shell("python $(echo x)")
    # A variable like PATH could make an allowed name run something else.
    for command in ("PATH=. pytest", "export PATH=.; pytest", "env PATH=. python"):
        with pytest.raises(SandboxViolation, match="environment variables"):
            sandbox.check_shell(command)


def test_check_shell_denylist_allows_assignments(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, denied_commands=["rm"])
    sandbox.check_shell("export MODE=test; FOO=1 python x.py")


@pytest.mark.parametrize(
    "command",
    [
        "for PATH in ./evil; do ls; done",
        "set -a; for GIT_EXTERNAL_DIFF in ./x; do ls; done",
        'env -S"touch pwned"',
        "exec {PATH}>/dev/null; ls",
    ],
)
def test_check_shell_allowlist_refuses_loop_and_split_bypasses(tmp_path, command):
    sandbox = WorkspaceSandbox(tmp_path, allowed_commands=["ls", "env"])
    with pytest.raises(SandboxViolation):
        sandbox.check_shell(command)


@pytest.mark.parametrize(
    "command",
    [
        "declare -x PATH=/tmp/evil:$PATH; ls",
        "local PATH=/tmp/evil; ls",
        "typeset -x LD_PRELOAD=/tmp/evil.so; ls",
        "readonly GIT_EXTERNAL_DIFF=/tmp/evil; ls",
        "read -r PATH <<< /tmp/evil; ls",
        "mapfile -t PATH <<< /tmp/evil; ls",
    ],
)
def test_check_shell_allowlist_refuses_variable_setting_builtins(tmp_path, command):
    sandbox = WorkspaceSandbox(
        tmp_path,
        allowed_commands=[
            "ls",
            "declare",
            "local",
            "typeset",
            "readonly",
            "read",
            "mapfile",
        ],
    )
    with pytest.raises(SandboxViolation):
        sandbox.check_shell(command)


def test_run_shell_allowlist_blocks_env_split_string(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, allowed_commands=["ls", "env"])
    with pytest.raises(SandboxViolation):
        sandbox.run_shell('env -S"touch pwned"', timeout=10)
    assert not (tmp_path / "pwned").exists()


@pytest.mark.parametrize(
    "command",
    [
        "echo evil > /etc/nexus-test",
        "ls >> ../outside",
        "echo x > /dev/tcp/127.0.0.1/9",
        "cat < /dev/udp/127.0.0.1/9",
        "cd .. && ls > x",
        "cd src && ls > ../out",
        "cd $d; ls > x",
        "while true; do cd src; ls > x; done",
        "ls > ~/x",
        'ls > "$f"',
    ],
)
def test_check_shell_allowlist_refuses_unverifiable_redirects(tmp_path, command):
    (tmp_path / "src").mkdir()
    sandbox = WorkspaceSandbox(tmp_path, allowed_commands=["ls", "cat"])
    with pytest.raises(SandboxViolation):
        sandbox.check_shell(command)


@pytest.mark.parametrize(
    "command",
    [
        "ls > out.txt 2>&1",
        "ls > /dev/null; echo hi >&2",
        "ls > {root}/inside.txt",
        "cd src && ls > log.txt",
        "cd src; cd pkg; ls > log.txt",
        "ls > src/../out",
    ],
)
def test_check_shell_allowlist_permits_redirects_inside_root(tmp_path, command):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    sandbox = WorkspaceSandbox(tmp_path, allowed_commands=["ls"])
    sandbox.check_shell(command.replace("{root}", str(sandbox.root)))


def test_check_shell_resolves_relative_redirects_from_the_start_dir(tmp_path):
    (tmp_path / "a").mkdir()
    sandbox = WorkspaceSandbox(tmp_path, allowed_commands=["ls"])
    sandbox.check_shell("ls > ../x", cwd="a")
    with pytest.raises(SandboxViolation):
        sandbox.check_shell("ls > ../x")


def test_check_shell_network_redirect_allowed_with_network(tmp_path):
    sandbox = WorkspaceSandbox(
        tmp_path, allowed_commands=["ls", "cat"], allow_network=True
    )
    sandbox.check_shell("cat < /dev/tcp/127.0.0.1/9; ls")


def test_denylist_still_permits_redirects(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, denied_commands=["rm"])
    sandbox.check_shell("echo x > ../outside; for x in a; do ls; done")


def test_run_shell_deleted_cwd_keeps_output(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    result = sandbox.run_shell("mkdir -p d && cd d && rmdir ../d; echo x", timeout=10)
    assert (result.returncode, result.stdout) == (0, "x\n")
    assert result.cwd is None or not Path(result.cwd).exists()


def test_check_shell_denylist(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, denied_commands=["rm"])
    sandbox.check_shell("ls | wc -l")
    for command in ("ls; rm x", "find . -exec rm {} +", "xargs rm", "env A=1 rm"):
        with pytest.raises(SandboxViolation, match="not permitted"):
            sandbox.check_shell(command)


def test_run_shell_returns_state(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    (sandbox.root / "sub").mkdir()
    result = sandbox.run_shell("cd sub && export K=v && echo hi", timeout=30)
    assert result.returncode == 0
    assert result.stdout == "hi\n"
    assert result.cwd == str(sandbox.root / "sub")
    assert result.env == {"K": "v"}

    again = sandbox.run_shell(
        'echo "$K"; pwd -P', timeout=30, cwd=result.cwd, env=result.env
    )
    assert again.stdout.splitlines() == ["v", str(sandbox.root / "sub")]


def test_run_shell_rejects_cwd_outside_root(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path / "root")
    with pytest.raises(SandboxViolation):
        sandbox.run_shell("true", timeout=30, cwd=str(tmp_path))


def test_protocol_run_shell_defaults_to_not_implemented(tmp_path):
    class ArgvOnly(Sandbox):
        def resolve_path(self, path):
            return tmp_path

        def check_command(self, argv):
            return None

        def run_command(self, argv, *, timeout, input=None):  # noqa: A002
            return SandboxResult(0, "", "")

    with pytest.raises(NotImplementedError):
        ArgvOnly().run_shell("ls", timeout=1)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec only")
def test_run_shell_confines_writes_on_macos(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path / "root")
    target = tmp_path / "outside.txt"
    result = sandbox.run_shell(f"echo x > {target}", timeout=30)
    assert result.returncode != 0
    assert not target.exists()
