from __future__ import annotations

import sys
from pathlib import Path

import pytest

from plugins.sandbox import WorkspaceSandbox
from protocols.sandbox import Sandbox, SandboxResult, SandboxViolation


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
    from plugins.tools import ListDir
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

    from protocols.lifecycle import Lifecycle

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
