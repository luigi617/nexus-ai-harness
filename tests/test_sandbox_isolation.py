from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path

import pytest

from nexus_ai_harness.plugins.sandbox import (
    BubblewrapBackend,
    IsolationBackend,
    IsolationSpec,
    IsolationUnavailable,
    IsolationWarning,
    NoIsolation,
    SandboxExecBackend,
    WorkspaceSandbox,
    select_backend,
)
from nexus_ai_harness.plugins.sandbox import isolation as isolation_mod
from nexus_ai_harness.plugins.sandbox import workspace as workspace_mod

BWRAP = "/usr/bin/bwrap"


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    isolation_mod._bwrap_works.cache_clear()
    yield
    isolation_mod._bwrap_works.cache_clear()


@pytest.fixture
def fresh_warning(monkeypatch):
    monkeypatch.setattr(workspace_mod, "_warned_unenforced", False)


def _fake_which(available: dict[str, str]):
    return lambda name, *args, **kwargs: available.get(name)


def _as_linux(monkeypatch, *, bwrap: bool, probe_ok: bool = True) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        isolation_mod.shutil, "which", _fake_which({"bwrap": BWRAP} if bwrap else {})
    )
    monkeypatch.setattr(isolation_mod, "_bwrap_works", lambda exe: probe_ok)


def _spec(tmp_path: Path, *, allow_network: bool = False) -> IsolationSpec:
    return IsolationSpec(
        root=tmp_path / "root", tmpdir=tmp_path / "scratch", allow_network=allow_network
    )


def _operands(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


def _pairs(args: list[str], flag: str) -> list[tuple[str, ...]]:
    # Collect each occurrence of a two-operand bwrap flag with its operands.
    return [tuple(args[i + 1 : i + 3]) for i, a in enumerate(args) if a == flag]


# --- bubblewrap argv construction ------------------------------------------


def test_bwrap_args_mount_layout(tmp_path):
    spec = _spec(tmp_path)
    args = isolation_mod.bwrap_args(spec)
    root, scratch = str(spec.root), str(spec.tmpdir)

    assert _pairs(args, "--ro-bind") == [("/", "/")]
    assert _pairs(args, "--bind") == [(scratch, scratch), (root, root)]
    assert args[args.index("--dev") + 1] == "/dev"
    assert args[args.index("--proc") + 1] == "/proc"
    assert _operands(args, "--tmpfs") == ["/tmp", "/run"]
    assert args[args.index("--chdir") + 1] == root
    for flag in (
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
    ):
        assert flag in args


def test_bwrap_args_masks_precede_writable_binds(tmp_path):
    # Later mounts shadow earlier ones; a root under /tmp must stay visible.
    args = isolation_mod.bwrap_args(
        _spec(tmp_path),
        masks=["/tmp", "/run", "/xdg"],
        symlinks=[("/nix/store/sys", "/run/current-system")],
        expose=["/tmp/venv"],
    )
    tmpfs = [i for i, a in enumerate(args) if a == "--tmpfs"]
    binds = [i for i, a in enumerate(args) if a == "--bind"]
    assert _operands(args, "--tmpfs") == ["/tmp", "/run", "/xdg"]
    assert args.index("--ro-bind") < min(tmpfs)
    assert max(tmpfs) < args.index("--symlink") < min(binds)
    assert max(tmpfs) < args.index("--ro-bind-try") < min(binds)
    assert _pairs(args, "--symlink") == [("/nix/store/sys", "/run/current-system")]
    assert _pairs(args, "--ro-bind-try") == [("/tmp/venv", "/tmp/venv")]


def test_bwrap_args_unshares_network_only_when_denied(tmp_path):
    assert "--unshare-net" in isolation_mod.bwrap_args(_spec(tmp_path))
    allowed = isolation_mod.bwrap_args(_spec(tmp_path, allow_network=True))
    assert "--unshare-net" not in allowed


def test_masked_dirs_always_hide_tmp_and_run(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(isolation_mod, "_VAR_RUN", tmp_path / "missing")
    assert isolation_mod.masked_dirs() == ("/tmp", "/run")


def test_masked_dirs_add_real_var_run(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    var_run = tmp_path / "var-run"
    var_run.mkdir()
    monkeypatch.setattr(isolation_mod, "_VAR_RUN", var_run)
    assert isolation_mod.masked_dirs() == ("/tmp", "/run", str(var_run))


def test_masked_dirs_skip_var_run_symlink(monkeypatch, tmp_path):
    # The usual layout: /var/run -> /run, already covered by the /run mask.
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    (tmp_path / "run").mkdir()
    link = tmp_path / "var-run"
    link.symlink_to(tmp_path / "run")
    monkeypatch.setattr(isolation_mod, "_VAR_RUN", link)
    assert isolation_mod.masked_dirs() == ("/tmp", "/run")


def test_masked_dirs_add_runtime_dir_outside_run(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation_mod, "_VAR_RUN", tmp_path / "missing")
    runtime = tmp_path / "xdg"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    masks = isolation_mod.masked_dirs()
    expected = () if str(runtime).startswith("/tmp/") else (str(runtime),)
    assert masks == ("/tmp", "/run", *expected)


def test_masked_dirs_do_not_repeat_runtime_dir_under_run(monkeypatch, tmp_path):
    run = tmp_path / "run"
    (run / "user" / "1000").mkdir(parents=True)
    monkeypatch.setattr(isolation_mod, "_BASE_MASKS", ("/tmp", str(run)))
    monkeypatch.setattr(isolation_mod, "_VAR_RUN", tmp_path / "missing")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(run / "user" / "1000"))
    assert isolation_mod.masked_dirs() == ("/tmp", str(run))


def test_masked_dirs_ignore_missing_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation_mod, "_VAR_RUN", tmp_path / "missing")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "gone"))
    assert isolation_mod.masked_dirs() == ("/tmp", "/run")


def test_escaping_symlinks_keep_only_links_leaving_the_masks(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "current-system").symlink_to("/nix/store/abc-system")
    (run / "inner").symlink_to("sub/dir")  # stays inside the mask
    (run / "sibling").symlink_to(tmp_path / "other" / "x")
    (run / "docker.sock").write_text("")  # not a link
    (run / "sub").mkdir()
    other = tmp_path / "other"

    links = isolation_mod.escaping_symlinks([str(run), str(other)])

    assert links == [("/nix/store/abc-system", str(run / "current-system"))]


def test_escaping_symlinks_skip_tmp_and_unreadable_masks(tmp_path):
    assert isolation_mod.escaping_symlinks(["/tmp", str(tmp_path / "missing")]) == []


def test_bubblewrap_wrap_uses_host_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation_mod.shutil, "which", _fake_which({"bwrap": BWRAP}))
    monkeypatch.setattr(isolation_mod, "masked_dirs", lambda: ("/tmp", "/run", "/x"))
    monkeypatch.setattr(
        isolation_mod,
        "escaping_symlinks",
        lambda masks: [("/nix/store/sys", "/run/current-system")],
    )
    backend = BubblewrapBackend(expose=[Path("/tmp/venv")])
    wrapped = backend.wrap(["true"], _spec(tmp_path))

    assert _operands(wrapped, "--tmpfs") == ["/tmp", "/run", "/x"]
    assert _pairs(wrapped, "--symlink") == [("/nix/store/sys", "/run/current-system")]
    assert _pairs(wrapped, "--ro-bind-try") == [("/tmp/venv", "/tmp/venv")]


def test_bubblewrap_wrap_places_command_after_separator(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation_mod.shutil, "which", _fake_which({"bwrap": BWRAP}))
    monkeypatch.setattr(isolation_mod, "masked_dirs", lambda: ("/tmp", "/run"))
    monkeypatch.setattr(isolation_mod, "escaping_symlinks", lambda masks: [])
    spec = _spec(tmp_path)
    wrapped = BubblewrapBackend().wrap(["echo", "--", "hi"], spec)

    assert wrapped[0] == BWRAP
    sep = wrapped.index("--")
    assert wrapped[1:sep] == isolation_mod.bwrap_args(spec)
    assert wrapped[sep + 1 :] == ["echo", "--", "hi"]


def test_bubblewrap_explicit_executable_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation_mod.shutil, "which", _fake_which({}))
    wrapped = BubblewrapBackend("/opt/bwrap").wrap(["true"], _spec(tmp_path))
    assert wrapped[0] == "/opt/bwrap"


# --- bubblewrap availability -------------------------------------------------


def test_bubblewrap_unavailable_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(isolation_mod.shutil, "which", _fake_which({"bwrap": BWRAP}))
    assert not BubblewrapBackend().available()


def test_bubblewrap_unavailable_when_missing(monkeypatch):
    _as_linux(monkeypatch, bwrap=False)
    assert not BubblewrapBackend().available()


def test_bubblewrap_available_when_probe_succeeds(monkeypatch):
    _as_linux(monkeypatch, bwrap=True, probe_ok=True)
    assert BubblewrapBackend().available()


def test_bubblewrap_unavailable_when_probe_fails(monkeypatch):
    # bwrap installed but user namespaces blocked (common in containers).
    _as_linux(monkeypatch, bwrap=True, probe_ok=False)
    assert not BubblewrapBackend().available()


def test_bwrap_probe_runs_once_and_reports_status(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(isolation_mod.subprocess, "run", fake_run)
    assert isolation_mod._bwrap_works(BWRAP) is False
    assert isolation_mod._bwrap_works(BWRAP) is False
    assert len(calls) == 1
    assert calls[0][0] == BWRAP
    # The probe exercises the same layout real commands get.
    for flag in ("--unshare-net", "--unshare-ipc", "--tmpfs", "--bind"):
        assert flag in calls[0]


def test_bwrap_probe_treats_os_error_as_unavailable(monkeypatch):
    def boom(argv, **kwargs):
        raise OSError("exec format error")

    monkeypatch.setattr(isolation_mod.subprocess, "run", boom)
    assert isolation_mod._bwrap_works(BWRAP) is False


def test_bwrap_probe_success(monkeypatch):
    monkeypatch.setattr(
        isolation_mod.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0),
    )
    assert isolation_mod._bwrap_works(BWRAP) is True


# --- sandbox-exec backend ----------------------------------------------------


def test_sandbox_exec_available_only_on_darwin(monkeypatch):
    monkeypatch.setattr(
        isolation_mod.shutil,
        "which",
        _fake_which({"sandbox-exec": "/usr/bin/sandbox-exec"}),
    )
    monkeypatch.setattr(sys, "platform", "linux")
    assert not SandboxExecBackend().available()
    monkeypatch.setattr(sys, "platform", "darwin")
    assert SandboxExecBackend().available()


def test_sandbox_exec_wrap_uses_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        isolation_mod.shutil,
        "which",
        _fake_which({"sandbox-exec": "/usr/bin/sandbox-exec"}),
    )
    spec = _spec(tmp_path)
    backend = SandboxExecBackend()
    wrapped = backend.wrap(["echo", "hi"], spec)
    profile = backend.profile(spec)
    assert wrapped == ["/usr/bin/sandbox-exec", "-p", profile, "echo", "hi"]
    assert "(deny network*)" in backend.profile(spec)
    assert "(deny network*)" not in backend.profile(_spec(tmp_path, allow_network=True))


# --- backend selection -------------------------------------------------------


def test_auto_selects_bubblewrap_on_linux(monkeypatch):
    _as_linux(monkeypatch, bwrap=True)
    backend = select_backend("auto")
    assert isinstance(backend, BubblewrapBackend)
    assert backend.enforced


def test_auto_selects_sandbox_exec_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        isolation_mod.shutil,
        "which",
        _fake_which({"sandbox-exec": "/usr/bin/sandbox-exec", "bwrap": BWRAP}),
    )
    assert isinstance(select_backend(), SandboxExecBackend)


def test_auto_degrades_to_none_without_backend(monkeypatch):
    _as_linux(monkeypatch, bwrap=False)
    backend = select_backend("auto")
    assert isinstance(backend, NoIsolation)
    assert not backend.enforced


def test_auto_require_raises_without_backend(monkeypatch):
    _as_linux(monkeypatch, bwrap=False)
    with pytest.raises(IsolationUnavailable, match="bubblewrap"):
        select_backend("auto", require=True)


def test_auto_degrades_on_unsupported_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(isolation_mod.shutil, "which", _fake_which({}))
    assert isinstance(select_backend(), NoIsolation)


def test_named_backend_unavailable_raises(monkeypatch):
    _as_linux(monkeypatch, bwrap=False)
    with pytest.raises(IsolationUnavailable):
        select_backend("bubblewrap")


def test_named_backend_selected(monkeypatch):
    _as_linux(monkeypatch, bwrap=True)
    assert isinstance(select_backend("bubblewrap"), BubblewrapBackend)
    assert isinstance(select_backend("none"), NoIsolation)


def test_unknown_backend_name_raises():
    with pytest.raises(ValueError, match="unknown isolation backend"):
        select_backend("firejail")


def test_none_with_require_is_contradictory():
    with pytest.raises(ValueError):
        select_backend("none", require=True)


class _RecordingBackend(IsolationBackend):
    name = "recording"

    def __init__(self, *, available: bool = True) -> None:
        self._available = available
        self.calls: list[tuple[list[str], IsolationSpec]] = []

    def available(self) -> bool:
        return self._available

    def wrap(self, argv: Sequence[str], spec: IsolationSpec) -> list[str]:
        self.calls.append((list(argv), spec))
        return list(argv)


def test_custom_backend_instance_is_used():
    backend = _RecordingBackend()
    assert select_backend(backend, require=True) is backend


def test_custom_backend_unavailable_raises():
    with pytest.raises(IsolationUnavailable):
        select_backend(_RecordingBackend(available=False))


# --- WorkspaceSandbox integration --------------------------------------------


def test_sandbox_reports_bubblewrap_on_linux(tmp_path, monkeypatch):
    _as_linux(monkeypatch, bwrap=True)
    sandbox = WorkspaceSandbox(tmp_path)
    assert sandbox.isolation_level == "bubblewrap"
    assert sandbox.isolation_enforced
    wrapped = sandbox._wrap(["echo", "hi"])
    assert wrapped[0] == BWRAP
    assert wrapped[-2:] == ["echo", "hi"]
    assert _pairs(wrapped, "--bind") == [
        (str(sandbox._tmpdir), str(sandbox._tmpdir)),
        (str(sandbox.root), str(sandbox.root)),
    ]
    assert "--unshare-net" in wrapped


def test_sandbox_allow_network_skips_unshare_net(tmp_path, monkeypatch):
    _as_linux(monkeypatch, bwrap=True)
    sandbox = WorkspaceSandbox(tmp_path, allow_network=True)
    assert "--unshare-net" not in sandbox._wrap(["echo"])


def test_sandbox_degradation_is_observable_and_warns_once(
    tmp_path, monkeypatch, fresh_warning
):
    _as_linux(monkeypatch, bwrap=False)
    with pytest.warns(IsolationWarning, match="NOT enforced"):
        first = WorkspaceSandbox(tmp_path / "a")
    assert first.isolation_level == "none"
    assert not first.isolation_enforced
    assert first._wrap(["echo", "hi"]) == ["echo", "hi"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        WorkspaceSandbox(tmp_path / "b")  # second degraded sandbox stays quiet


def test_sandbox_explicit_none_does_not_warn(tmp_path, fresh_warning):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sandbox = WorkspaceSandbox(tmp_path, isolation="none")
    assert not sandbox.isolation_enforced


def test_sandbox_require_isolation_raises_before_creating_root(tmp_path, monkeypatch):
    _as_linux(monkeypatch, bwrap=False)
    root = tmp_path / "workspace"
    with pytest.raises(IsolationUnavailable):
        WorkspaceSandbox(root, require_isolation=True)
    assert not root.exists()


def test_sandbox_require_isolation_accepts_enforcing_backend(tmp_path, monkeypatch):
    _as_linux(monkeypatch, bwrap=True)
    sandbox = WorkspaceSandbox(tmp_path, require_isolation=True)
    assert sandbox.isolation_enforced


def test_sandbox_runs_commands_through_custom_backend(tmp_path):
    backend = _RecordingBackend()
    sandbox = WorkspaceSandbox(tmp_path, isolation=backend)
    assert sandbox.isolation is backend
    assert sandbox.isolation_level == "recording"
    result = sandbox.run_command([sys.executable, "-c", "print('hi')"], timeout=30)
    assert result.returncode == 0
    (argv, spec), *_ = backend.calls
    assert argv[0] == sys.executable
    assert spec == IsolationSpec(
        root=sandbox.root, tmpdir=sandbox._tmpdir, allow_network=False
    )


def test_sandbox_recreates_scratch_dir_before_wrapping(tmp_path):
    # bwrap binds $TMPDIR, so a missing scratch dir would fail every command.
    import asyncio

    backend = _RecordingBackend()
    sandbox = WorkspaceSandbox(tmp_path, isolation=backend)
    asyncio.run(sandbox.stop())
    assert not sandbox._tmpdir.exists()
    result = sandbox.run_command([sys.executable, "-c", "pass"], timeout=30)
    assert result.returncode == 0
    assert sandbox._tmpdir.is_dir()
    asyncio.run(sandbox.stop())


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec only")
def test_sandbox_defaults_to_sandbox_exec_on_macos(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    assert sandbox.isolation_level == "sandbox-exec"
    assert sandbox.isolation_enforced


# --- real bubblewrap (Linux with a working bwrap only) -----------------------

# CI sets this on its bubblewrap job so a broken bwrap fails instead of skipping.
_REQUIRE_BWRAP = os.environ.get("NEXUS_TEST_REQUIRE_BWRAP") == "1"

_needs_bwrap = pytest.mark.skipif(
    not _REQUIRE_BWRAP and not BubblewrapBackend().available(),
    reason="needs a working bwrap on Linux",
)


@pytest.fixture
def host_dir():
    # A host dir outside /tmp and /run, so only the read-only bind can stop writes.
    base = Path.home()
    if any(base == Path(m) or Path(m) in base.parents for m in ("/tmp", "/run")):
        pytest.skip("home directory lives under a masked directory")
    path = Path(tempfile.mkdtemp(prefix=".nexus-bwrap-test-", dir=base))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _bwrap_sandbox(root: Path, **kwargs) -> WorkspaceSandbox:
    return WorkspaceSandbox(root, isolation="bubblewrap", **kwargs)


@_needs_bwrap
def test_bwrap_confines_writes_to_read_only_host(tmp_path, host_dir):
    sandbox = _bwrap_sandbox(tmp_path / "ws")
    outside = host_dir / "escape_target.txt"
    inside = sandbox.root / "inside.txt"
    script = (
        f"open({str(inside)!r}, 'w').write('ok'); "
        f"open({str(outside)!r}, 'w').write('x')"
    )
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    assert result.returncode != 0
    assert "Read-only file system" in result.stderr
    assert inside.read_text() == "ok"
    assert not outside.exists()


@_needs_bwrap
def test_bwrap_reads_host_outside_masks(tmp_path, host_dir):
    (host_dir / "data.txt").write_text("visible")
    sandbox = _bwrap_sandbox(tmp_path / "ws")
    result = sandbox.run_command(["cat", str(host_dir / "data.txt")], timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "visible"


@_needs_bwrap
def test_bwrap_masks_host_tmp(tmp_path):
    # tmp_path is on the host /tmp but outside the root: masked, not read-only.
    if not str(tmp_path).startswith("/tmp/"):
        pytest.skip("pytest's tmp_path is not under /tmp on this host")
    secret = tmp_path / "secret.txt"
    secret.write_text("host")
    sandbox = _bwrap_sandbox(tmp_path / "ws")
    script = (
        "import os\n"
        f"print(os.path.exists({str(secret)!r}))\n"
        f"open({str(tmp_path / 'dropped.txt')!r}, 'w').write('x')\n"
    )
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
    assert not (tmp_path / "dropped.txt").exists()


@_needs_bwrap
def test_bwrap_allows_tmpdir_writes(tmp_path):
    sandbox = _bwrap_sandbox(tmp_path)
    script = (
        "import os, tempfile; "
        "tempfile.NamedTemporaryFile(dir=os.environ['TMPDIR'], delete=False)"
        ".write(b'x'); "
        "open('/tmp/nexus-sandbox-private-probe', 'w').write('y'); print('ok')"
    )
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    assert result.returncode == 0, result.stderr
    assert not Path("/tmp/nexus-sandbox-private-probe").exists()
    assert any(sandbox._tmpdir.iterdir())


@_needs_bwrap
def test_bwrap_expose_reopens_a_masked_path(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "data.txt").write_text("exposed")
    backend = BubblewrapBackend(expose=[shared])
    sandbox = WorkspaceSandbox(tmp_path / "ws", isolation=backend)
    result = sandbox.run_command(["cat", str(shared / "data.txt")], timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "exposed"


_CONNECT_SCRIPT = (
    "import socket, sys\n"
    "s = socket.socket(socket.AF_UNIX)\n"
    "try:\n"
    "    s.connect(sys.argv[1])\n"
    "except OSError as exc:\n"
    "    print('blocked', exc)\n"
    "else:\n"
    "    print('connected')\n"
)


def _socket_dirs(tmp_path: Path) -> list[Path]:
    dirs = [tmp_path]
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and os.access(runtime, os.W_OK):
        dirs.append(Path(runtime))
    return dirs


@_needs_bwrap
def test_bwrap_hides_host_unix_sockets(tmp_path):
    for directory in _socket_dirs(tmp_path):
        path = Path(tempfile.mkdtemp(prefix="nx-", dir=directory)) / "s"
        server = socket.socket(socket.AF_UNIX)
        try:
            server.bind(str(path))
            server.listen(1)
            argv = [sys.executable, "-c", _CONNECT_SCRIPT, str(path)]
            # Control: the socket is reachable without isolation.
            plain = WorkspaceSandbox(tmp_path / "plain", isolation="none")
            assert plain.run_command(argv, timeout=30).stdout.startswith("connected")
            sandbox = _bwrap_sandbox(tmp_path / "ws")
            result = sandbox.run_command(argv, timeout=30)
            assert result.stdout.startswith("blocked"), (directory, result)
        finally:
            server.close()
            shutil.rmtree(path.parent, ignore_errors=True)


@_needs_bwrap
def test_bwrap_denies_network(tmp_path):
    sandbox = _bwrap_sandbox(tmp_path)
    script = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "except OSError:\n"
        "    print('blocked')\n"
    )
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    assert "blocked" in result.stdout
