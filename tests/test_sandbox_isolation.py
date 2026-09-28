from __future__ import annotations

import subprocess
import sys
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
    assert args[args.index("--tmpfs") + 1] == "/tmp"
    assert args[args.index("--chdir") + 1] == root
    for flag in ("--die-with-parent", "--new-session", "--unshare-pid"):
        assert flag in args


def test_bwrap_args_private_tmp_precedes_writable_binds(tmp_path):
    # Later mounts shadow earlier ones; a root under /tmp must stay visible.
    args = isolation_mod.bwrap_args(_spec(tmp_path))
    bind_positions = [i for i, a in enumerate(args) if a == "--bind"]
    assert args.index("--ro-bind") < args.index("--tmpfs") < min(bind_positions)


def test_bwrap_args_unshares_network_only_when_denied(tmp_path):
    assert "--unshare-net" in isolation_mod.bwrap_args(_spec(tmp_path))
    allowed = isolation_mod.bwrap_args(_spec(tmp_path, allow_network=True))
    assert "--unshare-net" not in allowed


def test_bubblewrap_wrap_places_command_after_separator(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation_mod.shutil, "which", _fake_which({"bwrap": BWRAP}))
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
    assert "--unshare-net" in calls[0]


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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec only")
def test_sandbox_defaults_to_sandbox_exec_on_macos(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    assert sandbox.isolation_level == "sandbox-exec"
    assert sandbox.isolation_enforced


# --- real bubblewrap (Linux with a working bwrap only) -----------------------

_needs_bwrap = pytest.mark.skipif(
    not BubblewrapBackend().available(), reason="needs a working bwrap on Linux"
)


@_needs_bwrap
def test_bwrap_confines_writes(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path / "ws", isolation="bubblewrap")
    outside = tmp_path / "escape_target.txt"
    inside = sandbox.root / "inside.txt"
    script = (
        f"open({str(inside)!r}, 'w').write('ok'); "
        f"open({str(outside)!r}, 'w').write('x')"
    )
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    assert result.returncode != 0
    assert inside.read_text() == "ok"
    assert not outside.exists()


@_needs_bwrap
def test_bwrap_allows_tmpdir_writes(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, isolation="bubblewrap")
    script = (
        "import os, tempfile; "
        "tempfile.NamedTemporaryFile(dir=os.environ['TMPDIR'], delete=False)"
        ".write(b'x'); "
        "open('/tmp/nexus-sandbox-private-probe', 'w').write('y'); print('ok')"
    )
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    assert result.returncode == 0, result.stderr
    assert not Path("/tmp/nexus-sandbox-private-probe").exists()


@_needs_bwrap
def test_bwrap_denies_network(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path, isolation="bubblewrap")
    script = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "except OSError:\n"
        "    print('blocked')\n"
    )
    result = sandbox.run_command([sys.executable, "-c", script], timeout=30)
    assert "blocked" in result.stdout
