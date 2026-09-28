from __future__ import annotations

from pathlib import Path

from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
from nexus_ai_harness.plugins.tools import ListDir, ReadFile, WriteFile
from tests.conftest import make_ctx


def test_write_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    expected = (tmp_path / "notes.txt").resolve()

    result = WriteFile().run({"path": "notes.txt", "content": "hello"}, ctx)
    assert result == f"wrote 5 bytes to {expected}"
    assert ReadFile().run({"path": "notes.txt"}, ctx) == "hello"


def test_write_creates_parent_dirs_within_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    result = WriteFile().run({"path": "a/b/c.txt", "content": "deep"}, ctx)
    assert result.startswith("wrote 4 bytes to")
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


def test_list_dir_marks_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "child").mkdir()
    ctx = make_ctx()

    listing = ListDir().run({}, ctx)
    assert listing == "child/\nfile.txt"


def test_read_missing_file_returns_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    assert ReadFile().run({"path": "nope.txt"}, ctx).startswith("error:")


def test_absolute_path_escape_returns_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()

    result = ReadFile().run({"path": "/etc/hosts"}, ctx)
    assert result.startswith("error:")


def test_sandbox_confines_paths(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    ctx = make_ctx(sandbox)

    WriteFile().run({"path": "inside.txt", "content": "ok"}, ctx)
    assert (Path(sandbox.root) / "inside.txt").read_text() == "ok"
    assert ReadFile().run({"path": "inside.txt"}, ctx) == "ok"


def test_path_escape_with_sandbox_returns_error(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    ctx = make_ctx(sandbox)

    result = ReadFile().run({"path": "../escape.txt"}, ctx)
    assert result.startswith("error:")

    written = WriteFile().run({"path": "../../evil.txt", "content": "x"}, ctx)
    assert written.startswith("error:")


def test_large_file_read_is_truncated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("a" * 20_000)
    ctx = make_ctx()

    result = ReadFile().run({"path": "big.txt"}, ctx)
    assert result.endswith("... (truncated)")
    assert len(result) < 20_000


def test_large_multibyte_file_gets_truncation_marker(tmp_path, monkeypatch):
    # Truncation is decided by bytes read, not decoded chars: a large 3-byte-per
    # -char file must still get the marker (regression — a char-based check
    # silently dropped the tail with no signal).
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cjk.txt").write_text("漢" * 8000, encoding="utf-8")  # 24 KB
    ctx = make_ctx()

    result = ReadFile().run({"path": "cjk.txt"}, ctx)
    assert result.endswith("... (truncated)")


def test_read_symlink_escape_returns_error(tmp_path):
    # A symlink inside the root pointing outside it must be rejected — the exact
    # case .resolve() + is_relative_to exists to catch.
    sandbox = WorkspaceSandbox(tmp_path / "root")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "s.txt").write_text("top secret")
    (Path(sandbox.root) / "link").symlink_to(tmp_path / "secret")
    ctx = make_ctx(sandbox)

    assert ReadFile().run({"path": "link/s.txt"}, ctx).startswith("error:")


def test_list_dir_on_file_returns_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    ctx = make_ctx()

    assert ListDir().run({"path": "f.txt"}, ctx).startswith("error:")


def test_read_binary_file_does_not_crash(tmp_path, monkeypatch):
    # A non-UTF-8/binary file must decode leniently and return text, not raise
    # UnicodeDecodeError (a ValueError, which the OSError handler wouldn't catch).
    monkeypatch.chdir(tmp_path)
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00\x01binary")
    ctx = make_ctx()

    result = ReadFile().run({"path": "blob.bin"}, ctx)
    assert not result.startswith("error:")
    assert "binary" in result
