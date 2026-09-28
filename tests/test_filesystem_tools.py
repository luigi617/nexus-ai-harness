from __future__ import annotations

import io
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
from nexus_ai_harness.plugins.tools import (
    EditFile,
    Grep,
    ListDir,
    ReadFile,
    WriteFile,
    filesystem,
)
from nexus_ai_harness.plugins.tools._text import iter_lines, split_lines
from tests.conftest import make_ctx


def test_write_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = make_ctx()
    expected = (tmp_path / "notes.txt").resolve()

    result = WriteFile().run({"path": "notes.txt", "content": "hello"}, ctx)
    assert result == f"wrote 5 bytes to {expected}"
    assert ReadFile().run({"path": "notes.txt"}, ctx) == "     1\thello"
    raw = ReadFile().run({"path": "notes.txt", "line_numbers": False}, ctx)
    assert raw == "hello"


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


def test_list_dir_truncates_between_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for i in range(3000):
        (tmp_path / f"file_{i:05d}.txt").write_text("")
    ctx = make_ctx()

    listing = ListDir().run({}, ctx)
    lines = listing.splitlines()
    assert all(line.startswith("file_") for line in lines[:-1])
    assert lines[-1].startswith("... (") and "more entries not shown" in lines[-1]
    assert len(listing) <= 10_100


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
    assert ReadFile().run({"path": "inside.txt"}, ctx) == "     1\tok"


def test_path_escape_with_sandbox_returns_error(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    ctx = make_ctx(sandbox)

    result = ReadFile().run({"path": "../escape.txt"}, ctx)
    assert result.startswith("error:")

    written = WriteFile().run({"path": "../../evil.txt", "content": "x"}, ctx)
    assert written.startswith("error:")


def test_read_numbers_lines_like_cat_n(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.py").write_text("a\n\n  b\r\nc")
    ctx = make_ctx()

    result = ReadFile().run({"path": "f.py"}, ctx)
    assert result == "     1\ta\n     2\t\n     3\t  b\n     4\tc"


def test_read_offset_and_limit_select_a_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    ctx = make_ctx()

    result = ReadFile().run({"path": "f.txt", "offset": 4, "limit": 3}, ctx)
    assert result.splitlines() == [
        "     4\tline4",
        "     5\tline5",
        "     6\tline6",
        "... (showing lines 4-6 of 10; call read_file with offset=7 to read more)",
    ]
    tail = ReadFile().run({"path": "f.txt", "offset": 9}, ctx)
    assert tail == "     9\tline9\n    10\tline10"


def test_read_offset_past_end_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    ctx = make_ctx()

    result = ReadFile().run({"path": "f.txt", "offset": 5}, ctx)
    assert result == "error: offset 5 is past the end of the file (2 lines)"


def test_read_bad_offset_and_limit_fall_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    ctx = make_ctx()

    for bad in (0, -3, "2", True, float("nan"), None):
        args = {"path": "f.txt", "offset": bad, "limit": bad}
        assert ReadFile().run(args, ctx) == "     1\tone\n     2\ttwo"


def test_read_empty_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "empty.txt").write_text("")
    ctx = make_ctx()

    assert ReadFile().run({"path": "empty.txt"}, ctx) == "(empty file)"


def test_read_default_line_cap_tells_how_to_continue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("".join(f"{i}\n" for i in range(1, 5001)))
    ctx = make_ctx()

    result = ReadFile().run({"path": "big.txt"}, ctx)
    lines = result.splitlines()
    assert lines[-2] == "  2000\t2000"
    assert lines[-1] == (
        "... (showing lines 1-2000 of 5000; call read_file with offset=2001 "
        "to read more)"
    )


def test_read_char_cap_stops_on_a_line_boundary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wide.txt").write_text(("x" * 1000 + "\n") * 200)
    ctx = make_ctx()

    result = ReadFile().run({"path": "wide.txt"}, ctx)
    lines = result.splitlines()
    assert all(line.endswith("x" * 1000) for line in lines[:-1])
    assert "call read_file with offset=" in lines[-1]
    assert len(result) <= 51_000


def test_read_long_line_is_clipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("a" * 20_000)
    ctx = make_ctx()

    result = ReadFile().run({"path": "big.txt"}, ctx)
    assert result.endswith("... (line truncated)")
    assert len(result) < 2100


def test_read_truncation_never_splits_multibyte_chars(tmp_path, monkeypatch):
    # The old byte-based cut could split a UTF-8 sequence into U+FFFD garbage.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cjk.txt").write_text("漢" * 8000, encoding="utf-8")
    ctx = make_ctx()

    result = ReadFile().run({"path": "cjk.txt"}, ctx)
    assert result.endswith("... (line truncated)")
    assert "�" not in result
    assert result.count("漢") == 2000


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


# --- edit_file ------------------------------------------------------------


def _edit(tmp_path, monkeypatch, content: str, **arguments) -> tuple[str, str]:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "app.py"
    target.write_bytes(content.encode("utf-8"))
    result = EditFile().run({"path": "app.py", **arguments}, make_ctx())
    return result, target.read_bytes().decode("utf-8")


def test_edit_replaces_a_unique_string(tmp_path, monkeypatch):
    result, content = _edit(
        tmp_path,
        monkeypatch,
        "def add(a, b):\n    return a - b\n",
        old_string="return a - b",
        new_string="return a + b",
    )
    assert content == "def add(a, b):\n    return a + b\n"
    assert result.splitlines() == [
        "edited app.py: 1 replacement",
        "     1\tdef add(a, b):",
        "     2\t    return a + b",
    ]


def test_edit_not_found_leaves_file_untouched(tmp_path, monkeypatch):
    result, content = _edit(
        tmp_path, monkeypatch, "x = 1\n", old_string="y = 1", new_string="y = 2"
    )
    assert result.startswith("error: old_string not found in app.py")
    assert content == "x = 1\n"


def test_edit_ambiguous_match_is_refused(tmp_path, monkeypatch):
    result, content = _edit(
        tmp_path, monkeypatch, "x = 1\nx = 1\n", old_string="x = 1", new_string="x=2"
    )
    assert result.startswith("error: old_string occurs 2 times in app.py")
    assert "replace_all" in result
    assert content == "x = 1\nx = 1\n"


def test_edit_replace_all(tmp_path, monkeypatch):
    result, content = _edit(
        tmp_path,
        monkeypatch,
        "x = 1\nx = 1\n",
        old_string="x = 1",
        new_string="x = 2",
        replace_all=True,
    )
    assert result.startswith("edited app.py: 2 replacements")
    assert content == "x = 2\nx = 2\n"


def test_edit_multi_edit_applies_in_order(tmp_path, monkeypatch):
    result, content = _edit(
        tmp_path,
        monkeypatch,
        "alpha\nbeta\n",
        edits=[
            {"old_string": "alpha", "new_string": "gamma"},
            {"old_string": "gamma\nbeta", "new_string": "gamma\ndelta"},
        ],
    )
    assert result == "edited app.py: 2 edits, 2 replacements"
    assert content == "gamma\ndelta\n"


def test_edit_multi_edit_is_atomic(tmp_path, monkeypatch):
    result, content = _edit(
        tmp_path,
        monkeypatch,
        "alpha\nbeta\n",
        edits=[
            {"old_string": "alpha", "new_string": "gamma"},
            {"old_string": "missing", "new_string": "x"},
        ],
    )
    assert result.startswith("error: edit 2: old_string not found")
    assert content == "alpha\nbeta\n"


def test_edit_rejects_bad_requests(tmp_path, monkeypatch):
    cases = [
        ({"old_string": "", "new_string": "x"}, "old_string must not be empty"),
        ({"old_string": "a", "new_string": "a"}, "are identical"),
        ({"old_string": "a"}, "old_string and new_string are required"),
        ({"edits": []}, "edits must be a non-empty list"),
        ({"edits": ["a"]}, "edit 1: each edit must be an object"),
        (
            {"old_string": "a", "new_string": "b", "edits": [{}]},
            "not both",
        ),
    ]
    for arguments, message in cases:
        result, content = _edit(tmp_path, monkeypatch, "abc\n", **arguments)
        assert result.startswith("error:") and message in result, result
        assert content == "abc\n"


def test_edit_matches_lf_text_against_crlf_file(tmp_path, monkeypatch):
    result, content = _edit(
        tmp_path,
        monkeypatch,
        "one\r\ntwo\r\nthree\r\n",
        old_string="one\ntwo",
        new_string="uno\ndos",
    )
    assert result.startswith("edited app.py: 1 replacement")
    assert content == "uno\r\ndos\r\nthree\r\n"


def test_edit_non_utf8_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfeabc")
    result = EditFile().run(
        {"path": "blob.bin", "old_string": "abc", "new_string": "x"}, make_ctx()
    )
    assert result == "error: blob.bin is not valid UTF-8 text"
    assert (tmp_path / "blob.bin").read_bytes() == b"\xff\xfeabc"


def test_edit_missing_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = EditFile().run(
        {"path": "nope.py", "old_string": "a", "new_string": "b"}, make_ctx()
    )
    assert result.startswith("error:")
    assert not (tmp_path / "nope.py").exists()


def test_edit_preserves_permissions_and_leaves_no_temp_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "run.sh"
    script.write_text("echo old\n")
    os.chmod(script, 0o751)
    result = EditFile().run(
        {"path": "run.sh", "old_string": "old", "new_string": "new"}, make_ctx()
    )
    assert result.startswith("edited run.sh")
    assert stat.S_IMODE(script.stat().st_mode) == 0o751
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.sh"]


def test_edit_is_confined_to_the_sandbox(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path / "root")
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n")
    ctx = make_ctx(sandbox)

    result = EditFile().run(
        {"path": "../outside.py", "old_string": "1", "new_string": "2"}, ctx
    )
    assert result.startswith("error: path escapes sandbox root")
    assert outside.read_text() == "secret = 1\n"

    (Path(sandbox.root) / "pkg").mkdir()
    (Path(sandbox.root) / "pkg" / "m.py").write_text("v = 1\n")
    result = EditFile().run(
        {"path": "pkg/m.py", "old_string": "v = 1", "new_string": "v = 2"}, ctx
    )
    assert result.startswith("edited pkg/m.py: 1 replacement")


# --- review regressions ------------------------------------------------------


def test_read_file_and_grep_agree_on_line_numbers_with_form_feeds(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n\x0c\ndef f():\n    return 2\n")
    ctx = make_ctx(WorkspaceSandbox(tmp_path))
    assert Grep().run({"pattern": "return 2"}, ctx) == "m.py:4:    return 2"
    shown = ReadFile().run({"path": "m.py", "offset": 4, "limit": 1}, ctx)
    assert shown.startswith("     4\t    return 2")


def test_edit_snippet_numbers_match_read_file_with_form_feeds(tmp_path):
    (tmp_path / "m.py").write_text("a\n\x0c\nb\nc\n")
    ctx = make_ctx(WorkspaceSandbox(tmp_path))
    result = EditFile().run({"path": "m.py", "old_string": "c", "new_string": "C"}, ctx)
    assert "     4\tC" in result.splitlines()


def test_read_file_clips_a_huge_line_without_breaking_numbering(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 300_000 + "\r\nnext\n")
    ctx = make_ctx(WorkspaceSandbox(tmp_path))
    lines = ReadFile().run({"path": "big.txt"}, ctx).splitlines()
    assert lines[0].endswith("... (line truncated)")
    assert len(lines[0]) < 2100
    assert lines[1] == "     2\tnext"


def test_iter_lines_handles_crlf_split_across_chunks():
    text = "a" * (65_536 - 1) + "\r\nb\rc\n\nd"
    assert [line for line, _ in iter_lines(io.StringIO(text, newline=""), 5)] == [
        "aaaaa",
        "b",
        "c",
        "",
        "d",
    ]
    assert split_lines("a\x0cb\r\nc\rd\n") == ["a\x0cb", "c", "d"]


def test_read_file_stops_early_on_a_huge_file(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem, "_MAX_COUNT_BYTES", 10)
    (tmp_path / "f.txt").write_text("".join(f"{n}\n" for n in range(1, 100)))
    ctx = make_ctx(WorkspaceSandbox(tmp_path))
    result = ReadFile().run({"path": "f.txt", "limit": 2}, ctx)
    assert result.splitlines()[-1] == (
        "... (showing lines 1-2; the file has more; call read_file with "
        "offset=3 to read more)"
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_read_and_edit_refuse_a_fifo(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    ctx = make_ctx(WorkspaceSandbox(tmp_path))
    assert ReadFile().run({"path": "pipe"}, ctx).startswith("error: not a regular")
    edit = {"path": "pipe", "old_string": "a", "new_string": "b"}
    assert EditFile().run(edit, ctx).startswith("error: not a regular")


def test_concurrent_edits_to_one_file_all_land(tmp_path):
    names = [f"v{n}" for n in range(16)]
    target = tmp_path / "m.py"
    target.write_text("".join(f"{name} = 0\n" for name in names))
    ctx = make_ctx(WorkspaceSandbox(tmp_path))

    def edit(name: str) -> str:
        args = {
            "path": "m.py",
            "old_string": f"{name} = 0",
            "new_string": f"{name} = 1",
        }
        return EditFile().run(args, ctx)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(edit, names))
    assert all(r.startswith("edited m.py: 1 replacement") for r in results)
    assert target.read_text() == "".join(f"{name} = 1\n" for name in names)
