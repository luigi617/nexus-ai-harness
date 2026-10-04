from __future__ import annotations

import os
from pathlib import Path

import pytest

from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
from nexus_ai_harness.plugins.tools import Glob, Grep
from nexus_ai_harness.plugins.tools.search import compile_glob
from tests.conftest import make_ctx


@pytest.fixture
def repo(tmp_path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "pkg" / "core.py").write_text(
        "import os\n\ndef add(a, b):\n    return a + b\n\nclass Adder:\n    pass\n"
    )
    (root / "src" / "pkg" / "util.pyi").write_text("def add(a: int) -> int: ...\n")
    (root / "tests" / "test_core.py").write_text("from pkg.core import add\n")
    (root / "README.md").write_text("Add numbers.\n")
    (root / "logo.png").write_bytes(b"\x89PNG\x00\x00add\x00")
    for junk in (".git", "node_modules", "__pycache__", ".venv"):
        (root / junk).mkdir()
        (root / junk / "hidden.py").write_text("def add(): pass\n")
    return root


def sandboxed(root: Path):
    return make_ctx(WorkspaceSandbox(root))


# --- grep -----------------------------------------------------------------


def test_grep_reports_path_line_text(repo):
    result = Grep().run({"pattern": r"def add"}, sandboxed(repo))
    assert result.splitlines() == [
        "src/pkg/core.py:3:def add(a, b):",
        "src/pkg/util.pyi:1:def add(a: int) -> int: ...",
    ]


def test_grep_skips_binary_files_and_junk_dirs(repo):
    result = Grep().run({"pattern": "add"}, sandboxed(repo))
    assert "logo.png" not in result
    for junk in (".git", "node_modules", "__pycache__", ".venv"):
        assert junk not in result


def test_grep_glob_filters_by_name_or_path(repo):
    ctx = sandboxed(repo)
    by_name = Grep().run({"pattern": "add", "glob": "*.py"}, ctx)
    assert {line.split(":")[0] for line in by_name.splitlines()} == {
        "src/pkg/core.py",
        "tests/test_core.py",
    }
    by_path = Grep().run({"pattern": "add", "glob": "src/**/*.{py,pyi}"}, ctx)
    assert {line.split(":")[0] for line in by_path.splitlines()} == {
        "src/pkg/core.py",
        "src/pkg/util.pyi",
    }


def test_grep_glob_naming_a_junk_dir_searches_it(repo):
    result = Grep().run(
        {"pattern": "add", "glob": "node_modules/**/*.py"}, sandboxed(repo)
    )
    assert result == "node_modules/hidden.py:1:def add(): pass"


def test_grep_glob_substring_of_junk_dir_name_does_not_unskip_it(repo):
    (repo / ".github").mkdir()
    (repo / ".github" / "workflow.py").write_text("def add(): pass\n")
    result = Grep().run({"pattern": "add", "glob": "**/.github/*.py"}, sandboxed(repo))
    assert result == ".github/workflow.py:1:def add(): pass"
    assert ".git/hidden.py" not in result


def test_grep_case_insensitive_and_path(repo):
    ctx = sandboxed(repo)
    assert Grep().run({"pattern": "^add", "path": "README.md"}, ctx) == "no matches"
    result = Grep().run(
        {"pattern": "^add", "path": "README.md", "case_insensitive": True}, ctx
    )
    assert result == "README.md:1:Add numbers."
    scoped = Grep().run({"pattern": "add", "path": "tests"}, ctx)
    assert scoped == "tests/test_core.py:1:from pkg.core import add"


def test_grep_context_lines(repo):
    result = Grep().run(
        {"pattern": "return", "path": "src/pkg/core.py", "context": 1},
        sandboxed(repo),
    )
    assert result.splitlines() == [
        "src/pkg/core.py-3-def add(a, b):",
        "src/pkg/core.py:4:    return a + b",
        "src/pkg/core.py-5-",
    ]


def test_grep_files_only(repo):
    result = Grep().run({"pattern": "add", "files_only": True}, sandboxed(repo))
    assert result.splitlines() == [
        "src/pkg/core.py",
        "src/pkg/util.pyi",
        "tests/test_core.py",
    ]


def test_grep_max_results_caps_output(repo):
    result = Grep().run(
        {"pattern": ".", "path": "src/pkg/core.py", "max_results": 2}, sandboxed(repo)
    )
    lines = result.splitlines()
    assert lines[:2] == [
        "src/pkg/core.py:1:import os",
        "src/pkg/core.py:3:def add(a, b):",
    ]
    assert lines[2].startswith("... (stopped at max_results=2 matches")


def test_grep_errors(repo):
    ctx = sandboxed(repo)
    assert Grep().run({"pattern": "("}, ctx).startswith("error: invalid regular")
    assert Grep().run({"pattern": ""}, ctx) == "error: pattern is required"
    assert Grep().run({"pattern": "x", "path": "nope"}, ctx).startswith("error:")
    escaped = Grep().run({"pattern": "x", "path": "../"}, ctx)
    assert escaped.startswith("error: path escapes sandbox root")


def test_grep_ignores_symlinks_leaving_the_root(repo, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("add the password\n")
    (repo / "leak.txt").symlink_to(secret)
    (repo / "leakdir").symlink_to(tmp_path)
    result = Grep().run({"pattern": "password"}, sandboxed(repo))
    assert result == "no matches"


def test_grep_without_sandbox_uses_cwd(repo, monkeypatch):
    monkeypatch.chdir(repo)
    result = Grep().run({"pattern": "import add"}, make_ctx())
    assert result == "tests/test_core.py:1:from pkg.core import add"


# --- glob -----------------------------------------------------------------


def test_glob_recursive_sorted(repo):
    result = Glob().run({"pattern": "**/*.py"}, sandboxed(repo))
    assert result.splitlines() == ["src/pkg/core.py", "tests/test_core.py"]


def test_glob_star_stays_in_one_segment(repo):
    ctx = sandboxed(repo)
    assert Glob().run({"pattern": "*.md"}, ctx) == "README.md"
    assert Glob().run({"pattern": "*.py"}, ctx) == "no files match"
    assert Glob().run({"pattern": "src/*/core.py"}, ctx) == "src/pkg/core.py"


def test_glob_braces_classes_and_base_path(repo):
    ctx = sandboxed(repo)
    result = Glob().run({"pattern": "**/*.{py,pyi}", "path": "src"}, ctx)
    assert result.splitlines() == ["src/pkg/core.py", "src/pkg/util.pyi"]
    assert Glob().run({"pattern": "tests/test_[a-c]*.py"}, ctx) == "tests/test_core.py"


def test_glob_skips_junk_dirs_unless_named(repo):
    ctx = sandboxed(repo)
    assert "hidden.py" not in Glob().run({"pattern": "**/hidden.py"}, ctx)
    assert Glob().run({"pattern": ".venv/*.py"}, ctx) == ".venv/hidden.py"


def test_glob_caps_results(repo):
    for i in range(10):
        (repo / f"f{i}.txt").write_text("")
    result = Glob().run({"pattern": "*.txt", "max_results": 3}, sandboxed(repo))
    assert result.splitlines() == [
        "f0.txt",
        "f1.txt",
        "f2.txt",
        (
            "... (stopped at max_results=3 files; there may be more: narrow "
            "the pattern or raise max_results)"
        ),
    ]


def test_glob_caps_results_without_walking_whole_tree(repo, monkeypatch):
    for i in range(4):
        (repo / f"f{i}.txt").write_text("")
    trap = repo / "zzz_trap"
    trap.mkdir()
    (trap / "late.txt").write_text("")

    scanned: list[Path] = []
    original_scandir = os.scandir

    def tracking_scandir(path):
        scanned.append(Path(path))
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", tracking_scandir)
    result = Glob().run({"pattern": "*.txt", "max_results": 3}, sandboxed(repo))
    assert "late.txt" not in result
    assert trap not in scanned


def test_glob_errors(repo):
    ctx = sandboxed(repo)
    assert Glob().run({"pattern": ""}, ctx) == "error: pattern is required"
    assert (
        Glob()
        .run({"pattern": "*", "path": "README.md"}, ctx)
        .startswith("error: not a directory")
    )
    assert Glob().run({"pattern": "*", "path": "/"}, ctx).startswith("error:")


@pytest.mark.parametrize(
    ("pattern", "path", "matches"),
    [
        ("**/*.py", "a.py", True),
        ("**/*.py", "a/b/c.py", True),
        ("*.py", "a/b.py", False),
        ("a/**", "a/b/c", True),
        ("a/**/c", "a/c", True),
        ("?.txt", "ab.txt", False),
        ("[!a]*", "bcd", True),
        ("[!a]*", "abc", False),
        ("*.{js,ts}", "x.ts", True),
        ("{src,lib}/**/*.{c,h}", "lib/x/y.h", True),
        ("a+b.txt", "a+b.txt", True),
    ],
)
def test_compile_glob(pattern, path, matches):
    assert (compile_glob(pattern).fullmatch(path) is not None) is matches


# --- review regressions ------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_grep_and_glob_skip_fifos(repo):
    os.mkfifo(repo / "src" / "pipe")
    ctx = sandboxed(repo)
    assert "core.py" in Grep().run({"pattern": "add", "path": "src"}, ctx)
    assert "pipe" not in Glob().run({"pattern": "src/*"}, ctx)
    assert Grep().run({"pattern": "x", "path": "src/pipe"}, ctx).startswith("error:")


@pytest.mark.parametrize("tool", [Grep, Glob])
def test_invalid_glob_class_is_a_clean_error(repo, tool):
    args = (
        {"pattern": "[z-a].py"} if tool is Glob else {"pattern": "x", "glob": "[z-a]"}
    )
    assert tool().run(args, sandboxed(repo)).startswith("error: invalid glob")


def test_leading_dot_slash_in_globs(repo):
    ctx = sandboxed(repo)
    assert Glob().run({"pattern": "./src/pkg/*.py"}, ctx) == "src/pkg/core.py"
    result = Grep().run({"pattern": "import", "glob": "./tests/*.py"}, ctx)
    assert result == "tests/test_core.py:1:from pkg.core import add"
