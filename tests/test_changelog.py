from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "changelog", REPO / "scripts" / "changelog.py"
)
assert _spec is not None and _spec.loader is not None
changelog = importlib.util.module_from_spec(_spec)
sys.modules["changelog"] = changelog  # @dataclass needs the module resolvable
_spec.loader.exec_module(changelog)

HEADER = "# Changelog\n\nIntro.\n"


def _write(directory: Path, name: str, data: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


# --- fragment validation --------------------------------------------------


def test_load_fragment_accepts_int_and_str_ids(tmp_path):
    a = _write(tmp_path, "7.json", {"id": 7, "description": "Fix  a\nbug."})
    b = _write(tmp_path, "abc.json", {"id": "abc", "description": "Other."})
    assert changelog.load_fragment(a) == changelog.Fragment("7", "Fix a bug.")
    assert changelog.load_fragment(b) == changelog.Fragment("abc", "Other.")


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("1.json", "{not json", "invalid JSON"),
        ("1.json", ["x"], "exactly 'id' and 'description'"),
        ("1.json", {"id": 1}, "exactly 'id' and 'description'"),
        ("1.json", {"id": 1, "description": "d", "x": 1}, "exactly"),
        ("1.json", {"id": True, "description": "d"}, "string or integer"),
        ("1.json", {"id": 2, "description": "d"}, "must match file name"),
        ("1.json", {"id": 1, "description": "  "}, "non-empty string"),
    ],
)
def test_load_fragment_rejects_invalid(tmp_path, name, data, message):
    path = _write(tmp_path, name, data)
    with pytest.raises(changelog.ChangelogError, match=message):
        changelog.load_fragment(path)


def test_load_fragments_orders_numeric_then_named(tmp_path):
    for frag_id in (10, 9, "b", "a"):
        _write(tmp_path, f"{frag_id}.json", {"id": frag_id, "description": "d"})
    ids = [f.id for f in changelog.load_fragments(tmp_path)]
    assert ids == ["9", "10", "a", "b"]


def test_load_fragments_missing_directory_is_empty(tmp_path):
    assert changelog.load_fragments(tmp_path / "missing") == []


# --- rendering ------------------------------------------------------------


def test_render_links_numeric_ids_to_prs():
    assert changelog.Fragment("12", "Add X.").render() == (
        "- Add X. ([#12](https://github.com/luigi617/nexus-ai-harness/pull/12))"
    )
    assert changelog.Fragment("misc", "Tidy.").render() == "- Tidy. (misc)"


def test_insert_section_goes_above_newest_release():
    existing = HEADER + "\n## [0.1.0] - 2026-01-01\n\n- Old.\n"
    result = changelog.insert_section(existing, "## [0.2.0] - 2026-02-01\n\n- New.\n")
    assert result.index("[0.2.0]") < result.index("[0.1.0]")
    assert result.startswith(HEADER)


def test_insert_section_into_empty_changelog():
    result = changelog.insert_section(HEADER, "## [0.1.0] - d\n\n- A.\n")
    assert result == HEADER + "\n## [0.1.0] - d\n\n- A.\n"


def test_extract_section_returns_only_that_release():
    text = HEADER + "\n## [0.2.0] - d\n\n- New.\n\n## [0.1.0] - d\n\n- Old.\n"
    assert changelog.extract_section(text, "0.2.0") == "- New."
    assert changelog.extract_section(text, "0.1.0") == "- Old."
    assert changelog.extract_section(text, "0.3.0") == ""


# --- release --------------------------------------------------------------


def test_release_writes_section_and_deletes_fragments(tmp_path):
    log = tmp_path / "CHANGELOG.md"
    log.write_text(HEADER)
    frags = tmp_path / "unreleased"
    _write(frags, "2.json", {"id": 2, "description": "Second."})
    _write(frags, "1.json", {"id": 1, "description": "First."})

    changelog.release("0.1.0", "2026-09-28", log, frags)

    body = changelog.extract_section(log.read_text(), "0.1.0")
    assert body.splitlines()[0].startswith("- First.")
    assert body.splitlines()[1].startswith("- Second.")
    assert "## [0.1.0] - 2026-09-28" in log.read_text()
    assert list(frags.glob("*.json")) == []


def test_release_rejects_bad_version_duplicate_and_empty(tmp_path):
    log = tmp_path / "CHANGELOG.md"
    log.write_text(HEADER + "\n## [0.1.0] - d\n\n- A.\n")
    frags = tmp_path / "unreleased"
    with pytest.raises(changelog.ChangelogError, match="invalid version"):
        changelog.release("v0.2.0", "d", log, frags)
    with pytest.raises(changelog.ChangelogError, match="already has a section"):
        changelog.release("0.1.0", "d", log, frags)
    with pytest.raises(changelog.ChangelogError, match="no changelog fragments"):
        changelog.release("0.2.0", "d", log, frags)


def test_release_allow_empty_records_no_changes(tmp_path):
    log = tmp_path / "CHANGELOG.md"
    log.write_text(HEADER)

    fragments = changelog.release(
        "0.1.1", "2026-09-28", log, tmp_path / "unreleased", allow_empty=True
    )

    assert fragments == []
    assert changelog.extract_section(log.read_text(), "0.1.1") == (
        "- No user-facing changes."
    )


def test_release_leaves_everything_untouched_on_invalid_fragment(tmp_path):
    log = tmp_path / "CHANGELOG.md"
    log.write_text(HEADER)
    frags = tmp_path / "unreleased"
    good = _write(frags, "1.json", {"id": 1, "description": "Good."})
    _write(frags, "2.json", {"id": 3, "description": "Bad."})
    with pytest.raises(changelog.ChangelogError):
        changelog.release("0.1.0", "d", log, frags)
    assert log.read_text() == HEADER
    assert good.exists()


def test_repo_fragments_are_valid():
    changelog.load_fragments()
