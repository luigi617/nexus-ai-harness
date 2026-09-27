from __future__ import annotations

import json
import subprocess

from benchmarks.suites.swe_bench import (
    _extract_test_ids,
    _git_diff,
    _score_from_report,
)

# --- _extract_test_ids ----------------------------------------------------


def test_extract_test_ids_parses_json_list_string():
    raw = json.dumps(["tests/test_a.py::test_one", "tests/test_b.py::test_two"])
    assert _extract_test_ids(raw) == [
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two",
    ]


def test_extract_test_ids_accepts_decoded_list():
    assert _extract_test_ids(["a::b", "c::d"]) == ["a::b", "c::d"]


def test_extract_test_ids_empty_and_none():
    assert _extract_test_ids("") == []
    assert _extract_test_ids(None) == []
    assert _extract_test_ids("[]") == []


# --- _score_from_report ---------------------------------------------------


def test_score_from_summary_report_resolved():
    report = {
        "resolved_ids": ["repo__proj-123", "repo__proj-999"],
        "unresolved_ids": ["repo__proj-456"],
    }
    score = _score_from_report("repo__proj-123", report)
    assert score.passed is True
    assert score.value == 1.0


def test_score_from_summary_report_unresolved():
    report = {
        "resolved_ids": ["repo__proj-999"],
        "unresolved_ids": ["repo__proj-456"],
    }
    score = _score_from_report("repo__proj-456", report)
    assert score.passed is False
    assert score.value == 0.0


def test_score_from_per_instance_report_resolved():
    report = {
        "repo__proj-123": {
            "resolved": True,
            "tests_status": {"FAIL_TO_PASS": {"success": ["t::a"], "failure": []}},
        }
    }
    score = _score_from_report("repo__proj-123", report)
    assert score.passed is True
    assert score.detail["tests_status"]["FAIL_TO_PASS"]["success"] == ["t::a"]


def test_score_from_per_instance_report_unresolved():
    report = {"repo__proj-123": {"resolved": False, "tests_status": {}}}
    score = _score_from_report("repo__proj-123", report)
    assert score.passed is False


def test_score_from_report_missing_instance_is_failure():
    score = _score_from_report("repo__proj-777", {"resolved_ids": []})
    assert score.passed is False


def test_score_from_report_instance_absent_entirely():
    # No resolved_ids key and no per-instance entry -> errored/skipped -> fail.
    score = _score_from_report("repo__proj-777", {})
    assert score.passed is False
    assert score.detail["reason"] == "instance not present in report"


# --- _git_diff ------------------------------------------------------------


def _git(cwd, *args):
    subprocess.run(
        ["git", "-C", cwd, *args], check=True, capture_output=True, text=True
    )


def test_git_diff_captures_edit_to_tracked_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cwd = str(repo)
    _git(cwd, "init")
    _git(cwd, "config", "user.email", "test@example.com")
    _git(cwd, "config", "user.name", "Test")
    (repo / "app.py").write_text("def add(a, b):\n    return a - b\n")
    _git(cwd, "add", "app.py")
    _git(cwd, "commit", "-m", "initial")

    # The "fix": correct the operator.
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n")

    diff = _git_diff(cwd)
    assert "app.py" in diff
    assert "+    return a + b" in diff
    assert "-    return a - b" in diff


def test_git_diff_captures_new_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cwd = str(repo)
    _git(cwd, "init")
    _git(cwd, "config", "user.email", "test@example.com")
    _git(cwd, "config", "user.name", "Test")
    (repo / "seed.txt").write_text("seed\n")
    _git(cwd, "add", "seed.txt")
    _git(cwd, "commit", "-m", "initial")

    (repo / "new_module.py").write_text("VALUE = 42\n")

    diff = _git_diff(cwd)
    assert "new_module.py" in diff
    assert "+VALUE = 42" in diff
