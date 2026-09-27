from __future__ import annotations

from benchmarks.core.task import Task
from benchmarks.suites.gaia import (
    GAIA,
    _as_number,
    _extract_final_answer,
    _gaia_match,
    _normalize_str,
)
from harness.result import RunResult
from harness.session import Session


def _result(text: str) -> RunResult:
    return RunResult(output=text, session=Session())


# --- _extract_final_answer ------------------------------------------------


def test_extract_takes_text_after_marker():
    text = "Let me reason.\nStep one.\nFINAL ANSWER: 42"
    assert _extract_final_answer(text) == "42"


def test_extract_uses_last_marker_when_repeated():
    text = "FINAL ANSWER: draft\nactually, reconsidering\nFINAL ANSWER: final"
    assert _extract_final_answer(text) == "final"


def test_extract_is_case_insensitive():
    assert _extract_final_answer("blah\nFinal Answer: Paris") == "Paris"


def test_extract_without_marker_returns_trimmed_whole_output():
    assert _extract_final_answer("  just the answer  ") == "just the answer"


def test_extract_strips_trailing_lines_after_marker():
    assert _extract_final_answer("work\nFINAL ANSWER:  7  \n") == "7"


# --- _as_number -----------------------------------------------------------


def test_as_number_strips_thousands_commas():
    assert _as_number("1,000") == 1000.0


def test_as_number_strips_currency_and_percent():
    assert _as_number("$1,000") == 1000.0
    assert _as_number("50%") == 50.0


def test_as_number_rejects_worded_answer():
    assert _as_number("3 apples") is None
    assert _as_number("Paris") is None


# --- _gaia_match: numbers -------------------------------------------------


def test_match_number_with_thousands_separator():
    assert _gaia_match("1000", "1,000")
    assert _gaia_match("1,000", "1000")


def test_match_number_integer_equals_float():
    assert _gaia_match("42", "42.0")
    assert _gaia_match("42.0", "42")


def test_match_number_mismatch_fails():
    assert not _gaia_match("41", "42")


# --- _gaia_match: strings -------------------------------------------------


def test_match_string_case_and_punctuation_insensitive():
    assert _gaia_match("The Cat.", "the cat")
    assert _gaia_match("PARIS", "paris")


def test_match_string_mismatch_fails():
    assert not _gaia_match("dog", "cat")


def test_normalize_str_drops_punctuation_and_case():
    assert _normalize_str("Hello, World!") == "hello world"


# --- _gaia_match: lists ---------------------------------------------------


def test_match_comma_list_same_order():
    assert _gaia_match("a, b, c", "a, b, c")


def test_match_comma_list_is_order_sensitive():
    assert not _gaia_match("c, b, a", "a, b, c")


def test_match_comma_list_length_mismatch_fails():
    assert not _gaia_match("a, b", "a, b, c")


def test_match_comma_list_applies_numeric_rule_per_element():
    # Per-element numeric rule ("2" == "2.0"); list elements carry no thousands
    # separators (GAIA forbids them), so the comma is the list delimiter.
    assert _gaia_match("1000, 2", "1000, 2.0")


# --- grade() on synthetic Task/RunResult ----------------------------------


def test_grade_passes_on_normalized_match():
    task = Task(task_id="t", prompt="q", expected="1000")
    score = GAIA().grade(task, _result("reasoning\nFINAL ANSWER: 1,000"))
    assert score.passed
    assert score.detail == {"predicted": "1,000", "expected": "1000"}


def test_grade_fails_on_wrong_answer():
    task = Task(task_id="t", prompt="q", expected="Paris")
    score = GAIA().grade(task, _result("FINAL ANSWER: London"))
    assert not score.passed


def test_grade_handles_missing_marker():
    task = Task(task_id="t", prompt="q", expected="yes")
    score = GAIA().grade(task, _result("Yes."))
    assert score.passed


def test_grade_handles_none_expected():
    task = Task(task_id="t", prompt="q", expected=None)
    score = GAIA().grade(task, _result("FINAL ANSWER: anything"))
    assert not score.passed
