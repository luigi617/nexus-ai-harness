from __future__ import annotations

from benchmarks.core.task import Task
from benchmarks.suites.gpqa import (
    GPQA,
    _extract_choice,
    _render_prompt,
    _shuffle_options,
)
from nexus_ai_harness.harness.result import RunResult
from nexus_ai_harness.harness.session import Session


def _result(text: str) -> RunResult:
    return RunResult(output=text, session=Session())


def test_extract_choice_explicit_answer_line():
    assert _extract_choice("Reasoning here.\nAnswer: C") == "C"


def test_extract_choice_case_insensitive_and_parenthesized():
    assert _extract_choice("the answer is (B)") == "B"


def test_extract_choice_falls_back_to_last_standalone_token():
    assert _extract_choice("It must be option D.") == "D"


def test_extract_choice_ignores_letters_inside_words():
    # "Avogadro" / "Boltzmann" must not be read as A / B choices.
    assert _extract_choice("Avogadro and Boltzmann.\nAnswer: C") == "C"


def test_extract_choice_empty_is_none():
    assert _extract_choice("") is None
    assert _extract_choice(None) is None


def test_extract_choice_unparseable_is_none():
    assert _extract_choice("I am not sure which one.") is None


def test_shuffle_is_deterministic_for_same_seed():
    a = _shuffle_options(7, "right", ["w1", "w2", "w3"])
    b = _shuffle_options(7, "right", ["w1", "w2", "w3"])
    assert a == b
    options, correct_letter = a
    assert set(options) == set("ABCD")
    assert options[correct_letter] == "right"


def test_shuffle_correct_letter_varies_across_rows():
    letters = {_shuffle_options(i, "right", ["w1", "w2", "w3"])[1] for i in range(12)}
    # A per-row seed should not pin the correct answer to a single letter.
    assert len(letters) > 1


def test_shuffle_correct_letter_survives_duplicate_option_text():
    # Correct text equal to a distractor must still map to the right position.
    options, correct_letter = _shuffle_options(3, "dup", ["dup", "x", "y"])
    assert options[correct_letter] == "dup"


def test_render_prompt_lists_all_options_and_instruction():
    prompt = _render_prompt("Q?", {"A": "a", "B": "b", "C": "c", "D": "d"})
    for letter in "ABCD":
        assert f"{letter}) " in prompt
    assert "Answer:" in prompt


def test_grade_correct():
    task = Task(task_id="t", prompt="", expected="B")
    score = GPQA().grade(task, _result("I think it is B.\nAnswer: B"))
    assert score.passed
    assert score.detail["chosen"] == "B"


def test_grade_wrong():
    task = Task(task_id="t", prompt="", expected="B")
    score = GPQA().grade(task, _result("Answer: A"))
    assert not score.passed
    assert score.detail == {"expected": "B", "chosen": "A"}


def test_grade_unparseable_fails():
    task = Task(task_id="t", prompt="", expected="B")
    score = GPQA().grade(task, _result("no idea"))
    assert not score.passed
    assert score.detail["chosen"] is None
