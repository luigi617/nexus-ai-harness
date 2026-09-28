from __future__ import annotations

from benchmarks.core.task import Task
from benchmarks.suites.human_eval import (
    HumanEval,
    _assemble_program,
    _extract_code,
    _run_check,
)
from nexus_ai_harness.harness.result import RunResult
from nexus_ai_harness.harness.session import Session

_ADD_PROMPT = 'def add(a, b):\n    """Add two numbers."""\n'
_ADD_TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n"


def test_extract_code_from_python_fence():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
    assert _extract_code(text) == "def f():\n    return 1"


def test_extract_code_from_bare_fence():
    text = "```\ndef f():\n    return 1\n```"
    assert _extract_code(text) == "def f():\n    return 1"


def test_extract_code_plain_text_unwrapped():
    text = "def f():\n    return 1"
    assert _extract_code(text) == "def f():\n    return 1"


def test_extract_code_takes_first_fence_when_prose_wrapped():
    text = "Explanation first.\n```python\ndef f():\n    return 2\n```\nmore prose"
    assert _extract_code(text) == "def f():\n    return 2"


def test_run_check_passes_on_correct_program():
    program = "assert 1 + 1 == 2\n"
    assert _run_check(program, timeout=10) is True


def test_run_check_fails_on_failing_assert():
    program = "assert 1 + 1 == 3\n"
    assert _run_check(program, timeout=10) is False


def test_run_check_times_out_without_hanging():
    program = "while True:\n    pass\n"
    assert _run_check(program, timeout=1) is False


def test_assemble_prefers_full_code_when_signature_reemitted():
    # A self-contained completion must not be double-prefixed with the prompt.
    completion = "def add(a, b):\n    return a + b"
    program = _assemble_program(_ADD_PROMPT, completion, _ADD_TEST, "add")
    assert program.count("def add(") == 1


def test_assemble_appends_body_when_only_body_returned():
    completion = "    return a + b\n"
    program = _assemble_program(_ADD_PROMPT, completion, _ADD_TEST, "add")
    assert program.startswith(_ADD_PROMPT)
    assert "def add(" in program


def _grade(output: str) -> bool:
    task = Task(
        task_id="add_0",
        prompt="",
        metadata={"prompt": _ADD_PROMPT, "test": _ADD_TEST, "entry_point": "add"},
    )
    result = RunResult(output=output, session=Session())
    return HumanEval().grade(task, result).passed


def test_grade_passes_on_correct_completion():
    assert _grade("```python\ndef add(a, b):\n    return a + b\n```") is True


def test_grade_fails_on_incorrect_completion():
    assert _grade("```python\ndef add(a, b):\n    return a - b\n```") is False
