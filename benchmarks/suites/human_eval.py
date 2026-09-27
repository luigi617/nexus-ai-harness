from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from datasets import load_dataset

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.config import BenchmarkConfig
from benchmarks.core.registry import register
from benchmarks.core.task import Score, Task
from harness import NexusAIHarness
from harness.result import RunResult
from harness.session import Session
from plugins.loops import ChatLoop
from protocols.model import Model

_INSTRUCTION = (
    "Complete the following Python function. Return the full function "
    "definition including its signature, as Python code and nothing else."
)
# The model is told to return full code, so grab the first fenced block if the
# reply is wrapped in Markdown/prose; otherwise treat the whole reply as code.
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


@dataclass
class HumanEvalConfig(BenchmarkConfig):
    """HumanEval knobs.

    Attributes:
        timeout_s: Per-completion execution timeout, in seconds.
    """

    timeout_s: float = 15.0


@register
class HumanEval(Benchmark):
    """HumanEval: complete a Python function, graded by executing its unit tests."""

    name = "humaneval"
    description = "HumanEval: function completion graded by executing unit tests"
    config_type = HumanEvalConfig

    def load_tasks(self, *, limit: int | None = None) -> list[Task]:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        return [
            Task(
                task_id=row["task_id"],
                prompt=f"{_INSTRUCTION}\n\n{row['prompt']}",
                metadata={
                    "prompt": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                },
            )
            for row in dataset
        ]

    def build_harness(self, task: Task, model: Model, session: Session) -> Episode:
        harness = NexusAIHarness().use(ChatLoop()).use(model)
        return Episode(harness=harness, initial_input=task.prompt)  # single-turn

    def grade(self, task: Task, result: RunResult) -> Score:
        completion = _extract_code(result.output)
        program = _assemble_program(
            task.metadata["prompt"],
            completion,
            task.metadata["test"],
            task.metadata["entry_point"],
        )
        passed = _run_check(program, self.config.timeout_s)
        return Score(
            passed=passed,
            detail={
                "task_id": task.task_id,
                "entry_point": task.metadata["entry_point"],
            },
        )


# --- code extraction and assembly (offline-testable) ----------------------


def _extract_code(text: str) -> str:
    """Pull runnable Python out of a model reply, stripping Markdown/prose.

    Returns the first fenced code block's contents when the reply is fenced,
    else the whole reply trimmed of surrounding whitespace.
    """
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip("\n")
    return text.strip()


def _defines_entry_point(code: str, entry_point: str) -> bool:
    """Whether ``code`` already contains a ``def entry_point(...)``."""
    pattern = rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\("
    return re.search(pattern, code) is not None


def _assemble_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """Build the runnable program that defines the solution and invokes ``check``.

    When the completion already re-emits the signature (a full function/module),
    it is self-contained and used verbatim; otherwise it is treated as the body
    and appended to the prompt's signature, so both model styles run correctly.
    """
    if _defines_entry_point(completion, entry_point):
        body = completion
    else:
        body = prompt + completion
    return f"{body}\n{test}\n\ncheck({entry_point})\n"


def _run_check(program: str, timeout: float) -> bool:
    """Execute ``program`` in a subprocess; True iff it exits 0 within ``timeout``.

    Runs in a throwaway cwd so file-touching completions cannot pollute the
    repo, and swallows timeouts/spawn errors as a failing (not raising) result.
    """
    workdir = tempfile.mkdtemp(prefix="humaneval_")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            cwd=workdir,
            timeout=timeout,
            check=False,
        )
        return completed.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
