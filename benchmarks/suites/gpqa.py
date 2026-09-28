from __future__ import annotations

import random
import re
from dataclasses import dataclass

from datasets import load_dataset

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.config import BenchmarkConfig
from benchmarks.core.registry import register
from benchmarks.core.task import Score, Task
from nexus_ai_harness.harness import NexusAIHarness
from nexus_ai_harness.harness.result import RunResult
from nexus_ai_harness.harness.session import Session
from nexus_ai_harness.plugins.loops import ChatLoop
from nexus_ai_harness.protocols.model import Model


@dataclass
class GpqaConfig(BenchmarkConfig):
    """GPQA knobs.

    Attributes:
        subset: The Hugging Face dataset config (e.g. ``gpqa_diamond``).
    """

    subset: str = "gpqa_diamond"


_LETTERS = "ABCD"
_INSTRUCTION = (
    "Choose the single best answer. Reason briefly, then end your response with "
    'a line of the form "Answer: <letter>".'
)
# Prefer an explicit "Answer: X" declaration (optionally markdown-bold or
# parenthesized) so a letter mentioned in passing doesn't override the verdict.
_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\**\(?\s*([A-Da-d])\b", re.IGNORECASE)
# A standalone A-D token: not glued to another alphanumeric (so "A" but not
# "Avogadro" or "3A"). Used as a fallback when no explicit answer line is found.
_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Da-d])(?![A-Za-z0-9])")


def _shuffle_options(
    seed: int, correct: str, incorrect: list[str]
) -> tuple[dict[str, str], str]:
    """Lay the correct + incorrect answers into a deterministic A-D ordering.

    Seeding a fresh ``random.Random`` per row makes the layout reproducible
    across runs while keeping the correct answer's letter from always landing
    on ``A``. The correct position is tracked by index, so it stays right even
    when an option's text coincides with another's.

    Args:
        seed: Per-row seed (the row index); equal seeds yield equal orderings.
        correct: The correct answer text.
        incorrect: The three distractor answer texts.

    Returns:
        A ``(letter -> option text, correct letter)`` pair.
    """
    items = [correct, *incorrect]
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    options: dict[str, str] = {}
    correct_letter = ""
    for letter, idx in zip(_LETTERS, order, strict=True):
        options[letter] = items[idx]
        if idx == 0:  # index 0 is the correct answer in ``items``
            correct_letter = letter
    return options, correct_letter


def _render_prompt(question: str, options: dict[str, str]) -> str:
    """Render a multiple-choice prompt from a question and labeled options."""
    lines = [question.strip(), ""]
    lines += [f"{letter}) {options[letter]}" for letter in _LETTERS]
    lines += ["", _INSTRUCTION]
    return "\n".join(lines)


def _extract_choice(text: str | None) -> str | None:
    """Extract the chosen letter (A-D) from a model's answer, or ``None``.

    Prefers an explicit ``Answer: X`` declaration; otherwise falls back to the
    last standalone A-D token in the text. Matching is case-insensitive and the
    result is upper-cased. Returns ``None`` when nothing parses.
    """
    if not text:
        return None
    explicit = _ANSWER_RE.findall(text)
    if explicit:
        return explicit[-1].upper()
    tokens = _TOKEN_RE.findall(text)
    if tokens:
        return tokens[-1].upper()
    return None


@register
class GPQA(Benchmark):
    """GPQA Diamond: graduate-level multiple-choice science QA, single-turn."""

    name = "gpqa"
    description = (
        "GPQA Diamond: graduate-level multiple-choice science QA (exact match)"
    )
    config_type = GpqaConfig

    def load_tasks(self, *, limit: int | None = None) -> list[Task]:
        """Load GPQA rows from the gated HF dataset into labeled MCQ tasks.

        Raises:
            RuntimeError: If the dataset cannot be loaded (it is gated and
                requires Hugging Face authentication).
        """
        subset = self.config.subset
        try:
            rows = load_dataset("Idavidrein/gpqa", subset, split="train")
        except Exception as exc:
            raise RuntimeError(
                f"could not load 'Idavidrein/gpqa' ({subset}); it is a gated "
                "dataset — request access on Hugging Face and authenticate "
                "(huggingface-cli login or set HF_TOKEN)"
            ) from exc

        tasks: list[Task] = []
        for index, row in enumerate(rows):
            if limit is not None and len(tasks) >= limit:
                break
            correct = str(row["Correct Answer"]).strip()
            incorrect = [str(row[f"Incorrect Answer {i}"]).strip() for i in (1, 2, 3)]
            options, correct_letter = _shuffle_options(index, correct, incorrect)
            tasks.append(
                Task(
                    task_id=f"{subset}_{index}",
                    prompt=_render_prompt(str(row["Question"]), options),
                    expected=correct_letter,
                    metadata={"options": options, "subset": subset},
                )
            )
        return tasks

    def build_harness(self, task: Task, model: Model, session: Session) -> Episode:
        harness = NexusAIHarness().use(ChatLoop()).use(model)
        return Episode(harness=harness, initial_input=task.prompt)  # single-turn

    def grade(self, task: Task, result: RunResult) -> Score:
        chosen = _extract_choice(result.output)
        return Score(
            passed=chosen is not None and chosen == task.expected,
            detail={"expected": task.expected, "chosen": chosen},
        )
