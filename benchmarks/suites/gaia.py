from __future__ import annotations

import os
import re
import shutil
import string
import tempfile
from dataclasses import dataclass

from datasets import load_dataset

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.config import BenchmarkConfig
from benchmarks.core.registry import register
from benchmarks.core.task import Score, Task
from nexus_ai_harness.harness import NexusAIHarness
from nexus_ai_harness.harness.result import RunResult
from nexus_ai_harness.harness.session import Session
from nexus_ai_harness.plugins.guards import MaxIterations, Timeout
from nexus_ai_harness.plugins.hooks import ElapsedTime, IterationCounter
from nexus_ai_harness.plugins.loops import AgenticLoop
from nexus_ai_harness.plugins.permissions import AutoApprove
from nexus_ai_harness.plugins.sandbox import WorkspaceSandbox
from nexus_ai_harness.plugins.tools import ListDir, ReadFile, Shell
from nexus_ai_harness.protocols.lifecycle import Lifecycle
from nexus_ai_harness.protocols.model import Model
from nexus_ai_harness.protocols.plugin import Plugin


@dataclass
class GaiaConfig(BenchmarkConfig):
    """GAIA knobs.

    Attributes:
        subset: The Hugging Face dataset config (e.g. ``2023_all``).
        split: The dataset split (e.g. ``validation``).
        max_steps: Max agent steps per task.
        timeout_s: Per-task wall-clock limit, in seconds.
    """

    subset: str = "2023_all"
    split: str = "validation"
    max_steps: int = 30
    timeout_s: float = 600.0


# GAIA prompts the agent to end with this exact marker; grading reads whatever
# follows the last one.
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
# Trimmed from the ends of a numeric answer: currency and percent decorations
# GAIA gold values may carry. Commas (thousands separators) are removed globally.
_NUMBER_EDGE = "$%£€ \t"

_PROMPT = (
    "You are a general AI assistant. Answer the question below. Reason step by "
    "step, using the available tools to inspect any attached file, then finish "
    "with a single line in the exact form 'FINAL ANSWER: <answer>'. The answer "
    "must be a number, as few words as possible, or a comma-separated list of "
    "those; do not add units unless the question asks for them, and do not use "
    "thousands separators in numbers.\n\nQuestion: "
)


class _WorkspaceCleanup(Plugin, Lifecycle):
    """Deletes the per-task temp workspace when the harness stops.

    The runner stops the harness after grading, so ``stop`` is the contracted
    teardown hook for a per-task environment (here, the sandbox root directory).
    """

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

    async def stop(self) -> None:
        """Remove the workspace tree, ignoring an already-gone directory."""
        shutil.rmtree(self._workspace, ignore_errors=True)


@register
class GAIA(Benchmark):
    """The GAIA suite: general assistant tasks graded by normalized exact match.

    Each task runs the production ``AgenticLoop`` with filesystem/shell tools
    confined to a per-task temp workspace, so the agent can inspect any file
    attached to the question. Full GAIA also expects web browsing and richer
    file parsing; the operator must supply those extra tools to score well.
    """

    name = "gaia"
    description = "GAIA: general assistant tasks graded by normalized exact match"
    config_type = GaiaConfig

    def load_tasks(self, *, limit: int | None = None) -> list[Task]:
        dataset = load_dataset(
            "gaia-benchmark/GAIA",
            self.config.subset,
            split=self.config.split,
        )
        tasks: list[Task] = []
        for row in dataset:
            tasks.append(
                Task(
                    task_id=str(row["task_id"]),
                    prompt=row["Question"],
                    expected=row["Final answer"],
                    metadata={
                        "level": row.get("Level"),
                        "file_name": row.get("file_name", ""),
                        "file_path": row.get("file_path", ""),
                    },
                )
            )
            if limit is not None and len(tasks) >= limit:
                break
        return tasks

    def build_harness(self, task: Task, model: Model, session: Session) -> Episode:
        # mkdtemp runs off the event loop here, so blocking setup (creating the
        # dir, copying the attached file in) is fine before tools initialize.
        workspace = tempfile.mkdtemp(prefix="nexus-gaia-")
        file_path = task.metadata.get("file_path") or ""
        if file_path and os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(workspace, os.path.basename(file_path)))

        harness = (
            NexusAIHarness()
            .use(AgenticLoop())  # the real production loop under test
            .use(model)
            .use(AutoApprove())
            .use(IterationCounter())
            .use(ElapsedTime())
            .use(MaxIterations(self.config.max_steps))
            .use(Timeout(self.config.timeout_s))
            .use(WorkspaceSandbox(workspace))
            .use(ReadFile())
            .use(ListDir())
            .use(Shell())
            .use(_WorkspaceCleanup(workspace))  # stop() rmtrees the workspace
        )
        return Episode(harness=harness, initial_input=_build_prompt(task))

    def grade(self, task: Task, result: RunResult) -> Score:
        predicted = _extract_final_answer(result.output or "")
        gold = "" if task.expected is None else str(task.expected)
        return Score(
            passed=_gaia_match(predicted, gold),
            detail={"predicted": predicted, "expected": gold},
        )


def _build_prompt(task: Task) -> str:
    """Render the GAIA instruction, noting any file attached in the workspace."""
    prompt = _PROMPT + task.prompt
    file_name = task.metadata.get("file_name") or ""
    if file_name:
        prompt += (
            f"\n\nAn attached file named '{file_name}' is available in your "
            "workspace; use list_dir and read_file to inspect it."
        )
    return prompt


def _extract_final_answer(text: str) -> str:
    """Return the text after the last ``FINAL ANSWER:`` marker, else all of it."""
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return text.strip()
    return text[matches[-1].end() :].strip()


def _gaia_match(pred: str, gold: str) -> bool:
    """Compare a prediction to a gold answer under GAIA's normalization rules.

    Numbers are compared as floats (commas and currency/percent decoration
    stripped); comma-separated lists are compared element-wise and are
    order-sensitive; everything else is a case- and punctuation-insensitive
    string compare. Whether ``gold`` is a number is decided first, so a value
    like ``"1,000"`` stays a single number rather than being split as a list.
    """
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    if _as_number(gold) is None and "," in gold:
        gold_parts = _split_list(gold)
        pred_parts = _split_list(pred)
        if len(gold_parts) != len(pred_parts):
            return False
        return all(
            _match_element(p, g) for p, g in zip(pred_parts, gold_parts, strict=True)
        )
    return _match_element(pred, gold)


def _match_element(pred: str, gold: str) -> bool:
    """Match one scalar element: numeric when ``gold`` is a number, else string."""
    gold_num = _as_number(gold)
    if gold_num is not None:
        pred_num = _as_number(pred)
        return pred_num is not None and pred_num == gold_num
    return _normalize_str(pred) == _normalize_str(gold)


def _as_number(text: str) -> float | None:
    """Parse ``text`` as a float after stripping commas and currency/percent.

    Arbitrary trailing units (e.g. ``"12 km"``) are intentionally not stripped:
    doing so would misclassify worded string answers as numbers.
    """
    cleaned = text.strip().replace(",", "").strip(_NUMBER_EDGE)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_str(text: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace for string compare."""
    return " ".join(text.strip().lower().translate(_PUNCT_TABLE).split())


def _split_list(text: str) -> list[str]:
    """Split a comma-separated answer into trimmed elements."""
    return [part.strip() for part in text.split(",")]
