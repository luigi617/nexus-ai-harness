from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A single benchmark item.

    Attributes:
        task_id: Stable identifier, unique within a benchmark.
        prompt: The user input handed to ``harness.run(...)``.
        expected: Benchmark-specific gold answer (opaque to the runner).
        metadata: Anything else the benchmark needs at grade time (category,
            available functions, env index, ...).
    """

    task_id: str
    prompt: str
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Score:
    """The verdict for one attempt.

    Attributes:
        passed: Whether the attempt counts as correct.
        value: A continuous score in ``[0, 1]`` when the benchmark defines one;
            leave unset (``None``) to default to ``1.0``/``0.0`` mirroring
            ``passed``. An explicit ``0.0`` is preserved as-is.
        detail: Human-readable diagnostics (what was expected vs. produced).
    """

    passed: bool
    value: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Default the continuous score to the boolean verdict only when the
        # caller did not supply one (None), so an explicit 0.0 survives.
        if self.value is None:
            self.value = 1.0 if self.passed else 0.0


@dataclass
class RunMetrics:
    """Per-attempt cost/latency, read off the session after a run."""

    seconds: float = 0.0
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    stop_reason: str = ""


@dataclass
class Attempt:
    """One graded run of one task (there are ``k`` attempts per task)."""

    task_id: str
    run_index: int
    score: Score
    output: str = ""
    metrics: RunMetrics = field(default_factory=RunMetrics)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.score.passed
