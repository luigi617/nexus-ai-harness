from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ClassVar

from benchmarks.core.task import Score, Task
from harness import NexusAIHarness
from harness.result import RunResult
from harness.session import Session
from protocols.model import Model


@dataclass
class Episode:
    """What a benchmark hands the runner to attempt one task.

    Attributes:
        harness: The composed harness (loop + model + tools + guards).
        initial_input: The opening user message for the first ``harness.run``.
        on_user_turn: Optional conversation driver for interactive benchmarks.
            Given the agent's user-facing output for a turn, it returns the
            simulated user's next message, or ``None`` when the episode is over.
            ``None`` (the default) means single-turn: the runner runs the harness
            exactly once. It is called off the event loop, so it may block.
    """

    harness: NexusAIHarness
    initial_input: str
    on_user_turn: Callable[[str], str | None] | None = None


class Benchmark(ABC):
    """Base class for all benchmarks. Subclass and register with ``@register``."""

    #: Stable CLI name, e.g. ``"bfcl"``. Unique across registered benchmarks.
    name: ClassVar[str] = ""
    #: One-line human description shown by ``python -m benchmarks list``.
    description: ClassVar[str] = ""

    @abstractmethod
    def load_tasks(self, *, limit: int | None = None) -> list[Task]:
        """Load (up to ``limit``) tasks for this benchmark."""

    @abstractmethod
    def build_harness(self, task: Task, model: Model, session: Session) -> Episode:
        """Compose the harness and opening input for ``task`` as an ``Episode``.

        Register the loop, the model, this benchmark's tools, and any guards.
        The runner adds its own metrics hook and drives the episode on the given
        ``session``; it does not add a loop or model for you.

        Single-turn benchmarks return ``Episode(harness, initial_input)``.
        Interactive benchmarks (simulated user, multi-turn) also supply
        ``on_user_turn``: the runner runs the harness for one agent turn, passes
        the agent's output to ``on_user_turn``, and re-runs the harness with the
        returned reply on the same ``session`` until ``on_user_turn`` returns
        ``None``. This keeps the production loop under test and layers the
        user simulation outside it. ``session`` is passed in so the driver
        callback and any tools can share the same typed run state.
        """

    @abstractmethod
    def grade(self, task: Task, result: RunResult) -> Score | Awaitable[Score]:
        """Score a finished run. May be sync or ``async def``.

        Read gold data from ``task`` and the model's behaviour from
        ``result`` — ``result.output`` (final text), ``result.session.history``,
        or any typed state a benchmark tool wrote via ``ctx.state(...)``.
        """
