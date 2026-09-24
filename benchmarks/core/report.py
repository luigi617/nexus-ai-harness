from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

from benchmarks.core.task import Attempt


@dataclass
class Report:
    """Aggregated results for one (benchmark, model) run."""

    benchmark: str
    model: str
    k: int
    attempts: list[Attempt] = field(default_factory=list)

    def _by_task(self) -> dict[str, list[Attempt]]:
        grouped: dict[str, list[Attempt]] = defaultdict(list)
        for attempt in self.attempts:
            grouped[attempt.task_id].append(attempt)
        return grouped

    @property
    def num_tasks(self) -> int:
        return len(self._by_task())

    @property
    def pass_at_1(self) -> float:
        """Fraction of tasks whose first attempt (run_index 0) passed."""
        firsts = [
            next((a for a in runs if a.run_index == 0), None)
            for runs in self._by_task().values()
        ]
        graded = [a for a in firsts if a is not None]
        return _ratio(sum(a.passed for a in graded), len(graded))

    @property
    def pass_hat_k(self) -> float:
        """Fraction of tasks where all k attempts (run_index 0..k-1) passed.

        Keyed on run_index so a task with fewer than k recorded attempts fails
        pass^k and extra attempts (e.g. from resuming with a smaller ``--k``)
        are ignored rather than silently changing the denominator.
        """
        tasks = self._by_task()
        if not tasks:
            return 0.0
        solid = 0
        for runs in tasks.values():
            by_index = {a.run_index: a for a in runs}
            wanted = (by_index.get(i) for i in range(self.k))
            if all(a is not None and a.passed for a in wanted):
                solid += 1
        return _ratio(solid, len(tasks))

    @property
    def mean_score(self) -> float:
        """Mean over tasks of each task's mean ``Score.value`` (partial credit).

        Lets benchmarks whose signal is continuous (reward, F1, rubric score)
        surface a headline metric that pass@1/pass^k can't express.
        """
        tasks = self._by_task()
        if not tasks:
            return 0.0
        return mean(mean(a.score.value for a in runs) for runs in tasks.values())

    @property
    def error_rate(self) -> float:
        """Fraction of attempts that raised rather than being graded normally."""
        return _ratio(sum(a.error is not None for a in self.attempts), self.attempts)

    @property
    def total_cost(self) -> float:
        return sum(a.metrics.cost for a in self.attempts)

    @property
    def avg_cost_per_task(self) -> float:
        # Per task, not per attempt: with k>1 a task costs all k attempts.
        return _ratio(self.total_cost, self.num_tasks)

    @property
    def avg_tokens_per_task(self) -> float:
        toks = sum(
            a.metrics.input_tokens + a.metrics.output_tokens for a in self.attempts
        )
        return _ratio(toks, self.num_tasks)

    @property
    def cost_per_solved_task(self) -> float:
        solved = sum(a.passed for a in self.attempts)
        return self.total_cost / solved if solved else float("inf")

    def summary(self) -> dict[str, float | int | str | None]:
        # ``cost_per_solved_task`` is inf when nothing solved; emit None so the
        # --json output stays valid JSON (json.dumps writes inf as `Infinity`).
        cps = self.cost_per_solved_task
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "k": self.k,
            "tasks": self.num_tasks,
            "attempts": len(self.attempts),
            "pass@1": round(self.pass_at_1, 4),
            f"pass^{self.k}": round(self.pass_hat_k, 4),
            "mean_score": round(self.mean_score, 4),
            "error_rate": round(self.error_rate, 4),
            "total_cost_usd": round(self.total_cost, 4),
            "avg_cost_per_task_usd": round(self.avg_cost_per_task, 6),
            "avg_tokens_per_task": round(self.avg_tokens_per_task, 1),
            "cost_per_solved_task_usd": round(cps, 6) if cps != float("inf") else None,
        }

    def render(self) -> str:
        lines = [f"=== {self.benchmark} @ {self.model} ==="]
        lines += [f"  {k:<26} {v}" for k, v in self.summary().items()]
        return "\n".join(lines)


def _ratio(num: float, denom) -> float:
    n = denom if isinstance(denom, int) else len(denom)
    if not n:
        return 0.0
    return num / n
