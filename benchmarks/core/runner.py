from __future__ import annotations

import asyncio
import inspect
import json
import time
import traceback
from collections.abc import Awaitable
from pathlib import Path

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.metrics import MetricsCollector, read_metrics
from benchmarks.core.report import Report
from benchmarks.core.task import Attempt, RunMetrics, Score, Task
from harness.result import RunResult
from harness.session import Session
from protocols.model import Model


class Runner:
    """Drives a single benchmark against a single model."""

    def __init__(
        self,
        benchmark: Benchmark,
        model: Model,
        *,
        k: int = 1,
        concurrency: int = 4,
        task_timeout_s: float = 600.0,
        max_user_turns: int = 50,
        output_path: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        self.benchmark = benchmark
        self.model = model
        self.k = k
        self.concurrency = concurrency
        self.task_timeout_s = task_timeout_s
        # Safety net for interactive episodes; the env's own `done` normally
        # ends the conversation well before this.
        self.max_user_turns = max_user_turns
        self.output_path = Path(output_path) if output_path else None
        self.resume = resume
        self._sem = asyncio.Semaphore(concurrency)
        self._write_lock = asyncio.Lock()

    async def run(self, *, limit: int | None = None) -> Report:
        """Load tasks, attempt each ``k`` times, and return an aggregated report."""
        tasks = self.benchmark.load_tasks(limit=limit)
        done = self._already_done()
        pending = [
            (task, run_index)
            for task in tasks
            for run_index in range(self.k)
            if (task.task_id, run_index) not in done
        ]
        attempts = await asyncio.gather(
            *(self._attempt(task, run_index) for task, run_index in pending)
        )
        recovered = [self._attempt_from_record(r) for r in done.values()]
        return Report(
            benchmark=self.benchmark.name,
            model=self.model.name,
            k=self.k,
            attempts=recovered + list(attempts),
        )

    async def _attempt(self, task: Task, run_index: int) -> Attempt:
        async with self._sem:
            start = time.monotonic()
            session = Session()
            try:
                episode = self.benchmark.build_harness(task, self.model, session)
                episode.harness.use(MetricsCollector())
                # NOTE: wait_for cancels only the awaiting coroutine. A model
                # call offloaded to a worker thread (sync backends) keeps running
                # until the backend's own HTTP timeout; keep backend timeouts
                # below task_timeout_s so a stuck attempt doesn't hold a slot.
                result: RunResult = await asyncio.wait_for(
                    self._drive(episode, session),
                    timeout=self.task_timeout_s,
                )
                score = await _resolve(self.benchmark.grade(task, result))
                attempt = Attempt(
                    task_id=task.task_id,
                    run_index=run_index,
                    score=score,
                    output=result.output,
                    metrics=read_metrics(session, seconds=time.monotonic() - start),
                )
            except Exception:  # one bad task must not sink the whole run
                attempt = Attempt(
                    task_id=task.task_id,
                    run_index=run_index,
                    score=Score(passed=False, detail={"error": "exception"}),
                    metrics=read_metrics(session, seconds=time.monotonic() - start),
                    error=traceback.format_exc(limit=4),
                )
            await self._record(attempt)
            return attempt

    async def _drive(self, episode: Episode, session: Session) -> RunResult:
        """Drive one episode to completion and return the final ``RunResult``.

        Runs the harness for one agent turn; for interactive episodes, relays
        the agent's output to ``on_user_turn`` and re-runs on the same session
        with the user's reply until the episode ends (or the turn cap trips).
        The conversation driver lives here — outside the agent loop — so the
        real production loop is benchmarked and the user simulation is layered
        on top. History, metrics, and typed state accumulate on ``session``.
        """
        user_input = episode.initial_input
        result = await episode.harness.run(user_input, session=session)
        if episode.on_user_turn is None:
            return result
        for _ in range(self.max_user_turns):
            # Off the event loop: on_user_turn may call a (blocking) user model.
            reply = await asyncio.to_thread(episode.on_user_turn, result.output)
            if reply is None:  # episode over
                break
            result = await episode.harness.run(reply, session=session)
        return result

    # --- resumable JSONL output -------------------------------------------

    def _already_done(self) -> dict[tuple[str, int], dict]:
        if not (self.resume and self.output_path and self.output_path.exists()):
            return {}
        done: dict[tuple[str, int], dict] = {}
        for line in self.output_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[(rec["task_id"], rec["run_index"])] = rec
        return done

    async def _record(self, attempt: Attempt) -> None:
        if not self.output_path:
            return
        record = {
            "task_id": attempt.task_id,
            "run_index": attempt.run_index,
            "passed": attempt.passed,
            "value": attempt.score.value,
            "output": attempt.output,
            "error": attempt.error,
            "detail": attempt.score.detail,
            "metrics": attempt.metrics.__dict__,
        }
        async with self._write_lock:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    @staticmethod
    def _attempt_from_record(rec: dict) -> Attempt:
        return Attempt(
            task_id=rec["task_id"],
            run_index=rec["run_index"],
            score=Score(
                passed=rec["passed"], value=rec["value"], detail=rec.get("detail", {})
            ),
            output=rec.get("output", ""),
            metrics=RunMetrics(**rec.get("metrics", {})),
            error=rec.get("error"),
        )


async def _resolve(value: Score | Awaitable[Score]) -> Score:
    return await value if inspect.isawaitable(value) else value
