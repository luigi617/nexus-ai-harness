from __future__ import annotations

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.registry import get_benchmark, register, registered_benchmarks
from benchmarks.core.report import Report
from benchmarks.core.runner import Runner
from benchmarks.core.task import Attempt, RunMetrics, Score, Task

# Importing the suites registers them by name (side effect).
from benchmarks.suites import bfcl as _bfcl  # noqa: F401
from benchmarks.suites import tau_bench as _tau_bench  # noqa: F401

__all__ = [
    "Attempt",
    "Benchmark",
    "Episode",
    "Report",
    "RunMetrics",
    "Runner",
    "Score",
    "Task",
    "get_benchmark",
    "register",
    "registered_benchmarks",
]
