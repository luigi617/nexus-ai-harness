from __future__ import annotations

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.registry import get_benchmark, register, registered_benchmarks
from benchmarks.core.report import Report
from benchmarks.core.runner import Runner
from benchmarks.core.task import Attempt, RunMetrics, Score, Task

# Importing the suites registers them by name (side effect). Suites import their
# deps at module top, so `import benchmarks` requires the [benchmarks] extra.
from benchmarks.suites import bfcl as _bfcl  # noqa: F401
from benchmarks.suites import gaia as _gaia  # noqa: F401
from benchmarks.suites import gpqa as _gpqa  # noqa: F401
from benchmarks.suites import human_eval as _human_eval  # noqa: F401
from benchmarks.suites import swe_bench as _swe_bench  # noqa: F401
from benchmarks.suites import tau2 as _tau2  # noqa: F401
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
