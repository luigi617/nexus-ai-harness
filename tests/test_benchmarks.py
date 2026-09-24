from __future__ import annotations

import asyncio
import re

import pytest

from benchmarks import Runner, get_benchmark, registered_benchmarks
from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.models import build_model
from benchmarks.core.report import Report
from benchmarks.core.task import Attempt, RunMetrics, Score, Task
from benchmarks.suites.bfcl import (
    BFCL,
    _ast_match,
    _FunctionSpecTool,
    _normalize_schema,
    _params_match,
)
from core.response import Response
from harness import NexusAIHarness
from harness.result import RunResult
from harness.session import Session
from plugins.loops import ChatLoop
from tests.conftest import ScriptedModel


def _attempt(task_id, run_index, passed, *, cost=0.0, error=None) -> Attempt:
    return Attempt(
        task_id=task_id,
        run_index=run_index,
        score=Score(passed=passed),
        metrics=RunMetrics(cost=cost),
        error=error,
    )


# --- Report aggregation ---------------------------------------------------


def test_report_pass_at_1_and_pass_hat_k():
    report = Report(
        benchmark="x",
        model="m",
        k=2,
        attempts=[
            _attempt("a", 0, True),
            _attempt("a", 1, True),  # a passes all k -> counts for pass^k
            _attempt("b", 0, True),
            _attempt("b", 1, False),  # b flaky -> pass@1 yes, pass^k no
            _attempt("c", 0, False),
            _attempt("c", 1, False),
        ],
    )
    assert report.num_tasks == 3
    assert report.pass_at_1 == pytest.approx(2 / 3)  # a, b pass on run 0
    assert report.pass_hat_k == pytest.approx(1 / 3)  # only a passes both


def test_report_avg_cost_is_per_task_not_per_attempt():
    # k=2 over one task, each attempt $0.10 -> task costs $0.20, not $0.10.
    report = Report(
        benchmark="x",
        model="m",
        k=2,
        attempts=[
            _attempt("a", 0, True, cost=0.10),
            _attempt("a", 1, True, cost=0.10),
        ],
    )
    assert report.num_tasks == 1
    assert report.avg_cost_per_task == pytest.approx(0.20)


def test_report_pass_hat_k_requires_all_k_present():
    # Only run_index 0 recorded (e.g. resumed with a larger k) -> not pass^2.
    report = Report(benchmark="x", model="m", k=2, attempts=[_attempt("a", 0, True)])
    assert report.pass_hat_k == 0.0


def test_report_mean_score_partial_credit():
    report = Report(
        benchmark="x",
        model="m",
        k=1,
        attempts=[
            Attempt("a", 0, Score(passed=False, value=0.25)),
            Attempt("b", 0, Score(passed=True, value=1.0)),
        ],
    )
    assert report.mean_score == pytest.approx(0.625)


def test_report_summary_json_safe_when_nothing_solved():
    import json

    report = Report(benchmark="x", model="m", k=1, attempts=[_attempt("a", 0, False)])
    summary = report.summary()
    assert summary["cost_per_solved_task_usd"] is None
    json.loads(json.dumps(summary))  # valid JSON: no bare Infinity token


def test_score_preserves_explicit_zero_value():
    assert Score(passed=True).value == 1.0  # unset -> mirrors passed
    assert Score(passed=True, value=0.0).value == 0.0  # explicit 0.0 kept


def test_report_cost_per_solved_and_error_rate():
    report = Report(
        benchmark="x",
        model="m",
        k=1,
        attempts=[
            _attempt("a", 0, True, cost=0.10),
            _attempt("b", 0, False, cost=0.05),
            _attempt("c", 0, False, error="boom"),
        ],
    )
    assert report.total_cost == pytest.approx(0.15)
    assert report.cost_per_solved_task == pytest.approx(0.15)  # one solved
    assert report.error_rate == pytest.approx(1 / 3)


# --- Runner (end to end through a real harness with a scripted model) -----


class _EchoBenchmark(Benchmark):
    name = "_echo"
    description = "test double"

    def __init__(self, *, fail_grade: bool = False) -> None:
        self._fail_grade = fail_grade

    def load_tasks(self, *, limit=None):
        tasks = [Task(task_id="t1", prompt="hi", expected="X")]
        return tasks[:limit] if limit is not None else tasks

    def build_harness(self, task, model, session):
        return Episode(NexusAIHarness().use(ChatLoop()).use(model), task.prompt)

    def grade(self, task, result: RunResult) -> Score:
        if self._fail_grade:
            raise ValueError("grader blew up")
        return Score(passed=result.output == task.expected)


def test_runner_drives_k_attempts_and_grades():
    model = ScriptedModel(Response(text="X"))
    report = asyncio.run(Runner(_EchoBenchmark(), model, k=3, concurrency=2).run())
    assert report.num_tasks == 1
    assert len(report.attempts) == 3
    assert report.pass_at_1 == 1.0
    assert report.pass_hat_k == 1.0


def test_runner_isolates_grader_errors():
    model = ScriptedModel(Response(text="X"))
    report = asyncio.run(Runner(_EchoBenchmark(fail_grade=True), model, k=1).run())
    (attempt,) = report.attempts
    assert attempt.error is not None
    assert not attempt.passed
    assert report.error_rate == 1.0


def test_runner_resume_skips_recorded_attempts(tmp_path):
    out = tmp_path / "runs.jsonl"
    model = ScriptedModel(Response(text="X"))

    first = asyncio.run(Runner(_EchoBenchmark(), model, k=1, output_path=out).run())
    assert len(first.attempts) == 1

    # Second run resumes: nothing new to attempt, results recovered from disk.
    resumed = asyncio.run(
        Runner(_EchoBenchmark(), model, k=1, output_path=out, resume=True).run()
    )
    assert resumed.num_tasks == 1
    assert resumed.pass_at_1 == 1.0
    assert len(out.read_text().splitlines()) == 1  # not double-written


# --- BFCL AST checker (offline unit tests) --------------------------------


def test_bfcl_params_match_value_sets_and_absent_optionals():
    gold = {"city": ["Paris", "paris"], "unit": ["celsius", ""]}
    assert _params_match(gold, {"city": "Paris"})  # unit optional (accepts "")
    assert _params_match(gold, {"city": "paris", "unit": "celsius"})
    assert not _params_match(gold, {"city": "London"})  # wrong value
    assert not _params_match(gold, {"city": "Paris", "zoom": 3})  # extra param


def test_bfcl_ast_match_parallel_multiset():
    gold = [
        {"get_weather": {"city": ["Paris"]}},
        {"get_weather": {"city": ["Rome"]}},
    ]
    emitted_ok = [
        {"name": "get_weather", "arguments": {"city": "Rome"}},
        {"name": "get_weather", "arguments": {"city": "Paris"}},
    ]
    assert _ast_match(gold, emitted_ok)
    assert not _ast_match(gold, emitted_ok[:1])  # wrong count
    wrong = [
        {"name": "get_weather", "arguments": {"city": "Paris"}},
        {"name": "get_weather", "arguments": {"city": "Berlin"}},
    ]
    assert not _ast_match(gold, wrong)


def test_bfcl_number_tolerance():
    assert _params_match({"n": [3]}, {"n": 3.0})
    assert _params_match({"n": [3.0]}, {"n": 3})


def test_bfcl_bool_not_equal_to_int():
    # bool is an int subclass; a boolean arg must not satisfy an int gold value.
    assert not _params_match({"n": [1]}, {"n": True})
    assert not _params_match({"n": [0]}, {"n": False})
    assert _params_match({"flag": [True]}, {"flag": True})  # genuine bool still ok


def test_bfcl_dotted_names_sanitized_and_still_grade():
    # Backends reject '.' in tool names; the registered tool name must be legal
    # while gold matching still lines up against the original dotted name.
    tool = _FunctionSpecTool({"name": "math.factorial", "parameters": {}})
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", tool.name)
    assert tool.name == "math_factorial"
    gold = [{"math.factorial": {"n": [5]}}]
    emitted = [{"name": "math_factorial", "arguments": {"n": 5}}]
    assert _ast_match(gold, emitted)


def test_bfcl_nested_dict_argument_graded_recursively():
    # BFCL encodes a dict-valued arg as {key: [accepted, ...]}; a correct nested
    # dict must match, and a wrong nested value must still fail.
    gold = [{"find": {"budget": [{"min": [300000], "max": [400000]}]}}]
    ok = [{"name": "find", "arguments": {"budget": {"min": 300000, "max": 400000}}}]
    bad = [{"name": "find", "arguments": {"budget": {"min": 300000, "max": 999}}}]
    assert _ast_match(gold, ok)
    assert not _ast_match(gold, bad)


def test_bfcl_normalize_schema_maps_nested_types():
    # BFCL 'dict'/'float'/'tuple' must be mapped at every depth, not just the top.
    schema = {
        "type": "dict",
        "properties": {
            "budget": {
                "type": "dict",
                "properties": {
                    "min": {"type": "float"},
                    "max": {"type": "float"},
                },
            },
            "tags": {"type": "array", "items": {"type": "tuple"}},
        },
    }
    out = _normalize_schema(schema)
    assert out["type"] == "object"
    budget = out["properties"]["budget"]
    assert budget["type"] == "object"
    assert budget["properties"]["min"]["type"] == "number"
    assert budget["properties"]["max"]["type"] == "number"
    assert out["properties"]["tags"]["items"]["type"] == "array"


def test_bfcl_limit_fills_across_categories(monkeypatch):
    # limit not divisible by category count must still fill to exactly limit.
    from benchmarks.suites import bfcl

    def fake_load(relpath):
        # Each category file has plenty of rows; possible_answer files empty-ish.
        if relpath.startswith("possible_answer/"):
            return [{"id": f"{relpath}_{i}", "ground_truth": []} for i in range(50)]
        return [
            {
                "id": f"{relpath}_{i}",
                "question": [[{"role": "user", "content": "hi"}]],
                "function": [],
            }
            for i in range(50)
        ]

    monkeypatch.setattr(bfcl, "_load_jsonl", fake_load)
    bench = BFCL()  # 4 default categories
    assert len(bench.load_tasks(limit=10)) == 10
    assert len(bench.load_tasks(limit=7)) == 7


def test_bfcl_grade_captures_emitted_calls_offline():
    """build_harness + grade end to end without touching the network."""
    task = Task(
        task_id="simple_0",
        prompt="What's the weather in Paris?",
        expected=[{"get_weather": {"city": ["Paris"]}}],
        metadata={
            "category": "simple",
            "irrelevance": False,
            "functions": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    )
    model = ScriptedModel(
        Response(
            text="",
            tool_calls=[
                {"id": "c1", "name": "get_weather", "arguments": {"city": "Paris"}}
            ],
        )
    )
    bench = BFCL()
    episode = bench.build_harness(task, model, Session())
    result = asyncio.run(episode.harness.run(episode.initial_input))
    assert bench.grade(task, result).passed


def test_bfcl_irrelevance_passes_when_no_call():
    task = Task(
        task_id="irr_0", prompt="hi", expected=[], metadata={"irrelevance": True}
    )
    # No CapturedCalls state written -> zero emitted calls -> irrelevance passes.
    result = RunResult(output="hello", session=Session())
    assert BFCL().grade(task, result).passed


# --- conversation driver (interactive benchmarks) -------------------------


def test_runner_drives_multi_turn_conversation():
    # For an Episode with on_user_turn, the runner re-runs the harness on the
    # same session with each simulated reply until on_user_turn returns None.
    seen: list[str] = []

    class _ConvoBenchmark(Benchmark):
        name = "_convo"
        description = "test double"

        def load_tasks(self, *, limit=None):
            return [Task(task_id="c1", prompt="hi")]

        def build_harness(self, task, model, session):
            def on_user_turn(agent_output):
                seen.append(agent_output)
                return "again" if len(seen) < 3 else None  # 3 agent turns total

            harness = NexusAIHarness().use(ChatLoop()).use(model)
            return Episode(harness, task.prompt, on_user_turn=on_user_turn)

        def grade(self, task, result: RunResult) -> Score:
            return Score(passed=len(seen) == 3)

    model = ScriptedModel(Response(text="turn"))
    report = asyncio.run(Runner(_ConvoBenchmark(), model, k=1).run())
    assert report.pass_at_1 == 1.0
    assert seen == ["turn", "turn", "turn"]  # looped, didn't stop after turn 1


def test_runner_respects_max_user_turns():
    # A driver that never says None is bounded by max_user_turns.
    class _NeverEnds(Benchmark):
        name = "_never"
        description = "test double"

        def load_tasks(self, *, limit=None):
            return [Task(task_id="n1", prompt="hi")]

        def build_harness(self, task, model, session):
            harness = NexusAIHarness().use(ChatLoop()).use(model)
            return Episode(harness, task.prompt, on_user_turn=lambda _out: "more")

        def grade(self, task, result: RunResult) -> Score:
            return Score(passed=True)

    model = ScriptedModel(Response(text="ok"))
    report = asyncio.run(Runner(_NeverEnds(), model, k=1, max_user_turns=3).run())
    assert report.error_rate == 0.0  # bounded, not hung


def test_tau_bench_respond_records_and_signals_done():
    # tau-bench's respond relay records reward/done into TauState and returns
    # None once the env ends the conversation (the driver's stop signal).
    pytest.importorskip("tau_bench")
    import threading
    from types import SimpleNamespace

    from benchmarks.suites.tau_bench import TauState, _respond

    class FakeEnv:
        def __init__(self):
            self._n = 0

        def step(self, action):
            self._n += 1
            if self._n == 1:
                return SimpleNamespace(
                    observation="my email is a@b.com", reward=0.0, done=False, info={}
                )
            return SimpleNamespace(observation="bye", reward=1.0, done=True, info={})

    env, state, lock = FakeEnv(), TauState(), threading.Lock()
    assert _respond(env, lock, state, "verify?") == "my email is a@b.com"
    assert state.done is False and state.reward == 0.0 and state.steps == 1
    assert _respond(env, lock, state, "done") is None  # env ended -> stop
    assert state.done is True and state.reward == 1.0 and state.steps == 2


# --- model factory + registry --------------------------------------------


def test_build_model_provider_prefix():
    model = build_model("bedrock:us.anthropic.claude-opus-4-8")
    assert model.name == "us.anthropic.claude-opus-4-8"
    assert model.provider == "bedrock"


def test_build_model_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        build_model("nope:whatever")


def test_suites_are_registered():
    names = registered_benchmarks()
    assert "bfcl" in names
    assert "tau-bench" in names
    assert get_benchmark("bfcl").name == "bfcl"
    with pytest.raises(KeyError):
        get_benchmark("does-not-exist")
