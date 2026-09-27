from __future__ import annotations

import threading
from types import SimpleNamespace

from benchmarks.core.task import Task
from benchmarks.suites.tau2 import (
    Tau2Bench,
    Tau2State,
    _EnvResponse,
    _record,
    _respond,
    _use_tool,
)
from harness.result import RunResult
from harness.session import Session


class FakeEnv:
    """A tau2 session stand-in: scripts tool outputs and user replies.

    Mirrors the ``call_tool`` / ``respond`` surface the harness-side relay uses,
    returning ``SimpleNamespace`` responses so the domain-agnostic record/relay
    logic can be exercised without importing tau2.
    """

    def __init__(self):
        self._turns = 0

    def call_tool(self, name, arguments):
        return SimpleNamespace(
            observation=f"{name}:{arguments.get('id', '')}",
            reward=0.0,
            done=False,
            info={"error": False},
        )

    def respond(self, message):
        self._turns += 1
        if self._turns == 1:
            return SimpleNamespace(
                observation="my confirmation is ABC123",
                reward=0.0,
                done=False,
                info={},
            )
        return SimpleNamespace(
            observation="thanks, bye",
            reward=1.0,
            done=True,
            info={"termination": "user_stop"},
        )


def test_respond_records_and_signals_done():
    # The respond relay records reward/done into Tau2State and returns None once
    # the simulated user ends the conversation (the driver's stop signal).
    env, state, lock = FakeEnv(), Tau2State(), threading.Lock()
    assert _respond(env, lock, state, "how can I help?") == "my confirmation is ABC123"
    assert state.done is False and state.reward == 0.0 and state.steps == 1
    assert _respond(env, lock, state, "all set") is None  # user ended -> stop
    assert state.done is True and state.reward == 1.0 and state.steps == 2


def test_use_tool_records_step_without_ending():
    env, state, lock = FakeEnv(), Tau2State(), threading.Lock()
    resp = _use_tool(env, "get_reservation", {"id": "7"}, lock, state)
    assert resp.observation == "get_reservation:7"
    assert state.steps == 1 and state.done is False and state.reward == 0.0


def test_record_latches_terminal_reading():
    # A late tool step after the terminal turn must not reset done/reward.
    state = Tau2State()
    _record(state, _EnvResponse(reward=1.0, done=True, info={"termination": "x"}))
    assert state.done is True and state.reward == 1.0 and state.steps == 1
    _record(state, _EnvResponse(reward=0.0, done=False))  # late, concurrent step
    assert state.done is True and state.reward == 1.0  # latched
    assert state.steps == 2  # but still counted


def test_record_reads_missing_fields_as_defaults():
    # SimpleNamespace lacking fields (a partial env response) folds to zeros.
    state = Tau2State()
    _record(state, SimpleNamespace(observation="hi"))
    assert state.reward == 0.0 and state.done is False and state.extra == {}


def _graded(reward: float) -> bool:
    task = Task(task_id="airline_0", prompt="", metadata={})
    result = RunResult(output="", session=Session())
    result.session.state(Tau2State).reward = reward
    return Tau2Bench().grade(task, result).passed


def test_grade_passes_only_on_full_reward():
    assert _graded(1.0) is True
    assert _graded(1.0 - 1e-9) is True  # within tolerance
    assert _graded(0.999) is False
    assert _graded(0.0) is False


def test_grade_reports_reward_as_value_and_detail():
    task = Task(task_id="airline_0", prompt="", metadata={})
    result = RunResult(output="", session=Session())
    state = result.session.state(Tau2State)
    state.reward, state.steps, state.done = 0.5, 4, True
    score = Tau2Bench().grade(task, result)
    assert score.value == 0.5 and score.passed is False
    assert score.detail == {"reward": 0.5, "steps": 4, "done": True}
