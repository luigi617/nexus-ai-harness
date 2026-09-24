from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar

from tau_bench.envs import get_env
from tau_bench.types import Action

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.registry import register
from benchmarks.core.task import Score, Task
from harness import NexusAIHarness
from harness.result import RunResult
from harness.session import Session
from plugins.guards import MaxIterations, Timeout
from plugins.hooks import ElapsedTime, IterationCounter
from plugins.loops import AgenticLoop
from plugins.permissions import AutoApprove
from protocols.context import Context
from protocols.model import Model
from protocols.tool import Tool

_MAX_STEPS = int(os.getenv("TAU_MAX_STEPS", "30"))


@dataclass
class TauState:
    """Latest env feedback for the grader, written during the run."""

    reward: float = 0.0
    done: bool = False
    steps: int = 0
    extra: dict = field(default_factory=dict)


def _make_env(task_index: int | None) -> Any:
    """Construct a tau-bench env for the configured domain."""
    return get_env(
        os.getenv("TAU_ENV", "retail"),
        user_strategy=os.getenv("TAU_USER_STRATEGY", "llm"),
        user_model=os.getenv("TAU_USER_MODEL", "gpt-4o"),
        user_provider=os.getenv("TAU_USER_PROVIDER", "openai"),
        task_split=os.getenv("TAU_TASK_SPLIT", "test"),
        task_index=task_index,
    )


def _record(state: TauState, resp: Any) -> None:
    """Fold an env response into ``state`` for the grader.

    Once the episode is ``done`` the terminal reading is latched: a late step
    (e.g. a read tool that ran concurrently with a terminating action in the
    same turn) must not reset ``done``/``reward`` to their non-terminal values.
    """
    state.steps += 1
    if state.done:
        return
    state.reward = float(getattr(resp, "reward", 0.0) or 0.0)
    state.done = bool(getattr(resp, "done", False))
    state.extra = dict(getattr(resp, "info", {}) or {})


def _step_env(env: Any, action: Any, lock: threading.Lock, state: TauState) -> Any:
    """Step the env and record the result atomically.

    tau-bench envs are stateful and not thread-safe; the agent loop may run a
    turn's tool calls concurrently on worker threads, so serialize env mutation
    and recording under a per-env lock.
    """
    with lock:
        resp = env.step(action)
        _record(state, resp)
    return resp


def _respond(
    env: Any, lock: threading.Lock, state: TauState, message: str
) -> str | None:
    """Relay the agent's message to the simulated user.

    Returns the user's reply, or ``None`` once the conversation is over — the
    signal the runner's conversation driver uses to stop the episode.
    """
    resp = _step_env(
        env, Action(name="respond", kwargs={"content": message}), lock, state
    )
    return None if getattr(resp, "done", False) else str(resp.observation)


class _EnvTool(Tool):
    """A tau-bench domain tool: ``run`` steps the env and returns the observation."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[dict] = {}

    def __init__(self, spec: dict, env: Any, lock: threading.Lock) -> None:
        fn = spec["function"]
        self.name = fn["name"]
        self.description = fn.get("description", "")
        self.parameters = fn.get("parameters", {"type": "object", "properties": {}})
        self._env = env
        self._lock = lock

    def run(self, arguments: dict, ctx: Context) -> str:
        resp = _step_env(
            self._env,
            Action(name=self.name, kwargs=arguments),
            self._lock,
            ctx.state(TauState),
        )
        return str(resp.observation)


@register
class TauBench(Benchmark):
    """The tau-bench suite: multi-turn tool use against a simulated user.

    The production ``AgenticLoop`` is the agent under test; the runner's
    conversation driver relays each agent message to tau-bench's simulated user
    (via ``Episode.on_user_turn``) until the env reports the episode done.
    """

    name = "tau-bench"
    description = "Multi-turn tool use with a simulated user, graded by DB state"

    def load_tasks(self, *, limit: int | None = None) -> list[Task]:
        probe = _make_env(task_index=0)
        count = len(probe.tasks)
        indices = range(count if limit is None else min(limit, count))
        return [
            Task(
                task_id=f"{os.getenv('TAU_ENV', 'retail')}_{i}",
                prompt="",  # the opening user message comes from env.reset()
                metadata={"task_index": i},
            )
            for i in indices
        ]

    def build_harness(self, task: Task, model: Model, session: Session) -> Episode:
        env = _make_env(task_index=task.metadata["task_index"])
        reset = env.reset(task_index=task.metadata["task_index"])
        lock = threading.Lock()  # serializes concurrent tool steps on this env
        # Seed the run with the domain policy (wiki) and the user's opening
        # message. Returned to the runner rather than written onto the shared
        # ``Task`` (which is reused across all k attempts).
        prompt = (
            f"{getattr(env, 'wiki', '')}\n\n"
            "You are a customer-service agent. Use the tools to resolve the "
            "user's request. To talk to the user, reply in plain text (with no "
            "tool call); that message is sent to them and their reply comes back "
            "on your next turn. The task ends when their request is fully "
            f"resolved.\n\nUser: {reset.observation}"
        )
        harness = (
            NexusAIHarness()
            .use(AgenticLoop())  # the real production loop under test
            .use(model)
            .use(AutoApprove())
            .use(IterationCounter())
            .use(ElapsedTime())
            .use(MaxIterations(_MAX_STEPS))  # bounds tool steps per agent turn
            .use(Timeout(float(os.getenv("TAU_TIMEOUT_S", "300"))))
        )
        for spec in env.tools_info:
            harness.use(_EnvTool(spec, env, lock))

        # AgenticLoop returns its final text (a message to the user); relay it to
        # the env and hand the user's reply back to the driver as the next input.
        def on_user_turn(agent_output: str) -> str | None:
            return _respond(env, lock, session.state(TauState), agent_output)

        return Episode(harness=harness, initial_input=prompt, on_user_turn=on_user_turn)

    def grade(self, task: Task, result: RunResult) -> Score:
        state = result.session.state(TauState)
        return Score(
            passed=state.reward >= 1.0 - 1e-6,
            value=state.reward,
            detail={"reward": state.reward, "steps": state.steps, "done": state.done},
        )
