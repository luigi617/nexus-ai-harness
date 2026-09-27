from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

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

_MAX_STEPS = int(os.getenv("TAU2_MAX_STEPS", "30"))


@dataclass
class Tau2State:
    """Latest env feedback for the grader, written during the run."""

    reward: float = 0.0
    done: bool = False
    steps: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class _EnvResponse:
    """Uniform result of one env interaction, read by the relay/record logic.

    tau2 splits tool execution (the Environment) from the user turn (a separate
    UserSimulator) and computes the reward only once, at the end. The session
    adapter normalizes all three into this one shape so ``_record``/``_respond``
    stay domain- and tau2-version-agnostic (and unit-testable with a fake).

    Attributes:
        observation: The text handed back to the agent (tool output or user
            reply).
        reward: The task reward; non-zero only on the terminal turn.
        done: Whether the simulated user has ended the conversation.
        info: Extra diagnostics (tool error flag, termination reason).
    """

    observation: str = ""
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


# --- domain-agnostic relay / record / grade core --------------------------


def _record(state: Tau2State, resp: Any) -> None:
    """Fold an env response into ``state`` for the grader.

    Once the episode is ``done`` the terminal reading is latched: a late step
    (e.g. a read tool that ran concurrently with the terminating user turn) must
    not reset ``done``/``reward`` back to their non-terminal values.
    """
    state.steps += 1
    if state.done:
        return
    state.reward = float(getattr(resp, "reward", 0.0) or 0.0)
    state.done = bool(getattr(resp, "done", False))
    state.extra = dict(getattr(resp, "info", {}) or {})


def _use_tool(
    session: Any,
    name: str,
    arguments: dict,
    lock: threading.Lock,
    state: Tau2State,
) -> Any:
    """Run an agent tool against the env and record the result atomically.

    tau2 envs are stateful and not thread-safe; the agent loop may run a turn's
    tool calls concurrently on worker threads, so serialize env mutation and
    recording under a per-env lock.
    """
    with lock:
        resp = session.call_tool(name, arguments)
        _record(state, resp)
    return resp


def _respond(
    session: Any, lock: threading.Lock, state: Tau2State, message: str
) -> str | None:
    """Relay the agent's message to the simulated user, recording the result.

    Returns the user's reply, or ``None`` once the conversation is over — the
    signal the runner's conversation driver uses to stop the episode.
    """
    with lock:
        resp = session.respond(message)
        _record(state, resp)
    return None if getattr(resp, "done", False) else str(resp.observation)


class _EnvTool(Tool):
    """A tau2 domain tool: ``run`` executes it on the env, returns the output."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[dict] = {}

    def __init__(self, spec: dict, session: Any, lock: threading.Lock) -> None:
        fn = spec["function"]  # tau2 Tool.openai_schema wraps under "function"
        self.name = fn["name"]
        self.description = fn.get("description", "")
        self.parameters = fn.get("parameters", {"type": "object", "properties": {}})
        self._session = session
        self._lock = lock

    def run(self, arguments: dict, ctx: Context) -> str:
        resp = _use_tool(
            self._session, self.name, arguments, self._lock, ctx.state(Tau2State)
        )
        return str(resp.observation)


# --- tau2 adapter ---------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class _Tau2Session:
    """Adapts one tau2 task (Environment + UserSimulator + evaluator) for a run.

    Presents the uniform ``call_tool`` / ``respond`` surface the harness-side
    relay logic depends on, hiding every tau2-specific detail.
    """

    def __init__(self, env: Any, user: Any, task: Any, domain: str) -> None:
        self._env = env
        self._user = user
        self._task = task
        self._domain = domain
        self._trajectory: list[Any] = []
        self.policy = env.get_policy()
        # tau2 Tool.openai_schema is a property shaped {"function": {...}}.
        self.tool_specs = [t.openai_schema for t in env.get_tools()]
        self.opening_message = self._open()

    def _open(self) -> str:
        """Elicit the simulated user's opening request to seed the run.

        Assumption (verify against the installed tau2): the user simulator
        speaks first when prompted with a neutral assistant greeting. The
        greeting is a scaffold, not agent output, so it is kept out of the
        graded trajectory.
        """
        self._user_state = self._user.get_init_state()
        greeting = AssistantMessage(
            role="assistant", content="Hi! How can I help you today?"
        )
        user_msg, self._user_state = self._user.generate_next_message(
            greeting, self._user_state
        )
        self._trajectory.append(user_msg)
        return user_msg.content or ""

    def call_tool(self, name: str, arguments: dict) -> _EnvResponse:
        """Execute an assistant tool call and return its output."""
        tool_call = ToolCall(
            id=uuid.uuid4().hex,
            name=name,
            arguments=dict(arguments),
            requestor="assistant",
        )
        self._trajectory.append(
            AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])
        )
        tool_msg = self._env.get_response(tool_call)  # runs the tool, syncs env state
        self._trajectory.append(tool_msg)
        return _EnvResponse(
            observation=str(tool_msg.content or ""),
            info={"error": bool(getattr(tool_msg, "error", False))},
        )

    def respond(self, message: str) -> _EnvResponse:
        """Relay the agent's message to the user simulator and return its reply.

        On the turn where the user signals stop, the task reward is computed
        over the recorded trajectory and latched onto the response.
        """
        assistant_msg = AssistantMessage(role="assistant", content=message)
        self._trajectory.append(assistant_msg)
        user_msg, self._user_state = self._user.generate_next_message(
            assistant_msg, self._user_state
        )
        self._trajectory.append(user_msg)
        observation = str(user_msg.content or "")
        if not type(self._user).is_stop(user_msg):
            return _EnvResponse(observation=observation)
        return _EnvResponse(
            observation=observation,
            reward=self._reward(),
            done=True,
            info={"termination": "user_stop"},
        )

    def _reward(self) -> float:
        """Score the finished trajectory with tau2's evaluator.

        Assumption (most likely to need adjusting to the installed tau2): a
        minimal ``SimulationRun`` over the recorded messages, scored with the
        ``ALL`` evaluator, yields the same reward the reference harness would.
        Adjust the SimulationRun fields / EvaluationType here if a version
        mismatch surfaces.
        """
        now = _now_iso()
        sim = SimulationRun(
            id=uuid.uuid4().hex,
            task_id=self._task.id,
            start_time=now,
            end_time=now,
            duration=0.0,
            termination_reason=TerminationReason.USER_STOP,
            messages=list(self._trajectory),
        )
        info = evaluate_simulation(
            simulation=sim,
            task=self._task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=self._domain,
        )
        return float(info.reward)


def _load_tau2_tasks(domain: str, split: str | None) -> list[Any]:
    """Load a tau2 domain's tasks, optionally restricted to ``split``."""
    try:
        loader = registry.get_tasks_loader(domain)
    except KeyError as exc:
        known = ", ".join(registry.get_task_sets())
        raise RuntimeError(
            f"unknown tau2 task set {domain!r}; set TAU2_DOMAIN to one of: {known}"
        ) from exc
    return loader(split) if split else loader()


def _build_session(domain: str, task_index: int, split: str | None) -> _Tau2Session:
    """Construct the per-task tau2 environment, user simulator, and adapter."""
    try:
        env = registry.get_env_constructor(domain)()
    except KeyError as exc:
        known = ", ".join(registry.get_domains())
        raise RuntimeError(
            f"unknown tau2 domain {domain!r}; set TAU2_DOMAIN to one of: {known}"
        ) from exc
    task = _load_tau2_tasks(domain, split)[task_index]
    user = UserSimulator(
        llm=os.getenv("TAU2_USER_MODEL", "gpt-4o"),
        instructions=str(task.user_scenario),
    )
    return _Tau2Session(env=env, user=user, task=task, domain=domain)


@register
class Tau2Bench(Benchmark):
    """The tau2-bench suite: dual-control multi-turn tool use, graded by reward.

    The successor to tau-bench. The production ``AgenticLoop`` is the agent under
    test; the runner's conversation driver relays each agent message to tau2's
    simulated user (via ``Episode.on_user_turn``) until the user ends the
    conversation, at which point tau2's evaluator scores the trajectory.
    """

    name = "tau2-bench"
    description = (
        "tau2-bench: dual-control multi-turn tool use with a simulated user, "
        "graded by task reward"
    )

    def load_tasks(self, *, limit: int | None = None) -> list[Task]:
        domain = os.getenv("TAU2_DOMAIN", "airline")
        split = os.getenv("TAU2_TASK_SPLIT") or None
        count = len(_load_tau2_tasks(domain, split))
        indices = range(count if limit is None else min(limit, count))
        return [
            Task(
                task_id=f"{domain}_{i}",
                prompt="",  # the opening user message is generated in build_harness
                metadata={"task_index": i, "domain": domain, "split": split},
            )
            for i in indices
        ]

    def build_harness(self, task: Task, model: Model, session: Session) -> Episode:
        meta = task.metadata
        env = _build_session(meta["domain"], meta["task_index"], meta.get("split"))
        lock = threading.Lock()  # serializes concurrent tool steps on this env
        # Seed with the domain policy + user's opening message; returned to the
        # runner, not written onto the shared Task (reused across k attempts).
        prompt = (
            f"{env.policy}\n\n"
            "You are a customer-service agent. Use the tools to resolve the "
            "user's request. To talk to the user, reply in plain text (with no "
            "tool call); that message is sent to them and their reply comes back "
            "on your next turn. The task ends when their request is fully "
            f"resolved.\n\nUser: {env.opening_message}"
        )
        harness = (
            NexusAIHarness()
            .use(AgenticLoop())  # the real production loop under test
            .use(model)
            .use(AutoApprove())
            .use(IterationCounter())
            .use(ElapsedTime())
            .use(MaxIterations(_MAX_STEPS))  # bounds tool steps per agent turn
            .use(Timeout(float(os.getenv("TAU2_TIMEOUT_S", "300"))))
        )
        for spec in env.tool_specs:
            harness.use(_EnvTool(spec, env, lock))

        def on_user_turn(agent_output: str) -> str | None:
            return _respond(env, lock, session.state(Tau2State), agent_output)

        return Episode(harness=harness, initial_input=prompt, on_user_turn=on_user_turn)

    def grade(self, task: Task, result: RunResult) -> Score:
        state = result.session.state(Tau2State)
        return Score(
            passed=state.reward >= 1.0 - 1e-6,
            value=state.reward,
            detail={"reward": state.reward, "steps": state.steps, "done": state.done},
        )
