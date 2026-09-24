from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

from core.events import Event, ResponseReceived, ToolCallCompleted, ToolCallDenied
from core.message import Message
from core.response import Response
from core.run import RunState
from core.spawn import SpawnState
from harness import NexusAIHarness
from harness.session import Session
from plugins.models import (
    AnthropicModel,
    BedrockModel,
    DeepSeekModel,
    GeminiModel,
    GLMModel,
    GroqModel,
    MiniMaxModel,
    OpenAICompatibleModel,
    OpenAIModel,
    QwenModel,
    XAIModel,
)
from protocols.context import Context
from protocols.hook import Hook
from protocols.model import Model
from tests.e2e.harnesses import build_harness
from tests.e2e.spec import Spec

_PROVIDERS: dict[str, type[Model]] = {
    "bedrock": BedrockModel,
    "anthropic": AnthropicModel,
    "openai": OpenAIModel,
    "openai_compatible": OpenAICompatibleModel,
    "deepseek": DeepSeekModel,
    "gemini": GeminiModel,
    "glm": GLMModel,
    "groq": GroqModel,
    "minimax": MiniMaxModel,
    "qwen": QwenModel,
    "xai": XAIModel,
}


class ScriptedModel(Model):
    """Replays queued responses in order; the last one repeats when exhausted.

    Deterministic stand-in for a real backend, so a case's control flow (tool
    dispatch, guards, stop reasons) can be asserted exactly with no API cost.
    """

    name = "scripted"

    def __init__(self, responses: list[Response]) -> None:
        self._responses = responses or [Response(text="done")]
        self._calls = 0

    async def complete(self, history: list[Message], ctx: Context) -> Response:
        idx = min(self._calls, len(self._responses) - 1)
        self._calls += 1
        return self._responses[idx]


def _scripted_model(spec: Spec) -> ScriptedModel:
    responses = [
        Response(
            text=r.get("text", ""),
            tool_calls=[
                {"id": str(i), **call} if "id" not in call else call
                for i, call in enumerate(r.get("tool_calls", []))
            ],
            cost=r.get("cost", 0.0),
            usage=r.get("usage", {}),
        )
        for r in spec.scripted_responses
    ]
    return ScriptedModel(responses)


def build_model(spec: Spec) -> Model:
    """Build the scripted model, or a live backend from ``provider:model-id``."""
    if not spec.is_live:
        return _scripted_model(spec)
    provider, _, model_id = spec.model.partition(":")
    if not model_id:  # a bare id means the default provider
        provider, model_id = "bedrock", spec.model
    if provider not in _PROVIDERS:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"unknown provider {provider!r}; known: {known}")
    return _PROVIDERS[provider](model=model_id, max_tokens=2048)


class _Recorder(Hook):
    """Captures the trajectory facts the assertions read off a finished run.

    Only records parent-conversation events. ``ctx.fork`` inherits every parent
    plugin by reference, so a subagent's child context would otherwise fire this
    same recorder and inflate model_calls/cost/tool_calls with the child's
    internal steps — which the Subagent contract keeps off the parent. We gate on
    ``SpawnState.depth`` (0 at the root, incremented per fork) to ignore them.
    """

    def __init__(self) -> None:
        self.tool_calls: list[dict[str, Any]] = []
        self.denied: list[dict[str, Any]] = []
        self.model_calls = 0
        self.cost = 0.0

    def on(self, event: Event, ctx: Context) -> None:
        if ctx.state(SpawnState).depth != 0:  # ignore subagent (forked) contexts
            return
        if isinstance(event, ResponseReceived):
            self.model_calls += 1
            self.cost += event.response.cost
        elif isinstance(event, ToolCallCompleted):
            self.tool_calls.append(event.call)
        elif isinstance(event, ToolCallDenied):
            self.denied.append(event.call)


@dataclass
class CaseResult:
    """Everything the assertions need about one finished case.

    ``output`` is the last turn's final text; ``outputs`` is every turn's final
    text. Text assertions match against ``transcript`` (all turns joined) so a
    fact answered in an earlier turn of a multi-turn case still counts.
    """

    output: str = ""
    outputs: list[str] = field(default_factory=list)
    stop_reason: str = ""
    cost: float = 0.0
    model_calls: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    denied: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def transcript(self) -> str:
        return "\n".join(self.outputs)


async def run_case(spec: Spec) -> CaseResult:
    """Compose the harness, drive the conversation, and snapshot the outcome.

    Never raises for an in-run failure: a caught exception is returned on
    ``CaseResult.error`` so the ``no_error`` assertion can report it.
    """
    # patch.dict restores os.environ on exit, dropping keys the case added.
    with mock.patch.dict(os.environ, spec.env):
        model = build_model(spec)
        harness = build_harness(spec.harness, model)
        recorder = _Recorder()
        harness.use(recorder)
        session = Session()

        try:
            outputs = await asyncio.wait_for(
                _drive(harness, session, spec),
                timeout=spec.timeout_s,
            )
            return CaseResult(
                output=outputs[-1] if outputs else "",
                outputs=outputs,
                stop_reason=_session_stop_reason(session),
                cost=recorder.cost,
                model_calls=recorder.model_calls,
                tool_calls=recorder.tool_calls,
                denied=recorder.denied,
            )
        except Exception as exc:  # surfaced to the no_error assertion
            return CaseResult(
                stop_reason=_session_stop_reason(session),
                cost=recorder.cost,
                model_calls=recorder.model_calls,
                tool_calls=recorder.tool_calls,
                denied=recorder.denied,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            await harness.stop()


async def _drive(harness: NexusAIHarness, session: Session, spec: Spec) -> list[str]:
    """Drive the initial input plus any follow-up turns; return each turn's text."""
    outputs = [(await harness.run(spec.input, session=session)).output]
    for turn in spec.user_turns:
        outputs.append((await harness.run(turn, session=session)).output)
    return outputs


def _session_stop_reason(session: Session) -> str:
    return session.state(RunState).stop_reason
