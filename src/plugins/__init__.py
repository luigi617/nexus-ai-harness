from __future__ import annotations

from harness import NexusAIHarness
from plugins.context_manager import SummarizingContextManager
from plugins.guards import BudgetGuard, MaxIterations, Timeout
from plugins.hooks import CostCounter, ElapsedTime, IterationCounter
from plugins.interventions import InjectMessage
from plugins.loops import AgenticLoop, ChatLoop
from plugins.memory import FileMemoryStore
from plugins.models import BedrockModel
from plugins.permissions import (
    AllowList,
    AskUnless,
    AutoApprove,
    ConsoleApprover,
    DenyList,
)
from plugins.routers import LLMRouter, StickyRouter
from plugins.spawner import InProcessSpawner
from plugins.tools import Forget, Recall, Remember, Subagent
from plugins.tracers import GraphTracer
from protocols.model import Model

__all__ = [
    "AgenticLoop",
    "AllowList",
    "AskUnless",
    "AutoApprove",
    "BedrockModel",
    "BudgetGuard",
    "ChatLoop",
    "ConsoleApprover",
    "CostCounter",
    "DenyList",
    "ElapsedTime",
    "FileMemoryStore",
    "Forget",
    "GraphTracer",
    "InProcessSpawner",
    "InjectMessage",
    "IterationCounter",
    "LLMRouter",
    "MaxIterations",
    "Recall",
    "Remember",
    "StickyRouter",
    "Subagent",
    "SummarizingContextManager",
    "Timeout",
    "default_harness",
]


def default_harness(
    model: Model,
    *,
    max_iterations: int = 20,
    timeout_s: float = 300.0,
    max_cost: float = 5.0,
    memory_dir: str = "~/.nexus-ai-harness/memory",
) -> NexusAIHarness:
    """A batteries-included harness built around the given Model.

    Wires an agentic loop and context summarization, plus long-term memory,
    subagent delegation, safety guards, and observability counters.
    """
    return (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(SummarizingContextManager())
        .use(model)
        # long-term memory + its tools
        .use(FileMemoryStore(memory_dir))
        .use(Remember())
        .use(Recall())
        # subagent delegation (the tool + the spawner it runs on)
        .use(Subagent())
        .use(InProcessSpawner())
        # permissions: trust local memory ops; ask the human for anything else
        .use(AskUnless(["remember", "recall"]))
        .use(ConsoleApprover())
        # observability — these also feed the guards below
        .use(IterationCounter())
        .use(ElapsedTime())
        .use(CostCounter())
        # safety rails: a run can't loop, hang, or overspend unbounded
        .use(MaxIterations(max_iterations))
        .use(Timeout(timeout_s))
        .use(BudgetGuard(max_cost))
    )
