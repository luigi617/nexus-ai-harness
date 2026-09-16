from __future__ import annotations

from harness import GraphAIHarness
from plugins.context_manager import SummarizingContextManager
from plugins.guards import BudgetGuard, MaxIterations, Timeout
from plugins.hooks import CostCounter, ElapsedTime, IterationCounter
from plugins.loops import AgenticLoop, ChatLoop
from plugins.memory import FileMemoryStore
from plugins.permissions import (
    AllowList,
    AskUnless,
    AutoApprove,
    ConsoleApprover,
    DenyList,
)
from plugins.providers import BedrockProvider
from plugins.spawner import InProcessSpawner
from plugins.tools import Forget, Recall, Remember, Subagent
from plugins.tracers import GraphTracer
from protocols.provider import Provider

__all__ = [
    "AgenticLoop",
    "AllowList",
    "AskUnless",
    "AutoApprove",
    "BedrockProvider",
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
    "IterationCounter",
    "MaxIterations",
    "Recall",
    "Remember",
    "Subagent",
    "SummarizingContextManager",
    "Timeout",
    "default_harness",
]


def default_harness(
    provider: Provider,
    *,
    max_iterations: int = 20,
    timeout_s: float = 300.0,
    max_cost: float = 5.0,
    memory_dir: str = "~/.nexus-ai-harness/memory",
) -> GraphAIHarness:
    """A batteries-included harness: agentic loop + context summarization + the
    given provider, plus long-term memory, subagent delegation, safety guards,
    and observability counters.

    Interactive by default: memory reads/writes run freely, but any other tool
    call (subagent, or tools you add) prompts on the terminal via
    ConsoleApprover. For non-interactive/automated use, override the approver
    with ``.use(AutoApprove())`` (last registration wins).
    """
    return (
        GraphAIHarness()
        .use(AgenticLoop())
        .use(SummarizingContextManager())
        .use(provider)
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
