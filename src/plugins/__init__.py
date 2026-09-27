from __future__ import annotations

from collections.abc import Iterable

from harness import NexusAIHarness
from plugins.context_manager import SummarizingContextManager
from plugins.evaluators import JevEvaluator
from plugins.guards import BudgetGuard, MaxIterations, Timeout
from plugins.hooks import CostCounter, ElapsedTime, IterationCounter
from plugins.interventions import InjectMessage
from plugins.loops import AgenticLoop, ChatLoop
from plugins.mcp import MCPClient, MCPServer
from plugins.memory import FileMemoryStore
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
from plugins.permissions import (
    AllowList,
    AskUnless,
    AutoApprove,
    ConsoleApprover,
    DenyList,
)
from plugins.persistence import AutoSave, FileSessionStore, resume
from plugins.routers import LLMRouter, StickyRouter
from plugins.sandbox import WorkspaceSandbox
from plugins.skills import MarkdownSkill, load_skills
from plugins.spawner import InProcessSpawner
from plugins.tools import (
    Forget,
    ListDir,
    ReadFile,
    Recall,
    Remember,
    Shell,
    Subagent,
    WriteFile,
)
from plugins.tracers import GraphTracer
from protocols.model import Model
from protocols.session_store import SessionStore

__all__ = [
    "AgenticLoop",
    "AllowList",
    "AnthropicModel",
    "AskUnless",
    "AutoApprove",
    "AutoSave",
    "BedrockModel",
    "BudgetGuard",
    "ChatLoop",
    "ConsoleApprover",
    "CostCounter",
    "DeepSeekModel",
    "DenyList",
    "ElapsedTime",
    "FileMemoryStore",
    "FileSessionStore",
    "Forget",
    "GLMModel",
    "GeminiModel",
    "GraphTracer",
    "GroqModel",
    "InProcessSpawner",
    "InjectMessage",
    "IterationCounter",
    "JevEvaluator",
    "LLMRouter",
    "ListDir",
    "MCPClient",
    "MCPServer",
    "MarkdownSkill",
    "MaxIterations",
    "MiniMaxModel",
    "OpenAICompatibleModel",
    "OpenAIModel",
    "QwenModel",
    "ReadFile",
    "Recall",
    "Remember",
    "Shell",
    "StickyRouter",
    "Subagent",
    "SummarizingContextManager",
    "Timeout",
    "WorkspaceSandbox",
    "WriteFile",
    "XAIModel",
    "default_harness",
    "load_skills",
    "resume",
]


def default_harness(
    model: Model,
    *,
    max_iterations: int = 20,
    timeout_s: float = 300.0,
    max_cost: float = 5.0,
    memory_dir: str = "~/.nexus-ai-harness/memory",
    workspace: str | None = None,
    session_store: SessionStore | None = None,
    mcp_servers: Iterable[MCPServer] | None = None,
) -> NexusAIHarness:
    """A batteries-included harness built around the given Model.

    Wires an agentic loop and context summarization, plus long-term memory,
    subagent delegation, safety guards, and observability counters. Add skills
    with ``.use(skill)`` (or ``.use(s) for s in load_skills(dir)``) like any
    other plugin.

    Args:
        model: The model that drives the loop.
        max_iterations: Hard cap on loop iterations before the run stops.
        timeout_s: Wall-clock budget for a run, in seconds.
        max_cost: Spend budget for a run, in dollars.
        memory_dir: Directory backing the long-term memory store.
        workspace: When set, register a :class:`WorkspaceSandbox` rooted here
            plus the real ``read_file``/``write_file``/``list_dir``/``shell``
            tools confined to it. The read-only tools (``read_file``,
            ``list_dir``) are trusted; ``write_file`` and ``shell`` mutate state
            and still ask for approval.
        session_store: When set, register it plus an :class:`AutoSave` hook so
            runs are persisted and can be resumed with :func:`resume`.
        mcp_servers: When set, register an :class:`MCPClient` that exposes the
            configured MCP servers' tools to the loop.

    Returns:
        A configured, unstarted harness.
    """
    trusted = ["remember", "recall"]
    if workspace is not None:
        # Only the read-only tools are auto-trusted; write_file mutates files and
        # shell runs commands, so both stay gated behind the approver.
        trusted += ["read_file", "list_dir"]

    harness = (
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
    )
    # optional OS sandbox + real filesystem/shell tools (sandbox first: the
    # tools resolve it as their dependency)
    if workspace is not None:
        harness.use(WorkspaceSandbox(workspace))
        harness.use(ReadFile())
        harness.use(WriteFile())
        harness.use(ListDir())
        harness.use(Shell())
    # optional MCP client contributing remote tools to the loop
    if mcp_servers is not None:
        harness.use(MCPClient(mcp_servers))
    # optional persistence: the store before the AutoSave hook that requires it
    if session_store is not None:
        harness.use(session_store)
        harness.use(AutoSave())
    return (
        harness
        # permissions: trust local memory ops; ask the human for anything else
        .use(AskUnless(trusted))
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
