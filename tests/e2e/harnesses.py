from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import ClassVar

from harness import NexusAIHarness
from plugins.guards import BudgetGuard, MaxIterations, Timeout
from plugins.hooks import CostCounter, ElapsedTime, IterationCounter
from plugins.loops import AgenticLoop
from plugins.mcp import MCPClient, MCPServer
from plugins.memory import FileMemoryStore
from plugins.permissions import AutoApprove, DenyList
from plugins.sandbox import WorkspaceSandbox
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
from protocols.context import Context
from protocols.model import Model
from protocols.skill import Skill
from protocols.tool import Tool

HarnessBuilder = Callable[[Model], NexusAIHarness]  # model under test -> harness

_BUILDERS: dict[str, HarnessBuilder] = {}


def e2e_harness(name: str) -> Callable[[HarnessBuilder], HarnessBuilder]:
    """Register a harness builder under ``name`` for JSON cases to reference.

    A builder registers the loop, tools, and guards and returns an unstarted
    harness. It takes the model as an argument (do not register one) and must
    not add the metrics recorder — the runner supplies both.
    """

    def register(fn: HarnessBuilder) -> HarnessBuilder:
        if name in _BUILDERS:
            raise ValueError(f"e2e harness {name!r} is already registered")
        _BUILDERS[name] = fn
        return fn

    return register


def build_harness(name: str, model: Model) -> NexusAIHarness:
    """Compose the harness registered under ``name`` around ``model``."""
    try:
        builder = _BUILDERS[name]
    except KeyError:
        known = ", ".join(sorted(_BUILDERS)) or "(none)"
        raise KeyError(f"unknown e2e harness {name!r}; registered: {known}") from None
    return builder(model)


def registered_harnesses() -> dict[str, HarnessBuilder]:
    """A copy of the name -> builder mapping."""
    return dict(_BUILDERS)


# --- tools available to the default harness --------------------------------


class Calculator(Tool):
    """Evaluate a basic arithmetic expression."""

    name = "calculator"
    description = "Evaluate an arithmetic expression, e.g. '2 * (3 + 4)'."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The expression."}
        },
        "required": ["expression"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        expr = str(arguments.get("expression", ""))
        allowed = set("0123456789+-*/(). %")
        if not expr or set(expr) - allowed:
            return "error: only basic arithmetic is supported"
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))  # input is sandboxed
        except Exception as exc:
            return f"error: {exc}"


class Echo(Tool):
    """Return the ``text`` argument unchanged — a trivial tool for wiring tests."""

    name = "echo"
    description = "Echo the given text back."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, arguments: dict, ctx: Context) -> str:
        return str(arguments.get("text", ""))


# --- builders ---------------------------------------------------------------


def _base(model: Model) -> NexusAIHarness:
    """Compose an agentic loop with calculator/echo tools, auto-approval, guards."""
    return (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(model)
        .use(Calculator())
        .use(Echo())
        # Non-interactive: approve every tool ask so nothing blocks on stdin.
        .use(AutoApprove())
        # Counters feed the matching guards (a runaway case still terminates).
        .use(IterationCounter())
        .use(ElapsedTime())
        .use(CostCounter())
        .use(MaxIterations(10))
        .use(Timeout(120))
        .use(BudgetGuard(1.0))
    )


@e2e_harness("default")
def _default(model: Model) -> NexusAIHarness:
    return _base(model)


@e2e_harness("tiny_budget")
def _tiny_budget(model: Model) -> NexusAIHarness:
    """Cap the budget so low any real model call trips the guard."""
    return _base(model).replace(BudgetGuard, BudgetGuard(1e-9))


@e2e_harness("one_iteration")
def _one_iteration(model: Model) -> NexusAIHarness:
    """Cap at a single iteration so a multi-step task is stopped by the guard."""
    return _base(model).replace(MaxIterations, MaxIterations(1))


@e2e_harness("deny_calculator")
def _deny_calculator(model: Model) -> NexusAIHarness:
    """Block the calculator at the permission layer, for testing tool denial."""
    return _base(model).use(DenyList(["calculator"]))


@e2e_harness("memory")
def _memory(model: Model) -> NexusAIHarness:
    """Add long-term memory tools over an isolated temp store."""
    directory = tempfile.mkdtemp(prefix="nexus-e2e-mem-")
    return (
        _base(model)
        .use(FileMemoryStore(directory))
        .use(Remember())
        .use(Recall())
        .use(Forget())
    )


@e2e_harness("subagent")
def _subagent(model: Model) -> NexusAIHarness:
    """Add subagent delegation; the child inherits all parent plugins."""
    return _base(model).use(Subagent()).use(InProcessSpawner())


class _PirateSkill(Skill):
    """A tiny code-authored skill for exercising the skill tool end to end."""

    name = "pirate-speak"
    description = "Respond in exaggerated pirate dialect."

    def instructions(self) -> str:
        return "Respond only as a pirate. Begin your reply with the word ARRR."


@e2e_harness("skills")
def _skills(model: Model) -> NexusAIHarness:
    """Register one skill — a skill is a tool, so plain .use() is all it takes."""
    return _base(model).use(_PirateSkill())


@e2e_harness("filesystem")
def _filesystem(model: Model) -> NexusAIHarness:
    """Add real filesystem/shell tools confined to an isolated temp workspace."""
    root = tempfile.mkdtemp(prefix="nexus-e2e-fs-")
    return (
        _base(model)
        .use(WorkspaceSandbox(root))
        .use(ReadFile())
        .use(WriteFile())
        .use(ListDir())
        .use(Shell())
    )


class _CannedMCPSession:
    """A fake MCP session for e2e: one 'greet' tool, no subprocess or socket."""

    async def list_tools(self) -> list[dict]:
        return [
            {
                "name": "greet",
                "description": "Greet someone by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            }
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        return f"Hello, {arguments.get('name', 'world')}! (via MCP {name})"


async def _canned_connector(server: MCPServer) -> _CannedMCPSession:
    return _CannedMCPSession()


@e2e_harness("mcp")
def _mcp(model: Model) -> NexusAIHarness:
    """Expose a fake MCP server's tools through the loop via an injected connector."""
    client = MCPClient(
        [MCPServer(name="demo", command=["unused"])],
        connector=_canned_connector,
    )
    return _base(model).use(client)
